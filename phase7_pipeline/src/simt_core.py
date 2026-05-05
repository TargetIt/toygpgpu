"""
SIMTCore — SIMT 核心 (Phase 3: + SIMT Stack + 分支发散)
=========================================================
对标 GPGPU-Sim 中 shader_core_ctx 的完整 SIMT 执行模型，
包含 simt_stack 的分支发散/重汇聚处理。

Phase 3 新增:
  - JMP/BEQ/BNE 分支指令
  - SIMT Stack 管理发散执行顺序
  - 自动重汇聚检测

参考: GPGPU-Sim gpgpu-sim/shader.cc shader_core_ctx::cycle()
      GPGPU-Sim gpgpu-sim/stack.cc simt_stack
"""

try:
    from .isa import (Instruction, decode,
                      OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                      OP_LD, OP_ST, OP_MOV,
                      OP_TID, OP_WID, OP_BAR,
                      OP_JMP, OP_BEQ, OP_BNE,
                      OP_SHLD, OP_SHST)
    from .alu import ALU
    from .memory import Memory
    from .warp import Warp
    from .scheduler import WarpScheduler
    from .simt_stack import SIMTStackEntry
    from .scoreboard import PIPELINE_LATENCY
    from .cache import L1Cache
    from .thread_block import ThreadBlock
    from .operand_collector import OperandCollector
except ImportError:
    from isa import (Instruction, decode,
                     OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                     OP_LD, OP_ST, OP_MOV,
                     OP_TID, OP_WID, OP_BAR,
                     OP_JMP, OP_BEQ, OP_BNE,
                     OP_SHLD, OP_SHST)
    from alu import ALU
    from memory import Memory
    from warp import Warp
    from scheduler import WarpScheduler
    from simt_stack import SIMTStackEntry
    from scoreboard import PIPELINE_LATENCY
    from cache import L1Cache
    from thread_block import ThreadBlock
    from operand_collector import OperandCollector

DEFAULT_WARP_SIZE = 8
DEFAULT_NUM_WARPS = 2
DEFAULT_MEM_SIZE = 1024


def popcount(x: int) -> int:
    """统计 bitmask 中置位的数量 (active thread count)"""
    return bin(x).count('1')


class SIMTCore:
    """SIMT 核心 (Phase 3: 完整 SIMT Stack 支持)

    对标 GPGPU-Sim 中 shader_core_ctx。
    """

    def __init__(self, warp_size: int = DEFAULT_WARP_SIZE,
                 num_warps: int = DEFAULT_NUM_WARPS,
                 memory_size: int = DEFAULT_MEM_SIZE):
        self.warp_size = warp_size
        self.warps = [Warp(wid, warp_size) for wid in range(num_warps)]
        self.scheduler = WarpScheduler(self.warps)
        self.memory = Memory(memory_size)      # Global DRAM
        self.l1_cache = L1Cache()               # L1 data cache
        self.op_collector = OperandCollector(4)  # 4-bank register file
        self.alu = ALU()
        self.program: list[int] = []
        self.instr_count = 0
        self.coalesce_count = 0
        self.total_mem_reqs = 0
        # Thread Block: all warps in one block share shared memory
        self.thread_block = ThreadBlock(0, self.warps, shared_mem_size=256)

    def load_program(self, program: list[int]):
        self.program = list(program)
        for w in self.warps:
            w.pc = 0
            w.done = False
            w.reset_barrier()
            w.active_mask = (1 << self.warp_size) - 1
            w.simt_stack.entries.clear()
            w.ibuffer.flush()
        self.instr_count = 0

    def step(self) -> bool:
        """执行一个周期 — Phase 7 多级流水线

        GPGPU-Sim 的逆序执行 (inverse pipeline):
          WRITEBACK → EXECUTE → READ_OPERANDS → ISSUE → DECODE → FETCH
        每周期从后往前推进。

        Phase 7 简化:
          1. advance scoreboards + release banks (writeback + operand release)
          2. ISSUE: scheduler pick warp → I-Buffer check → bank check → execute
          3. FETCH/DECODE: fill I-Buffer for warps that need instructions
        """
        # ===== 1. Advance + Release =====
        for w in self.warps:
            w.scoreboard.advance()
            if not w.scoreboard.stalled:
                w.scoreboard_stalled = False
            w._fetched_this_cycle = False
        self.op_collector.release_banks()

        # ===== 2. ISSUE + EXECUTE =====
        issued = False
        for _ in range(len(self.warps)):
            warp = self.scheduler.select_warp()
            if warp is None:
                break

            # Check reconvergence
            if warp.simt_stack.at_reconvergence(warp.pc):
                self._handle_reconvergence(warp)
                if warp.done:
                    continue

            # Check I-Buffer
            entry = warp.ibuffer.consume()
            if entry is None:
                continue

            instr = decode(entry.instruction_word)
            old_pc = entry.pc

            # Scoreboard check
            sb = warp.scoreboard
            if sb.check_waw(instr.rd) or sb.check_raw(instr.rs1, instr.rs2):
                warp.scoreboard_stalled = True
                # Put instruction back
                warp.ibuffer.write(entry.instruction_word, entry.pc)
                warp.ibuffer.set_ready(entry.pc)
                continue

            # Operand Collector bank check
            can_read, reason = self.op_collector.can_read_operands(instr.rs1, instr.rs2)
            if not can_read:
                # Bank conflict: put instruction back, try next cycle
                warp.ibuffer.write(entry.instruction_word, entry.pc)
                warp.ibuffer.set_ready(entry.pc)
                continue

            # Reserve banks and execute
            self.op_collector.reserve_banks(instr.rs1, instr.rs2)
            self._execute_warp(warp, instr, old_pc)
            self.instr_count += 1

            latency = self._get_latency(instr.opcode)
            sb.reserve(instr.rd, latency)
            issued = True
            break  # One issue per cycle

        # ===== 3. FETCH + DECODE =====
        self._fetch_decode()

        return self.scheduler.has_active_warps()

    def _fetch_decode(self):
        """Fetch: 为有 I-Buffer 空位的 warp 取指 + 译码

        对应 GPGPU-Sim 中 shader_core_ctx::fetch() + decode()。
        每个周期最多为 1 个 warp 取 1 条指令。
        """
        for warp in self.warps:
            if warp.done or warp._fetched_this_cycle:
                continue
            if not warp.ibuffer.has_free():
                continue
            if warp.pc < 0 or warp.pc >= len(self.program):
                continue

            raw_word = self.program[warp.pc]
            warp.ibuffer.write(raw_word, warp.pc)
            warp.ibuffer.set_ready(warp.pc)
            warp.pc += 1
            warp._fetched_this_cycle = True
            break

    def _handle_reconvergence(self, warp: Warp):
        """处理 SIMT Stack 重汇聚

        对应 GPGPU-Sim 中 simt_stack::pop + 路径切换逻辑。
        """
        entry = warp.simt_stack.pop()
        if entry is None:
            return

        remaining_mask = entry.orig_mask & ~entry.taken_mask
        if remaining_mask:
            warp.simt_stack.push(SIMTStackEntry(
                reconv_pc=entry.reconv_pc,
                orig_mask=entry.orig_mask,
                taken_mask=entry.taken_mask | remaining_mask,
                fallthrough_pc=entry.fallthrough_pc
            ))
            warp.active_mask = remaining_mask
            warp.pc = entry.fallthrough_pc
            warp.ibuffer.flush()
        else:
            warp.active_mask = entry.orig_mask
            warp.pc = entry.reconv_pc
            warp.ibuffer.flush()

    def _execute_warp(self, warp: Warp, instr: Instruction, old_pc: int):
        """对 warp 内所有 active 线程执行指令"""
        op = instr.opcode

        if op == OP_HALT:
            warp.done = True
            return

        if op == OP_BAR:
            if warp.at_barrier:
                warp.reset_barrier()
            return

        if op == OP_TID:
            for t in warp.active_threads():
                t.write_reg(instr.rd, t.thread_id)
            return

        if op == OP_WID:
            for t in warp.active_threads():
                t.write_reg(instr.rd, warp.warp_id)
            return

        # ---- 分支指令 (Phase 3) ----
        if op in (OP_JMP, OP_BEQ, OP_BNE):
            self._execute_branch(warp, instr, old_pc)
            return

        # ---- ALU ----
        if op == OP_ADD:
            for t in warp.active_threads():
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.add(a, b))
            return
        if op == OP_SUB:
            for t in warp.active_threads():
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.sub(a, b))
            return
        if op == OP_MUL:
            for t in warp.active_threads():
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.mul(a, b))
            return
        if op == OP_DIV:
            for t in warp.active_threads():
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.div(a, b))
            return
        if op == OP_MOV:
            for t in warp.active_threads():
                t.write_reg(instr.rd, instr.imm)
            return

        # ---- LD/ST (with L1 Cache + Coalescing) ----
        if op == OP_LD:
            self._mem_load(warp, instr)
            return
        if op == OP_ST:
            self._mem_store(warp, instr)
            return

        # ---- Shared Memory (Phase 5) ----
        if op == OP_SHLD:
            base = instr.imm & 0xFF
            smem = self.thread_block.shared_memory
            for t in warp.active_threads():
                addr = (base + t.thread_id) % smem.size_words
                t.write_reg(instr.rd, smem.read_word(addr))
            return
        if op == OP_SHST:
            base = instr.imm & 0xFF
            smem = self.thread_block.shared_memory
            for t in warp.active_threads():
                addr = (base + t.thread_id) % smem.size_words
                smem.write_word(addr, t.read_reg(instr.rs1))
            return

        raise ValueError(f"Unknown opcode: 0x{op:02X}")

    def _mem_load(self, warp, instr):
        """LD with L1 Cache + Coalescing"""
        base = instr.imm & 0x3FF
        self.total_mem_reqs += 1
        addrs = sorted((base + t.thread_id) & 0x3FF for t in warp.active_threads())

        # Coalescing: check if all addresses are contiguous
        if self._is_contiguous(addrs):
            self.coalesce_count += 1
            # One transaction: load full line
            start = addrs[0] & ~3  # align to 4-word boundary
            for i in range(start, start + 4):
                val = self.l1_cache.read(i)
                if val is None:
                    val = self.memory.read_word(i)
                    self.l1_cache.fill_line(i & ~3, [self.memory.read_word(j) for j in range(i & ~3, (i & ~3) + 4)])
            for t in warp.active_threads():
                addr = (base + t.thread_id) & 0x3FF
                val = self.l1_cache.read(addr)
                if val is None:
                    val = self.memory.read_word(addr)
                t.write_reg(instr.rd, val)
        else:
            for t in warp.active_threads():
                addr = (base + t.thread_id) & 0x3FF
                val = self.l1_cache.read(addr)
                if val is None:
                    val = self.memory.read_word(addr)
                t.write_reg(instr.rd, val)

    def _mem_store(self, warp, instr):
        """ST with L1 Cache + Coalescing"""
        base = instr.imm & 0x3FF
        self.total_mem_reqs += 1
        addrs = sorted((base + t.thread_id) & 0x3FF for t in warp.active_threads())

        if self._is_contiguous(addrs):
            self.coalesce_count += 1
        for t in warp.active_threads():
            addr = (base + t.thread_id) & 0x3FF
            val = t.read_reg(instr.rs1)
            self.l1_cache.write(addr, val)
            self.memory.write_word(addr, val)

    @staticmethod
    def _is_contiguous(addrs: list) -> bool:
        if len(addrs) <= 1:
            return True
        for i in range(1, len(addrs)):
            if addrs[i] != addrs[i-1] + 1:
                return False
        return True

    def _execute_branch(self, warp: Warp, instr: Instruction, old_pc: int):
        """执行分支指令 + 管理 SIMT Stack

        核心算法 (对标 GPGPU-Sim simt_stack::branch):
          1. 计算哪些 active 线程满足跳转条件 (taken_mask)
          2. 如果有线程跳转、有线程不跳转 → 发散
          3. Push SIMT Stack: {reconv=PC+1, orig=当前mask, taken=跳转mask}
          4. 设置 active_mask=taken_mask, PC=target
        """
        # 计算跳转目标 (相对偏移)
        op = instr.opcode
        target_pc = (old_pc + 1 + instr.imm) & 0xFFFF

        if op == OP_JMP:
            taken_mask = warp.active_mask
        elif op == OP_BEQ:
            taken_mask = 0
            for t in warp.active_threads():
                if t.read_reg(instr.rs1) == t.read_reg(instr.rs2):
                    taken_mask |= (1 << t.thread_id)
            taken_mask &= warp.active_mask
        elif op == OP_BNE:
            taken_mask = 0
            for t in warp.active_threads():
                if t.read_reg(instr.rs1) != t.read_reg(instr.rs2):
                    taken_mask |= (1 << t.thread_id)
            taken_mask &= warp.active_mask

        not_taken_mask = warp.active_mask & ~taken_mask

        # Check if this is a merging JMP within a divergent path
        is_merge_jmp = (op == OP_JMP and not warp.simt_stack.empty and
                        taken_mask == warp.active_mask)

        if not_taken_mask and taken_mask:
            warp.simt_stack.push(SIMTStackEntry(
                reconv_pc=old_pc + 1,
                orig_mask=warp.active_mask,
                taken_mask=taken_mask,
                fallthrough_pc=old_pc + 1
            ))
            warp.active_mask = taken_mask
            warp.pc = target_pc
            warp.ibuffer.flush()  # 分支后清空 I-Buffer
        elif taken_mask:
            if is_merge_jmp:
                top = warp.simt_stack.top()
                if top:
                    warp.simt_stack.entries[-1] = SIMTStackEntry(
                        reconv_pc=target_pc,
                        orig_mask=top.orig_mask,
                        taken_mask=top.taken_mask,
                        fallthrough_pc=top.fallthrough_pc
                    )
            warp.pc = target_pc
            warp.ibuffer.flush()  # 跳转后清空 I-Buffer
        else:
            pass

    def _get_latency(self, opcode: int) -> int:
        """获取指令的流水线延迟"""
        if opcode in (OP_LD, OP_ST):
            return PIPELINE_LATENCY['mem']
        if opcode in (OP_HALT, OP_BAR, OP_JMP, OP_BEQ, OP_BNE):
            return 0  # 控制指令不写寄存器
        return PIPELINE_LATENCY.get('default', 1)

    def run(self, trace: bool = False):
        while self.step():
            pass

    def dump_state(self) -> str:
        lines = ["=" * 60, "SIMT Core State (Phase 3: SIMT Stack)",
                 "=" * 60,
                 f"Warp Size: {self.warp_size} | "
                 f"Num Warps: {len(self.warps)} | "
                 f"Instructions: {self.instr_count}", ""]
        for w in self.warps:
            lines.append(w.dump())
            if not w.simt_stack.empty:
                lines.append(f"  SIMT Stack depth: {len(w.simt_stack)}")
            lines.append("")
        lines.append("Memory (non-zero):")
        lines.append(self.memory.dump())
        lines.append("")
        lines.append(self.l1_cache.stats())
        if self.total_mem_reqs > 0:
            eff = self.coalesce_count / self.total_mem_reqs * 100
            lines.append(f"Coalescing: {self.coalesce_count}/{self.total_mem_reqs} ({eff:.0f}%)")
        return "\n".join(lines)
