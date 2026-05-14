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
                      OP_JMP, OP_BEQ, OP_BNE, OP_SETP,
                      OP_WREAD, OP_WWRITE,
                      OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
                      PRED_FLAG, OPCODE_NAMES)
    from .alu import ALU
    from .vec4_alu import Vec4ALU
    from .memory import Memory
    from .warp import Warp
    from .scheduler import WarpScheduler
    from .simt_stack import SIMTStackEntry
except ImportError:
    from isa import (Instruction, decode,
                     OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                     OP_LD, OP_ST, OP_MOV,
                     OP_TID, OP_WID, OP_BAR,
                     OP_JMP, OP_BEQ, OP_BNE, OP_SETP,
                     OP_WREAD, OP_WWRITE,
                     OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
                     PRED_FLAG, OPCODE_NAMES)
    from alu import ALU
    from vec4_alu import Vec4ALU
    from memory import Memory
    from warp import Warp
    from scheduler import WarpScheduler
    from simt_stack import SIMTStackEntry

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
        self.memory = Memory(memory_size)
        self.alu = ALU()
        self.vec4_alu = Vec4ALU()
        self.program: list[int] = []
        self.instr_count = 0

    def load_program(self, program: list[int]):
        self.program = list(program)
        for w in self.warps:
            w.pc = 0
            w.done = False
            w.reset_barrier()
            w.active_mask = (1 << self.warp_size) - 1
            w.simt_stack.entries.clear()
        self.instr_count = 0

    def step(self) -> bool:
        """执行一个周期"""
        warp = self.scheduler.select_warp()
        if warp is None:
            self._last_warp_id = -1
            return False
        self._last_warp_id = warp.warp_id
        self._last_reconv_happened = False

        # ---- 重汇聚检测 (GPGPU-Sim: simt_stack::at_reconvergence) ----
        if warp.simt_stack.at_reconvergence(warp.pc):
            self._last_reconv_happened = True
            self._handle_reconvergence(warp)
            # reconvergence 可能更新 PC，本周期不执行指令
            if warp.done:
                return self.scheduler.has_active_warps()
            # 如果重汇聚后 active_mask 为空，跳过本次
            if popcount(warp.active_mask) == 0:
                return self.scheduler.has_active_warps()
            # 可能直接切换到另一路径，重新 fetch
            pass

        if warp.pc < 0 or warp.pc >= len(self.program):
            warp.done = True
            return self.scheduler.has_active_warps()

        raw_word = self.program[warp.pc]
        instr = decode(raw_word)
        # 执行前先 PC+1（分支指令可能覆盖 PC）
        old_pc = warp.pc
        warp.pc += 1
        self._execute_warp(warp, instr, old_pc)
        self.instr_count += 1

        return self.scheduler.has_active_warps()

    def _handle_reconvergence(self, warp: Warp):
        """处理 SIMT Stack 重汇聚

        对应 GPGPU-Sim 中 simt_stack::pop + 路径切换逻辑。
        """
        entry = warp.simt_stack.pop()
        if entry is None:
            return

        remaining_mask = entry.orig_mask & ~entry.taken_mask
        if remaining_mask:
            # 切换到未执行路径
            # taken_mask 累积：已执行路径 + 即将执行路径
            warp.simt_stack.push(SIMTStackEntry(
                reconv_pc=entry.reconv_pc,
                orig_mask=entry.orig_mask,
                taken_mask=entry.taken_mask | remaining_mask,  # 累积
                fallthrough_pc=entry.fallthrough_pc
            ))
            warp.active_mask = remaining_mask
            warp.pc = entry.fallthrough_pc
        else:
            warp.active_mask = entry.orig_mask
            warp.pc = entry.reconv_pc

    def _exec_threads(self, warp, instr):
        """返回应执行当前指令的线程列表 (考虑 active_mask + 谓词)"""
        active = warp.active_threads()
        # 检查 @p0 谓词标志 (bit 11)
        if instr.raw & PRED_FLAG:
            return [t for t in active if t.pred]
        return active

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
            for t in self._exec_threads(warp, instr):
                t.write_reg(instr.rd, t.thread_id)
            return

        if op == OP_WID:
            for t in self._exec_threads(warp, instr):
                t.write_reg(instr.rd, warp.warp_id)
            return

        # ---- Warp-level register read/write ----
        if op == OP_WREAD:
            wreg_idx = instr.imm & 0xF
            value = warp.read_warp_reg(wreg_idx)
            for t in self._exec_threads(warp, instr):
                t.write_reg(instr.rd, value)
            return
        if op == OP_WWRITE:
            wreg_idx = instr.imm & 0xF
            tlist = self._exec_threads(warp, instr)
            if tlist:
                value = tlist[0].read_reg(instr.rs1)
                warp.write_warp_reg(wreg_idx, value)
            return

        # ---- 谓词指令 (Phase 3: Predication) ----
        if op == OP_SETP:
            is_eq = (instr.imm & 1) == 0  # imm[0]=0 → EQ, 1 → NE
            for t in warp.active_threads():
                if is_eq:
                    t.pred = (t.read_reg(instr.rs1) == t.read_reg(instr.rs2))
                else:
                    t.pred = (t.read_reg(instr.rs1) != t.read_reg(instr.rs2))
            return

        # ---- 分支指令 (Phase 3) ----
        if op in (OP_JMP, OP_BEQ, OP_BNE):
            self._execute_branch(warp, instr, old_pc)
            return

        tlist = self._exec_threads(warp, instr)

        # ---- ALU ----
        if op == OP_ADD:
            for t in tlist:
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.add(a, b))
            return
        if op == OP_SUB:
            for t in tlist:
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.sub(a, b))
            return
        if op == OP_MUL:
            for t in tlist:
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.mul(a, b))
            return
        if op == OP_DIV:
            for t in tlist:
                a, b = t.read_reg(instr.rs1), t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.div(a, b))
            return
        if op == OP_MOV:
            for t in tlist:
                t.write_reg(instr.rd, instr.imm)
            return

        # ---- LD/ST ----
        if op == OP_LD:
            base = instr.imm & 0x3FF
            for t in tlist:
                addr = (base + t.thread_id) & 0x3FF
                t.write_reg(instr.rd, self.memory.read_word(addr))
            return
        if op == OP_ST:
            base = instr.imm & 0x3FF
            for t in tlist:
                addr = (base + t.thread_id) & 0x3FF
                self.memory.write_word(addr, t.read_reg(instr.rs1))
            return

        # ---- Vec4 指令 (Phase 1+ 扩展) ----
        if op == OP_V4PACK:
            for t in tlist:
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.vec4_alu.pack(a, b))
            return
        if op == OP_V4ADD:
            for t in tlist:
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.vec4_alu.add(a, b))
            return
        if op == OP_V4MUL:
            for t in tlist:
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.vec4_alu.mul(a, b))
            return
        if op == OP_V4UNPACK:
            for t in tlist:
                a = t.read_reg(instr.rs1)
                lane = instr.rs2 & 3
                t.write_reg(instr.rd, self.vec4_alu.unpack(a, lane))
            return

        raise ValueError(f"Unknown opcode: 0x{op:02X}")

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
        elif taken_mask:
            if is_merge_jmp:
                # JMP within divergent path: update reconv_pc to JMP target
                top = warp.simt_stack.top()
                if top:
                    warp.simt_stack.entries[-1] = SIMTStackEntry(
                        reconv_pc=target_pc,  # 新重汇聚点
                        orig_mask=top.orig_mask,
                        taken_mask=top.taken_mask,
                        fallthrough_pc=top.fallthrough_pc
                    )
            warp.pc = target_pc
        else:
            pass

    def run(self, trace: bool = False):
        cycle = 0
        if trace:
            self._t_regs = self._snapshot_regs()
            self._t_mem = self._snapshot_mem()
            self._t_pcs = {w.warp_id: w.pc for w in self.warps}
            self._t_masks = {w.warp_id: w.active_mask for w in self.warps}
            self._t_stack_len = {w.warp_id: len(w.simt_stack.entries) for w in self.warps}
        while self.step():
            if trace:
                self._trace_step(cycle)
            cycle += 1
        if trace:
            print(f"[Summary] {cycle} cycles, {self.instr_count} instructions")

    def _snapshot_regs(self) -> dict:
        """Snapshot all non-zero registers for trace comparison."""
        snap = {}
        for w in self.warps:
            for t in w.threads:
                for i in range(16):
                    v = t.read_reg(i)
                    if v != 0:
                        snap[(w.warp_id, t.thread_id, i)] = v
        return snap

    def _snapshot_mem(self) -> dict:
        """Snapshot non-zero memory for trace comparison."""
        snap = {}
        for i in range(self.memory.size_words):
            v = self.memory.read_word(i)
            if v != 0:
                snap[i] = v
        return snap

    def _trace_step(self, cycle: int):
        """Print one cycle's trace output."""
        wid = self._last_warp_id
        if wid < 0:
            print(f"[Cycle {cycle}] (no warp active)")
            return

        warp = self.warps[wid]
        curr_pc = warp.pc
        old_pc = self._t_pcs.get(wid, -1)
        new_mask = warp.active_mask
        old_mask = self._t_masks.get(wid, 0)
        mask_str = bin(new_mask)[2:].zfill(self.warp_size)
        old_stack_len = self._t_stack_len.get(wid, 0)
        curr_stack_len = len(warp.simt_stack.entries)
        reconv_happened = getattr(self, '_last_reconv_happened', False)

        # ---- Reconvergence (path switch, then instruction executed at new PC) ----
        if reconv_happened:
            exec_pc = curr_pc - 1
            if 0 <= exec_pc < len(self.program):
                raw_word = self.program[exec_pc]
                instr = decode(raw_word)
                opname = OPCODE_NAMES.get(instr.opcode, '?')
                desc = f"{opname} rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm}"
            else:
                desc = "???"
            print(f"[Cycle {cycle}] W{wid} RECONVERGE: mask={mask_str} path=PC{exec_pc}: {desc}")
            self._update_trace_state()
            return

        # ---- Normal instruction execution ----
        if old_pc < 0 or old_pc >= len(self.program):
            print(f"[Cycle {cycle}] W{wid} PC={old_pc} (out of range)")
            self._update_trace_state()
            return

        raw_word = self.program[old_pc]
        instr = decode(raw_word)
        opname = OPCODE_NAMES.get(instr.opcode, '?')

        # Build instruction description
        if instr.opcode in (OP_JMP, OP_BEQ, OP_BNE):
            target_pc = old_pc + 1 + instr.imm
            if instr.opcode == OP_JMP:
                desc = f"{opname} -> PC{target_pc}"
            else:
                desc = f"{opname} r{instr.rs1},r{instr.rs2} -> PC{target_pc}"
        else:
            desc = f"{opname} rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm}"

        parts = [f"[Cycle {cycle}] W{wid} PC={old_pc}: {desc} | active=0b{mask_str}"]

        # ---- Register changes (only for this warp) ----
        curr_regs = self._snapshot_regs()
        reg_diffs = []
        for (w_, t_, r_), nv in curr_regs.items():
            if w_ == wid:
                ov = self._t_regs.get((w_, t_, r_), 0)
                if ov != nv:
                    reg_diffs.append(f"T{t_}:r{r_}={ov}->{nv}")
        if reg_diffs:
            reg_str = ', '.join(reg_diffs[:8])
            if len(reg_diffs) > 8:
                reg_str += f" (+{len(reg_diffs)-8})"
            parts.append(f"reg: {reg_str}")

        # ---- Memory changes ----
        curr_mem = self._snapshot_mem()
        mem_diffs = []
        for addr, nv in curr_mem.items():
            ov = self._t_mem.get(addr, 0)
            if ov != nv:
                mem_diffs.append(f"mem[{addr}]={nv}")
        if mem_diffs:
            mem_str = ', '.join(mem_diffs[:4])
            if len(mem_diffs) > 4:
                mem_str += f" (+{len(mem_diffs)-4})"
            parts.append(f"mem: {mem_str}")

        # ---- Divergence (stack push from branch) ----
        if curr_stack_len > old_stack_len:
            not_taken = old_mask & ~new_mask
            if not_taken:
                parts.append(f"DIVERGE: taken=0b{new_mask:0{self.warp_size}b} not_taken=0b{not_taken:0{self.warp_size}b}")

        print(' | '.join(parts))

        # Update trace state
        self._t_regs = curr_regs
        self._t_mem = curr_mem
        self._t_pcs[wid] = warp.pc
        self._t_masks[wid] = warp.active_mask
        self._t_stack_len[wid] = curr_stack_len

    def _update_trace_state(self):
        """Update all trace state after a cycle without instruction execution."""
        self._t_regs = self._snapshot_regs()
        self._t_mem = self._snapshot_mem()
        self._t_pcs = {w.warp_id: w.pc for w in self.warps}
        self._t_masks = {w.warp_id: w.active_mask for w in self.warps}
        self._t_stack_len = {w.warp_id: len(w.simt_stack.entries) for w in self.warps}

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
        return "\n".join(lines)
