"""
SIMTCore — SIMT 核心顶层模块
=============================
对标 GPGPU-Sim 中 shader_core_ctx + gpgpu_sim 的 SIMT 执行模型。

Phase 2: 支持多 warp、多线程并行执行。
每个线程有独立的标量寄存器堆，同一 warp 内线程共享 PC。

Warp 内线程访存时使用 base_addr + thread_id 作为实际地址，
模仿 GPU 编程中每个线程访问全局内存不同位置的模式。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 shader_core_ctx::cycle()
      的 6 级流水线及 warp 管理
"""

try:
    from .isa import (Instruction, decode,
                      OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                      OP_LD, OP_ST, OP_MOV,
                      OP_TID, OP_WID, OP_BAR)
    from .alu import ALU
    from .memory import Memory
    from .warp import Warp
    from .scheduler import WarpScheduler
except ImportError:
    from isa import (Instruction, decode,
                     OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                     OP_LD, OP_ST, OP_MOV,
                     OP_TID, OP_WID, OP_BAR)
    from alu import ALU
    from memory import Memory
    from warp import Warp
    from scheduler import WarpScheduler


# 默认配置
DEFAULT_WARP_SIZE = 8
DEFAULT_NUM_WARPS = 2
DEFAULT_MEM_SIZE = 1024


class SIMTCore:
    """SIMT 核心

    对标 GPGPU-Sim 中一个 shader_core_ctx（SM/Streaming Multiprocessor）。

    GPGPU-Sim 的一个 SM 包含多个 warp，通过 scheduler_unit
    在每个周期选择一个 warp 发射指令。

    Attributes:
        scheduler: Warp 调度器
        warps: 所有 warp
        warp_size: 每个 warp 的线程数
        memory: 全局共享内存
        alu: 标量 ALU（所有线程共享执行单元）
        program: 指令内存
        instr_count: 总指令计数
    """

    def __init__(self, warp_size: int = DEFAULT_WARP_SIZE,
                 num_warps: int = DEFAULT_NUM_WARPS,
                 memory_size: int = DEFAULT_MEM_SIZE):
        self.warp_size = warp_size
        self.warps = [Warp(wid, warp_size) for wid in range(num_warps)]
        self.scheduler = WarpScheduler(self.warps)
        self.memory = Memory(memory_size)
        self.alu = ALU()
        self.program: list[int] = []
        self.instr_count = 0

    def load_program(self, program: list[int]):
        """加载程序到所有 warp

        对应 GPGPU-Sim 中 kernel launch: 所有 warp 从同一条指令开始。
        """
        self.program = list(program)
        for w in self.warps:
            w.pc = 0
            w.done = False
            w.reset_barrier()
            w.active_mask = (1 << self.warp_size) - 1
        self.instr_count = 0

    def step(self) -> bool:
        """执行一个 warp 的一条指令

        对应 GPGPU-Sim 中一个时钟周期内的 warp issue + execute。

        Returns:
            True: 继续执行
            False: 所有 warp 完成
        """
        warp = self.scheduler.select_warp()
        if warp is None:
            return False

        if warp.pc < 0 or warp.pc >= len(self.program):
            warp.done = True
            return self.scheduler.has_active_warps()

        # Fetch & Decode（共享 PC，同一条指令宽执行）
        raw_word = self.program[warp.pc]
        instr = decode(raw_word)

        # Execute: 对每个 active 线程
        self._execute_warp(warp, instr)

        warp.pc += 1
        self.instr_count += 1

        if warp.done:
            return self.scheduler.has_active_warps()
        return True

    def _execute_warp(self, warp: Warp, instr: Instruction):
        """对 warp 内所有 active 线程执行指令

        对标 GPGPU-Sim 的 SIMD 执行：一条指令广播到所有活跃线程。
        每个线程使用自己的寄存器堆。
        """
        op = instr.opcode

        # ---- HALT: warp 完成 ----
        if op == OP_HALT:
            warp.done = True
            return

        # ---- BAR (barrier sync) ----
        # SIMT: 同一 warp 内所有 active 线程共享 PC，同时到达 BAR。
        # 因此 BAR 对于单 warp 等同于 nop（Phase 3 SIMT Stack 会改变这一点）。
        # 仅需重置 at_barrier 状态（从上一轮 barrier wait 中恢复）。
        if op == OP_BAR:
            if warp.at_barrier:
                warp.reset_barrier()
            return

        # ---- TID / WID: 每线程不同值 ----
        if op == OP_TID:
            for t in warp.active_threads():
                t.write_reg(instr.rd, t.thread_id)
            return

        if op == OP_WID:
            for t in warp.active_threads():
                t.write_reg(instr.rd, warp.warp_id)
            return

        # ---- 标量 ALU 操作（每线程独立） ----
        if op == OP_ADD:
            for t in warp.active_threads():
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.add(a, b))
            return

        if op == OP_SUB:
            for t in warp.active_threads():
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.sub(a, b))
            return

        if op == OP_MUL:
            for t in warp.active_threads():
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.mul(a, b))
            return

        if op == OP_DIV:
            for t in warp.active_threads():
                a = t.read_reg(instr.rs1)
                b = t.read_reg(instr.rs2)
                t.write_reg(instr.rd, self.alu.div(a, b))
            return

        if op == OP_MOV:
            for t in warp.active_threads():
                t.write_reg(instr.rd, instr.imm)
            return

        # ---- LD/ST: base_addr + thread_id 寻址 ----
        if op == OP_LD:
            base_addr = instr.imm & 0x3FF
            for t in warp.active_threads():
                addr = (base_addr + t.thread_id) & 0x3FF
                val = self.memory.read_word(addr)
                t.write_reg(instr.rd, val)
            return

        if op == OP_ST:
            base_addr = instr.imm & 0x3FF
            for t in warp.active_threads():
                addr = (base_addr + t.thread_id) & 0x3FF
                val = t.read_reg(instr.rs1)
                self.memory.write_word(addr, val)
            return

        raise ValueError(f"Unknown opcode: 0x{op:02X}")

    def run(self, trace: bool = False):
        """运行所有 warp 直到完成"""
        while self.step():
            if trace:
                pass  # 可加调试输出
        if trace:
            print(f"\n[Complete] Total instructions: {self.instr_count}")

    def dump_state(self) -> str:
        lines = ["=" * 60,
                 "SIMT Core State",
                 "=" * 60,
                 f"Warp Size: {self.warp_size} | "
                 f"Num Warps: {len(self.warps)} | "
                 f"Instructions: {self.instr_count}", ""]
        for w in self.warps:
            lines.append(w.dump())
            lines.append("")
        lines.append("Memory (non-zero):")
        lines.append(self.memory.dump())
        return "\n".join(lines)
