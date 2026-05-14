"""
CPU — 标量处理器顶层模块
==========================
对标 GPGPU-Sim 中 gpgpu_sim 顶层类 + shader_core_ctx 的流水线。

Phase 0 实现最简执行模型: Fetch → Decode → Execute → Writeback
无流水线重叠、无分支预测、顺序执行。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 shader_core_ctx::cycle()
      的流水线实现 (fetch/decode/issue/read_operand/execute/writeback)
      TinyGPU 的 GPU 类 (github.com/deaneeth/tinygpu)
"""

try:
    from .isa import Instruction, decode, OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LD, OP_ST, OP_MOV, OPCODE_NAMES
    from .register_file import RegisterFile
    from .alu import ALU
    from .memory import Memory
except ImportError:
    from isa import Instruction, decode, OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LD, OP_ST, OP_MOV, OPCODE_NAMES
    from register_file import RegisterFile
    from alu import ALU
    from memory import Memory


class CPU:
    """标量处理器

    对应 GPGPU-Sim 中一个 SP (Streaming Processor) + 流水线的简化版。

    Attributes:
        pc: 程序计数器 (Program Counter)
        reg_file: 寄存器堆
        alu: 算术逻辑单元
        memory: 数据内存
        program: 指令内存 (机器码列表)
        halted: 是否已停止
        instr_count: 已执行指令计数
    """

    def __init__(self, memory_size: int = 256):
        self.pc = 0
        self.reg_file = RegisterFile(16)
        self.alu = ALU()
        self.memory = Memory(memory_size)
        self.program: list[int] = []
        self.halted = False
        self.instr_count = 0

    def load_program(self, program: list[int]):
        """加载程序到指令内存

        对应 GPGPU-Sim 中 kernel 加载过程。
        """
        self.program = list(program)
        self.pc = 0
        self.halted = False
        self.instr_count = 0

    def step(self) -> bool:
        """执行一条指令 (单步)

        对应 GPGPU-Sim 中 shader_core_ctx::cycle() 一个时钟周期。

        Returns:
            True: 继续执行
            False: HALT (停止)
        """
        if self.halted:
            return False

        if self.pc < 0 or self.pc >= len(self.program):
            self.halted = True
            return False

        # --- Fetch & Decode ---
        # 对应 GPGPU-Sim 流水线: FETCH → DECODE
        raw_word = self.program[self.pc]
        self.pc += 1
        instr = decode(raw_word)
        self.instr_count += 1

        # --- Execute & Writeback ---
        # 对应 GPGPU-Sim 流水线: ISSUE → READ_OPERAND → EXECUTE → WRITEBACK
        self._execute(instr)

        return not self.halted

    def _execute(self, instr: Instruction):
        """执行指令（操作数读取 + 运算 + 写回 合为一步）

        对应 GPGPU-Sim 的 read_operand → execute → writeback 三阶段。
        Phase 0 简化: 组合逻辑一步完成。
        """
        op = instr.opcode

        if op == OP_HALT:
            self.halted = True

        elif op == OP_ADD:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            result = self.alu.add(a, b)
            self.reg_file.write(instr.rd, result)

        elif op == OP_SUB:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            result = self.alu.sub(a, b)
            self.reg_file.write(instr.rd, result)

        elif op == OP_MUL:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            result = self.alu.mul(a, b)
            self.reg_file.write(instr.rd, result)

        elif op == OP_DIV:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            result = self.alu.div(a, b)
            self.reg_file.write(instr.rd, result)

        elif op == OP_LD:
            addr = instr.imm & 0xFF  # 12-bit → 8-bit 地址 (0-255)
            value = self.memory.read_word(addr)
            self.reg_file.write(instr.rd, value)

        elif op == OP_ST:
            value = self.reg_file.read(instr.rs1)
            addr = instr.imm & 0xFF
            self.memory.write_word(addr, value)

        elif op == OP_MOV:
            self.reg_file.write(instr.rd, instr.imm)

        else:
            raise ValueError(f"Unknown opcode: 0x{op:02X} at pc={self.pc - 1}")

    def run(self, trace: bool = False):
        """Run program until HALT with optional trace."""
        cycle = 0
        if trace:
            self._t_regs = {i: self.reg_file.read(i) for i in range(16)}
            self._t_mem = {}
            for i in range(self.memory.size_words):
                v = self.memory.read_word(i)
                if v != 0:
                    self._t_mem[i] = v
        while self.step():
            if trace:
                self._trace_step(cycle)
            cycle += 1
        if trace:
            print(f"[Summary] {cycle} cycles, {self.instr_count} instructions")

    def _snapshot_regs(self) -> dict:
        return {i: self.reg_file.read(i) for i in range(16)}

    def _snapshot_mem(self) -> dict:
        snap = {}
        for i in range(self.memory.size_words):
            v = self.memory.read_word(i)
            if v != 0:
                snap[i] = v
        return snap

    def _trace_step(self, cycle: int):
        exec_pc = self.pc - 1 if not self.halted and self.pc > 0 else self.pc
        if exec_pc < 0 or exec_pc >= len(self.program):
            self._update_trace_state()
            return
        raw_word = self.program[exec_pc]
        instr = decode(raw_word)
        opname = OPCODE_NAMES.get(instr.opcode, '?')
        curr_regs = self._snapshot_regs()
        reg_parts = []
        for i in range(16):
            ov = self._t_regs.get(i, 0)
            nv = curr_regs.get(i, 0)
            if ov != nv:
                reg_parts.append(f"r{i}:{ov}->{nv}")
        reg_str = ' '.join(reg_parts) if reg_parts else "none"
        curr_mem = self._snapshot_mem()
        mem_parts = []
        for addr, nv in curr_mem.items():
            ov = self._t_mem.get(addr, 0)
            if ov != nv:
                mem_parts.append(f"mem[{addr}]:{ov}->{nv}")
        mem_str = (' | ' + ', '.join(mem_parts[:4])) if mem_parts else ''
        print(f"[Cycle {cycle}] PC={exec_pc}: {opname} rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm} | reg: {reg_str}{mem_str}")
        self._t_regs = curr_regs
        self._t_mem = curr_mem

    def _update_trace_state(self):
        self._t_regs = {i: self.reg_file.read(i) for i in range(16)}
        self._t_mem = {}
        for i in range(self.memory.size_words):
            v = self.memory.read_word(i)
            if v != 0:
                self._t_mem[i] = v

    def dump_state(self) -> str:
        """打印 CPU 完整状态（调试用）"""
        lines = [
            "=" * 50,
            "CPU State",
            "=" * 50,
            f"PC: {self.pc} | Halted: {self.halted} | "
            f"Instructions: {self.instr_count}",
            "",
            "Registers:",
            self.reg_file.dump(),
            "",
            "Memory (non-zero):",
            self.memory.dump(),
            "=" * 50,
        ]
        return "\n".join(lines)
