"""
CPU — SIMD 向量处理器顶层模块
==============================
对标 GPGPU-Sim 中 gpgpu_sim + shader_core_ctx 的 SIMD 执行模型。

Phase 1: 在 Phase 0 标量处理器基础上增加 VLEN 路 SIMD 向量执行。
标量指令 (ADD/SUB/...) 与向量指令 (VADD/VSUB/...) 共存。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 shader_core_ctx::cycle()
      GPGPU-Sim 的 simd_function_unit 多 lane 并行执行
"""

try:
    from .isa import (Instruction, decode, OP_HALT,
                      OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                      OP_LD, OP_ST, OP_MOV,
                      OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
                      OP_VLD, OP_VST, OP_VMOV,
                      OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
                      OPCODE_NAMES, VECTOR_OPS)
    from .register_file import RegisterFile
    from .vector_register_file import VectorRegisterFile
    from .alu import ALU
    from .vector_alu import VectorALU
    from .vec4_alu import Vec4ALU
    from .memory import Memory
except ImportError:
    from isa import (Instruction, decode, OP_HALT,
                     OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                     OP_LD, OP_ST, OP_MOV,
                     OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
                     OP_VLD, OP_VST, OP_VMOV,
                     OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
                     OPCODE_NAMES, VECTOR_OPS)
    from register_file import RegisterFile
    from vector_register_file import VectorRegisterFile
    from alu import ALU
    from vector_alu import VectorALU
    from vec4_alu import Vec4ALU
    from memory import Memory


# 默认配置
DEFAULT_VLEN = 8
DEFAULT_MEM_SIZE = 1024


class CPU:
    """SIMD 向量处理器

    Phase 0 (标量) → Phase 1 (SIMD 向量) 的扩展。

    对标 GPGPU-Sim:
      - RegisterFile + ALU → warp 的标量操作（地址计算）
      - VectorRegisterFile + VectorALU → warp 的 SIMD 执行（数据并行）
      - VLEN = GPGPU-Sim 的 warp size (32 threads/warp)

    Attributes:
        pc: 程序计数器
        reg_file: 标量寄存器堆 (16 × 32-bit)
        vec_reg_file: 向量寄存器堆 (8 × VLEN × 32-bit)
        alu: 标量 ALU
        vec_alu: VLEN 路向量 ALU
        memory: 数据内存
        vlen: 向量长度 (lane 数)
        program: 指令内存
        halted: 是否已停止
        instr_count: 已执行指令计数
    """

    def __init__(self, vlen: int = DEFAULT_VLEN, memory_size: int = DEFAULT_MEM_SIZE):
        self.pc = 0
        self.vlen = vlen
        self.reg_file = RegisterFile(16)
        self.vec_reg_file = VectorRegisterFile(vlen, 8)
        self.alu = ALU()
        self.vec_alu = VectorALU(vlen)
        self.vec4_alu = Vec4ALU()
        self.memory = Memory(memory_size)
        self.program: list[int] = []
        self.halted = False
        self.instr_count = 0

    def load_program(self, program: list[int]):
        """加载程序到指令内存"""
        self.program = list(program)
        self.pc = 0
        self.halted = False
        self.instr_count = 0

    def step(self) -> bool:
        """执行一条指令"""
        if self.halted:
            return False
        if self.pc < 0 or self.pc >= len(self.program):
            self.halted = True
            return False

        raw_word = self.program[self.pc]
        self.pc += 1
        instr = decode(raw_word)
        self.instr_count += 1

        self._execute(instr)
        return not self.halted

    def _execute(self, instr: Instruction):
        """执行指令"""
        op = instr.opcode

        # ---- 标量指令 (Phase 0 兼容) ----
        if op == OP_HALT:
            self.halted = True

        elif op == OP_ADD:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.alu.add(a, b))

        elif op == OP_SUB:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.alu.sub(a, b))

        elif op == OP_MUL:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.alu.mul(a, b))

        elif op == OP_DIV:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.alu.div(a, b))

        elif op == OP_LD:
            addr = instr.imm & 0x3FF  # 10-bit 地址 (0-1023)
            self.reg_file.write(instr.rd, self.memory.read_word(addr))

        elif op == OP_ST:
            addr = instr.imm & 0x3FF
            self.memory.write_word(addr, self.reg_file.read(instr.rs1))

        elif op == OP_MOV:
            self.reg_file.write(instr.rd, instr.imm)

        # ---- 向量指令 (Phase 1 新增) ----
        elif op == OP_VADD:
            a = self.vec_reg_file.read(instr.rs1)
            b = self.vec_reg_file.read(instr.rs2)
            self.vec_reg_file.write(instr.rd, self.vec_alu.vadd(a, b))

        elif op == OP_VSUB:
            a = self.vec_reg_file.read(instr.rs1)
            b = self.vec_reg_file.read(instr.rs2)
            self.vec_reg_file.write(instr.rd, self.vec_alu.vsub(a, b))

        elif op == OP_VMUL:
            a = self.vec_reg_file.read(instr.rs1)
            b = self.vec_reg_file.read(instr.rs2)
            self.vec_reg_file.write(instr.rd, self.vec_alu.vmul(a, b))

        elif op == OP_VDIV:
            a = self.vec_reg_file.read(instr.rs1)
            b = self.vec_reg_file.read(instr.rs2)
            self.vec_reg_file.write(instr.rd, self.vec_alu.vdiv(a, b))

        elif op == OP_VLD:
            base_addr = instr.imm & 0x3FF
            values = [self.memory.read_word(base_addr + i)
                      for i in range(self.vlen)]
            self.vec_reg_file.write(instr.rd, values)

        elif op == OP_VST:
            base_addr = instr.imm & 0x3FF
            values = self.vec_reg_file.read(instr.rs1)
            for i in range(self.vlen):
                self.memory.write_word(base_addr + i, values[i])

        elif op == OP_VMOV:
            self.vec_reg_file.broadcast(instr.rd, instr.imm)

        # ---- Vec4 指令 (Phase 1+ 扩展) ----
        elif op == OP_V4PACK:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.vec4_alu.pack(a, b))

        elif op == OP_V4ADD:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.vec4_alu.add(a, b))

        elif op == OP_V4MUL:
            a = self.reg_file.read(instr.rs1)
            b = self.reg_file.read(instr.rs2)
            self.reg_file.write(instr.rd, self.vec4_alu.mul(a, b))

        elif op == OP_V4UNPACK:
            a = self.reg_file.read(instr.rs1)
            lane = instr.rs2 & 3
            self.reg_file.write(instr.rd, self.vec4_alu.unpack(a, lane))

        else:
            raise ValueError(f"Unknown opcode: 0x{op:02X} at pc={self.pc - 1}")

    def run(self, trace: bool = False):
        """Run until HALT with optional trace."""
        cycle = 0
        if trace:
            self._t_regs = {i: self.reg_file.read(i) for i in range(16)}
            self._t_vregs = {i: [self.vreg_file.read(i, lane) for lane in range(self.vlen)] for i in range(8)}
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

    def _snapshot_vregs(self) -> dict:
        return {i: [self.vreg_file.read(i, lane) for lane in range(self.vlen)] for i in range(8)}

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
        is_vector = instr.opcode in VECTOR_OPS
        curr_regs = self._snapshot_regs()
        reg_parts = []
        for i in range(16):
            ov = self._t_regs.get(i, 0)
            nv = curr_regs.get(i, 0)
            if ov != nv:
                reg_parts.append(f"r{i}:{ov}->{nv}")
        reg_str = ' '.join(reg_parts) if reg_parts else "none"
        vreg_parts = []
        if is_vector:
            curr_vregs = self._snapshot_vregs()
            for vi in range(8):
                ov_list = self._t_vregs.get(vi, [0]*self.vlen)
                nv_list = curr_vregs.get(vi, [0]*self.vlen)
                if ov_list != nv_list:
                    vreg_parts.append(f"v{vi}:{ov_list}->{nv_list}")
        curr_mem = self._snapshot_mem()
        mem_parts = []
        for addr, nv in curr_mem.items():
            ov = self._t_mem.get(addr, 0)
            if ov != nv:
                mem_parts.append(f"mem[{addr}]:{ov}->{nv}")
        mem_str = (' | mem: ' + ', '.join(mem_parts[:4])) if mem_parts else ''
        vreg_str = (' | vreg: ' + ', '.join(vreg_parts[:4])) if vreg_parts else ''
        print(f"[Cycle {cycle}] PC={exec_pc}: {opname} rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm} | reg: {reg_str}{vreg_str}{mem_str}")
        self._t_regs = curr_regs
        if is_vector:
            self._t_vregs = curr_vregs
        self._t_mem = curr_mem

    def _update_trace_state(self):
        self._t_regs = {i: self.reg_file.read(i) for i in range(16)}
        self._t_vregs = {i: [self.vreg_file.read(i, lane) for lane in range(self.vlen)] for i in range(8)}
        self._t_mem = {}
        for i in range(self.memory.size_words):
            v = self.memory.read_word(i)
            if v != 0:
                self._t_mem[i] = v

    def dump_state(self) -> str:
        """打印 CPU 完整状态"""
        lines = [
            "=" * 60,
            "CPU State (Phase 1: SIMD Vector)",
            "=" * 60,
            f"PC: {self.pc} | Halted: {self.halted} | "
            f"Instructions: {self.instr_count} | VLEN: {self.vlen}",
            "",
            "Scalar Registers:",
            self.reg_file.dump(),
            "",
            "Vector Registers:",
            self.vec_reg_file.dump(),
            "",
            "Memory (non-zero):",
            self.memory.dump(),
            "=" * 60,
        ]
        return "\n".join(lines)


def instr_name(word: int) -> str:
    """返回指令字的人类可读名称（调试用）"""
    from isa import OPCODE_NAMES
    op = (word >> 24) & 0xFF
    return OPCODE_NAMES.get(op, f"UNK({op:02X})")
