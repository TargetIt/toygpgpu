"""
ISA (Instruction Set Architecture) — 指令集定义与译码模块
==========================================================
对标 GPGPU-Sim 中 gpgpu-sim/instructions.cc 的指令定义和 ptx_parser 的译码功能。

每条指令 32-bit 定长编码:
  [31:24] opcode   (8-bit)
  [23:20] rd       (4-bit, 目的寄存器)
  [19:16] rs1      (4-bit, 源寄存器1)
  [15:12] rs2      (4-bit, 源寄存器2)
  [11:0]  imm      (12-bit, 立即数或地址)

参考: RISC-V RV32I 指令格式 (riscv.org)
      TinyGPU 的指令集定义 (github.com/deaneeth/tinygpu)
"""

from dataclasses import dataclass

# ============================================================
# 操作码定义
# ============================================================
OP_HALT = 0x00  # 停止执行
OP_ADD  = 0x01  # rd = rs1 + rs2
OP_SUB  = 0x02  # rd = rs1 - rs2
OP_MUL  = 0x03  # rd = rs1 * rs2 (低32位)
OP_DIV  = 0x04  # rd = rs1 / rs2 (除零得0)
OP_LD   = 0x05  # rd = mem[imm]
OP_ST   = 0x06  # mem[imm] = rs1
OP_MOV  = 0x07  # rd = sign_ext(imm)

# 操作码名称映射
OPCODE_NAMES = {
    OP_HALT: "HALT",
    OP_ADD:  "ADD",
    OP_SUB:  "SUB",
    OP_MUL:  "MUL",
    OP_DIV:  "DIV",
    OP_LD:   "LD",
    OP_ST:   "ST",
    OP_MOV:  "MOV",
}


@dataclass
class Instruction:
    """译码后的指令

    字段与 GPGPU-Sim 中 warp_inst_t 的简化版本对应。
    """
    opcode: int
    rd: int      # 0-15
    rs1: int     # 0-15
    rs2: int     # 0-15
    imm: int     # 12-bit, 符号扩展到 32-bit
    raw: int     # 原始 32-bit 编码

    @property
    def name(self) -> str:
        return OPCODE_NAMES.get(self.opcode, f"UNKNOWN({self.opcode:02X})")

    def __repr__(self):
        return (f"Instruction({self.name}, rd=r{self.rd}, "
                f"rs1=r{self.rs1}, rs2=r{self.rs2}, imm={self.imm})")


def decode(word: int) -> Instruction:
    """将 32-bit 指令字译码为 Instruction 对象

    对应 GPGPU-Sim 中 ptx_parser 将 PTX 指令翻译为内部表示的过程。
    """
    opcode = (word >> 24) & 0xFF
    rd     = (word >> 20) & 0xF
    rs1    = (word >> 16) & 0xF
    rs2    = (word >> 12) & 0xF
    imm12  = word & 0xFFF

    # 符号扩展 12-bit 立即数到 32-bit
    if imm12 & 0x800:
        imm = imm12 | 0xFFFFF000
    else:
        imm = imm12

    return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2, imm=imm, raw=word)


def encode_rtype(opcode: int, rd: int, rs1: int, rs2: int, imm: int = 0) -> int:
    """编码 R-type 指令 (ADD, SUB, MUL, DIV)"""
    w = 0
    w |= (opcode & 0xFF) << 24
    w |= (rd & 0xF) << 20
    w |= (rs1 & 0xF) << 16
    w |= (rs2 & 0xF) << 12
    w |= imm & 0xFFF
    return w


def encode_itype(opcode: int, rd: int, imm: int) -> int:
    """编码 I-type 指令 (MOV, LD)"""
    w = 0
    w |= (opcode & 0xFF) << 24
    w |= (rd & 0xF) << 20
    w |= imm & 0xFFF
    return w


def encode_stype(opcode: int, rs1: int, imm: int) -> int:
    """编码 S-type 指令 (ST)"""
    w = 0
    w |= (opcode & 0xFF) << 24
    w |= (rs1 & 0xF) << 16
    w |= imm & 0xFFF
    return w
