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
# --- 标量指令 (Phase 0) ---
OP_HALT = 0x00  # 停止执行
OP_ADD  = 0x01  # rd = rs1 + rs2
OP_SUB  = 0x02  # rd = rs1 - rs2
OP_MUL  = 0x03  # rd = rs1 * rs2 (低32位)
OP_DIV  = 0x04  # rd = rs1 / rs2 (除零得0)
OP_LD   = 0x05  # rd = mem[imm]
OP_ST   = 0x06  # mem[imm] = rs1
OP_MOV  = 0x07  # rd = sign_ext(imm)

# --- 向量指令 (Phase 1) ---
OP_VADD = 0x11  # vd[i] = vs1[i] + vs2[i]  (VLEN lanes)
OP_VSUB = 0x12  # vd[i] = vs1[i] - vs2[i]
OP_VMUL = 0x13  # vd[i] = vs1[i] * vs2[i]
OP_VDIV = 0x14  # vd[i] = vs1[i] / vs2[i]
OP_VLD  = 0x15  # vd[i] = mem[addr + i] (连续加载)
OP_VST  = 0x16  # mem[addr + i] = vs[i] (连续存储)
OP_VMOV = 0x17  # vd[i] = sign_ext(imm) (广播)

# --- SIMT 指令 (Phase 2) ---
OP_TID = 0x21  # rd = thread_id
OP_WID = 0x22  # rd = warp_id
OP_BAR = 0x23  # barrier synchronization

# --- 分支指令 (Phase 3) ---
OP_JMP = 0x08  # 无条件跳转: PC = PC + imm
OP_BEQ = 0x09  # rs1 == rs2 时跳转
OP_BNE = 0x0A  # rs1 != rs2 时跳转

# --- 谓词指令 (Phase 3: Predication) ---
OP_SETP = 0x24  # 设置谓词: imm[0]=0 表示 EQ, imm[0]=1 表示 NE

# @p0 谓词标志位 (指令字 bit 31, opcode 字段高位)
# decode() 会将 opcode 与 0x7F 做 & 运算来剥离谓词标志
PRED_FLAG = 0x80000000

# --- Warp-level register instructions ---
OP_WREAD  = 0x2A  # rd = warp_reg[imm] — read warp uniform register
OP_WWRITE = 0x2B  # warp_reg[imm] = rs1 — write warp uniform register

# --- Vec4 复合数据类型指令 (Phase 1+ 扩展) ---
OP_V4PACK   = 0x25  # rd = pack(rs1[7:0], rs1[15:8], rs2[7:0], rs2[15:8])
OP_V4ADD    = 0x26  # rd[i] = rs1[i] + rs2[i] for i in 0..3 (4×8-bit SIMD add)
OP_V4MUL    = 0x27  # rd[i] = rs1[i] * rs2[i] for i in 0..3 (4×8-bit SIMD mul)
OP_V4UNPACK = 0x28  # extract byte lane: rd = (rs1 >> (lane*8)) & 0xFF, lane in rs2

# Warp register name → index mapping (stored in imm field)
WREG_NAMES = {
    'wid': 0,    # warp_id
    'ntid': 1,   # num_threads per warp
}

# 操作码名称映射
OPCODE_NAMES = {
    OP_HALT: "HALT", OP_ADD: "ADD", OP_SUB: "SUB",
    OP_MUL: "MUL", OP_DIV: "DIV", OP_LD: "LD",
    OP_ST: "ST", OP_MOV: "MOV",
    OP_VADD: "VADD", OP_VSUB: "VSUB", OP_VMUL: "VMUL",
    OP_VDIV: "VDIV", OP_VLD: "VLD", OP_VST: "VST",
    OP_VMOV: "VMOV",
    OP_TID: "TID", OP_WID: "WID", OP_BAR: "BAR",
    OP_JMP: "JMP", OP_BEQ: "BEQ", OP_BNE: "BNE",
    OP_SETP: "SETP",
    OP_WREAD: "WREAD", OP_WWRITE: "WWRITE",
    OP_V4PACK: "V4PACK", OP_V4ADD: "V4ADD", OP_V4MUL: "V4MUL",
    OP_V4UNPACK: "V4UNPACK",
}

# 向量操作码集合（用于快速判断）
VECTOR_OPS = {OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV, OP_VLD, OP_VST, OP_VMOV}

# Vec4 操作码集合（用于快速判断）
VEC4_OPS = {OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK}

# 是否是向量指令
def is_vector(opcode: int) -> bool:
    return opcode in VECTOR_OPS


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
    bit 31 作为 @p0 谓词标志位, 译码时剥离。
    """
    opcode = (word >> 24) & 0x7F  # 7-bit opcode, bit 31 为谓词标志
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
