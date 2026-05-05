"""
Assembler — 汇编器模块
=======================
对标 GPGPU-Sim 中 ptx_parser (ptx.l/ptx.y) 的 PTX 汇编解析功能。

将文本汇编代码转换为机器码（32-bit 指令字列表）。

支持的语法:
    ADD r1, r2, r3        ; R-type: rd, rs1, rs2
    MOV r1, 42            ; I-type: rd, imm
    LD  r1, [addr]        ; I-type: rd, addr
    ST  r1, [addr]        ; S-type: rs1, addr
    HALT                  ; 伪指令

参考: GPGPU-Sim cuda-sim/ptx_parser.cc 的 PTX 语法解析
      RISC-V 汇编器设计 (asm-parser pattern)
"""

try:
    from .isa import (
        OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
        OP_LD, OP_ST, OP_MOV,
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
        OP_VLD, OP_VST, OP_VMOV,
        OP_TID, OP_WID, OP_BAR,
        encode_rtype, encode_itype, encode_stype
    )
except ImportError:
    from isa import (
        OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
        OP_LD, OP_ST, OP_MOV,
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
        OP_VLD, OP_VST, OP_VMOV,
        OP_TID, OP_WID, OP_BAR,
        encode_rtype, encode_itype, encode_stype
    )


def assemble(source: str) -> list[int]:
    """将文本汇编代码编译为机器码列表

    对应 GPGPU-Sim 中 ptx_parser 读取 .ptx 文件并生成内部指令表示的过程。

    Args:
        source: 汇编源代码文本

    Returns:
        32-bit 指令字列表 (机器码)

    Raises:
        ValueError: 语法错误或未知指令

    Example:
        >>> src = \"\"\"
        ... MOV r1, 5
        ... MOV r2, 3
        ... ADD r3, r1, r2
        ... HALT
        ... \"\"\"
        >>> program = assemble(src)
        >>> len(program)
        4
    """
    program = []

    for line_num, raw_line in enumerate(source.strip().split('\n'), 1):
        # 预处理：去注释、去空白
        line = raw_line.split('#')[0].split(';')[0].strip()
        if not line:
            continue  # 空行或纯注释

        try:
            word = _assemble_line(line)
            program.append(word)
        except Exception as e:
            raise ValueError(
                f"Line {line_num}: {raw_line}\n  Error: {e}"
            ) from e

    return program


def _assemble_line(line: str) -> int:
    """汇编一行指令"""

    # 标签 (label:)
    if line.endswith(':'):
        return 0  # Phase 0: 标签做 NOP (后续 Phase 用于跳转目标)

    parts = line.replace(',', ' ').split()
    if not parts:
        return 0

    mnemonic = parts[0].upper()

    if mnemonic == 'HALT':
        return encode_rtype(OP_HALT, 0, 0, 0)

    elif mnemonic == 'NOP':
        return encode_rtype(OP_ADD, 0, 0, 0)  # NOP = r0 + r0 → r0

    elif mnemonic in ('ADD', 'SUB', 'MUL', 'DIV'):
        # 格式: OP rd, rs1, rs2
        rd = _parse_reg(parts[1])
        rs1 = _parse_reg(parts[2])
        rs2 = _parse_reg(parts[3])
        opcode = {'ADD': OP_ADD, 'SUB': OP_SUB, 'MUL': OP_MUL, 'DIV': OP_DIV}[mnemonic]
        return encode_rtype(opcode, rd, rs1, rs2)

    elif mnemonic == 'MOV':
        # 格式: MOV rd, imm
        rd = _parse_reg(parts[1])
        imm = _parse_int(parts[2])
        return encode_itype(OP_MOV, rd, imm)

    elif mnemonic == 'LD':
        # 格式: LD rd, [addr]
        rd = _parse_reg(parts[1])
        addr = _parse_mem_addr(parts[2])
        return encode_itype(OP_LD, rd, addr)

    elif mnemonic == 'ST':
        # 格式: ST rs, [addr]
        rs = _parse_reg(parts[1])
        addr = _parse_mem_addr(parts[2])
        return encode_stype(OP_ST, rs, addr)

    # ---- 向量指令 ----
    elif mnemonic in ('VADD', 'VSUB', 'VMUL', 'VDIV'):
        vd = _parse_vreg(parts[1])
        vs1 = _parse_vreg(parts[2])
        vs2 = _parse_vreg(parts[3])
        opcode = {'VADD': OP_VADD, 'VSUB': OP_VSUB,
                  'VMUL': OP_VMUL, 'VDIV': OP_VDIV}[mnemonic]
        return encode_rtype(opcode, vd, vs1, vs2)

    elif mnemonic == 'VMOV':
        vd = _parse_vreg(parts[1])
        imm = _parse_int(parts[2])
        return encode_itype(OP_VMOV, vd, imm)

    elif mnemonic == 'VLD':
        vd = _parse_vreg(parts[1])
        addr = _parse_mem_addr(parts[2])
        return encode_itype(OP_VLD, vd, addr)

    elif mnemonic == 'VST':
        vs = _parse_vreg(parts[1])
        addr = _parse_mem_addr(parts[2])
        return encode_stype(OP_VST, vs, addr)

    # ---- SIMT 指令 ----
    elif mnemonic == 'TID':
        rd = _parse_reg(parts[1])
        return encode_rtype(OP_TID, rd, 0, 0)

    elif mnemonic == 'WID':
        rd = _parse_reg(parts[1])
        return encode_rtype(OP_WID, rd, 0, 0)

    elif mnemonic == 'BAR':
        return encode_rtype(OP_BAR, 0, 0, 0)

    else:
        raise ValueError(f"Unknown instruction: {mnemonic}")


def _parse_reg(token: str) -> int:
    """解析标量寄存器名: 'r0'-'r15' 或 '0'-'15'"""
    token = token.strip().lower()
    if token.startswith('r'):
        reg_id = int(token[1:])
    else:
        reg_id = int(token)
    if reg_id < 0 or reg_id > 15:
        raise ValueError(f"Invalid register: {token} (must be r0-r15)")
    return reg_id


def _parse_vreg(token: str) -> int:
    """解析向量寄存器名: 'v0'-'v7'"""
    token = token.strip().lower()
    if token.startswith('v'):
        reg_id = int(token[1:])
    else:
        reg_id = int(token)
    if reg_id < 0 or reg_id > 7:
        raise ValueError(f"Invalid vector register: {token} (must be v0-v7)")
    return reg_id


def _parse_int(token: str) -> int:
    """解析整数: 支持十进制和 0x 十六进制"""
    token = token.strip()
    if token.startswith('0x') or token.startswith('0X'):
        val = int(token, 16)
    else:
        val = int(token)
    # 限制在 12-bit 有符号范围
    if val < -2048 or val > 2047:
        raise ValueError(f"Immediate out of 12-bit range: {val} (-2048 to 2047)")
    return val & 0xFFF


def _parse_mem_addr(token: str) -> int:
    """解析内存地址: [42] 或 [0xFF]"""
    token = token.strip().strip('[]').strip()
    return _parse_int(token)
