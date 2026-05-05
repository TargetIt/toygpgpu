"""
Assembler — 汇编器 (Phase 3: + Label + 分支指令)
===================================================
Two-pass assembler with label support for branch targets.

Pass 1: collect labels → PC mapping
Pass 2: emit machine code with resolved label references

参考: GPGPU-Sim cuda-sim/ptx_parser.cc
      Standard two-pass assembler pattern
"""
try:
    from .isa import (
        OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
        OP_LD, OP_ST, OP_MOV,
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
        OP_VLD, OP_VST, OP_VMOV,
        OP_TID, OP_WID, OP_BAR,
        OP_JMP, OP_BEQ, OP_BNE,
        encode_rtype, encode_itype, encode_stype
    )
except ImportError:
    from isa import (
        OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
        OP_LD, OP_ST, OP_MOV,
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
        OP_VLD, OP_VST, OP_VMOV,
        OP_TID, OP_WID, OP_BAR,
        OP_JMP, OP_BEQ, OP_BNE,
        encode_rtype, encode_itype, encode_stype
    )


def assemble(source: str) -> list[int]:
    """Two-pass 汇编: Pass1 收集 label, Pass2 生成代码"""
    lines = []
    for raw_line in source.strip().split('\n'):
        line = raw_line.split('#')[0].split(';')[0].strip()
        if line:
            lines.append(line)

    # Pass 1: 收集 label→PC 映射
    labels = {}
    pc = 0
    for line in lines:
        if line.endswith(':'):
            labels[line[:-1].strip()] = pc
        else:
            pc += 1

    # Pass 2: 生成机器码
    program = []
    pc = 0
    for line_num, line in enumerate(lines, 1):
        if line.endswith(':'):
            continue  # 标签不占指令空间

        try:
            word = _assemble_line(line, labels, pc)
            program.append(word)
            pc += 1
        except Exception as e:
            raise ValueError(f"Line {line_num}: {line}\n  Error: {e}") from e

    return program


def _assemble_line(line: str, labels: dict, current_pc: int) -> int:
    """汇编一行指令 (含 label 解析)"""
    parts = line.replace(',', ' ').split()
    mnemonic = parts[0].upper()

    # ---- 标量 ----
    if mnemonic == 'HALT':     return encode_rtype(OP_HALT, 0, 0, 0)
    if mnemonic == 'NOP':      return encode_rtype(OP_ADD, 0, 0, 0)

    if mnemonic in ('ADD', 'SUB', 'MUL', 'DIV'):
        rd, rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2]), _parse_reg(parts[3])
        opcode = {'ADD': OP_ADD, 'SUB': OP_SUB, 'MUL': OP_MUL, 'DIV': OP_DIV}[mnemonic]
        return encode_rtype(opcode, rd, rs1, rs2)

    if mnemonic == 'MOV':
        return encode_itype(OP_MOV, _parse_reg(parts[1]), _parse_int(parts[2]))

    if mnemonic == 'LD':
        return encode_itype(OP_LD, _parse_reg(parts[1]), _parse_mem_addr(parts[2]))

    if mnemonic == 'ST':
        return encode_stype(OP_ST, _parse_reg(parts[1]), _parse_mem_addr(parts[2]))

    # ---- 向量 ----
    if mnemonic in ('VADD', 'VSUB', 'VMUL', 'VDIV'):
        vd, vs1, vs2 = _parse_vreg(parts[1]), _parse_vreg(parts[2]), _parse_vreg(parts[3])
        opcode = {'VADD': OP_VADD, 'VSUB': OP_VSUB, 'VMUL': OP_VMUL, 'VDIV': OP_VDIV}[mnemonic]
        return encode_rtype(opcode, vd, vs1, vs2)

    if mnemonic == 'VMOV':
        return encode_itype(OP_VMOV, _parse_vreg(parts[1]), _parse_int(parts[2]))
    if mnemonic == 'VLD':
        return encode_itype(OP_VLD, _parse_vreg(parts[1]), _parse_mem_addr(parts[2]))
    if mnemonic == 'VST':
        return encode_stype(OP_VST, _parse_vreg(parts[1]), _parse_mem_addr(parts[2]))

    # ---- SIMT ----
    if mnemonic == 'TID':      return encode_rtype(OP_TID, _parse_reg(parts[1]), 0, 0)
    if mnemonic == 'WID':      return encode_rtype(OP_WID, _parse_reg(parts[1]), 0, 0)
    if mnemonic == 'BAR':      return encode_rtype(OP_BAR, 0, 0, 0)

    # ---- 分支 (Phase 3) ----
    if mnemonic == 'JMP':
        target_pc = labels[_parse_label(parts[1])]
        offset = target_pc - (current_pc + 1)  # 相对偏移
        return encode_rtype(OP_JMP, 0, 0, 0, offset & 0xFFF)
    if mnemonic in ('BEQ', 'BNE'):
        rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2])
        target_pc = labels[_parse_label(parts[3])]
        offset = target_pc - (current_pc + 1)
        opcode = OP_BEQ if mnemonic == 'BEQ' else OP_BNE
        return encode_rtype(opcode, 0, rs1, rs2, offset & 0xFFF)

    raise ValueError(f"Unknown instruction: {mnemonic}")


# ---- Parse helpers ----
def _parse_reg(t): t = t.strip().lower(); r = int(t[1:]) if t.startswith('r') else int(t); return r
def _parse_vreg(t): t = t.strip().lower(); return int(t[1:]) if t.startswith('v') else int(t)
def _parse_label(t): return t.strip()
def _parse_mem_addr(t): return _parse_int(t.strip().strip('[]'))

def _parse_int(token: str) -> int:
    token = token.strip()
    val = int(token, 16) if token.startswith('0x') or token.startswith('0X') else int(token)
    if val < -2048 or val > 2047:
        raise ValueError(f"Immediate out of 12-bit range: {val}")
    return val & 0xFFF
