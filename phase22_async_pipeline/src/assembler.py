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
        OP_JMP, OP_BEQ, OP_BNE, OP_SETP, PRED_FLAG,
        OP_SHLD, OP_SHST,
        OP_WREAD, OP_WWRITE, WREG_NAMES,
        OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
        OP_SHFL, OP_VOTE, OP_BALLOT,
        OP_TLCONF, OP_TLDS, OP_TLSTS, OP_WGMMA, OP_CPYASYNC, OP_TMA_LOAD, OP_TMA_STORE,
        OP_MMA, encode_mma,
        encode_rtype, encode_itype, encode_stype
    )
except ImportError:
    from isa import (
        OP_MMA, encode_mma,
        OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
        OP_LD, OP_ST, OP_MOV,
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV,
        OP_VLD, OP_VST, OP_VMOV,
        OP_TID, OP_WID, OP_BAR,
        OP_JMP, OP_BEQ, OP_BNE, OP_SETP, PRED_FLAG,
        OP_SHLD, OP_SHST,
        OP_WREAD, OP_WWRITE, WREG_NAMES,
        OP_V4PACK, OP_V4ADD, OP_V4MUL, OP_V4UNPACK,
        OP_SHFL, OP_VOTE, OP_BALLOT,
        OP_TLCONF, OP_TLDS, OP_TLSTS, OP_WGMMA, OP_CPYASYNC, OP_TMA_LOAD, OP_TMA_STORE,
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
    # 处理 @p0 谓词前缀
    predicated = False
    if line.startswith('@p0'):
        predicated = True
        line = line[3:].strip()

    parts = line.replace(',', ' ').split()
    mnemonic = parts[0].upper()

    # ---- 标量 ----
    if mnemonic == 'HALT':     return _pred(encode_rtype(OP_HALT, 0, 0, 0), predicated)
    if mnemonic == 'NOP':      return _pred(encode_rtype(OP_ADD, 0, 0, 0), predicated)

    if mnemonic in ('ADD', 'SUB', 'MUL', 'DIV'):
        rd, rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2]), _parse_reg(parts[3])
        opcode = {'ADD': OP_ADD, 'SUB': OP_SUB, 'MUL': OP_MUL, 'DIV': OP_DIV}[mnemonic]
        return _pred(encode_rtype(opcode, rd, rs1, rs2), predicated)

    if mnemonic == 'MOV':
        return _pred(encode_itype(OP_MOV, _parse_reg(parts[1]), _parse_int(parts[2])), predicated)

    if mnemonic == 'LD':
        return _pred(encode_itype(OP_LD, _parse_reg(parts[1]), _parse_mem_addr(parts[2])), predicated)
    if mnemonic == 'ST':
        return _pred(encode_stype(OP_ST, _parse_reg(parts[1]), _parse_mem_addr(parts[2])), predicated)

    # ---- 共享内存 ----
    if mnemonic == 'SHLD':
        return _pred(encode_itype(OP_SHLD, _parse_reg(parts[1]), _parse_mem_addr(parts[2])), predicated)
    if mnemonic == 'SHST':
        return _pred(encode_stype(OP_SHST, _parse_reg(parts[1]), _parse_mem_addr(parts[2])), predicated)

    # ---- 向量 ----
    if mnemonic in ('VADD', 'VSUB', 'VMUL', 'VDIV'):
        vd, vs1, vs2 = _parse_vreg(parts[1]), _parse_vreg(parts[2]), _parse_vreg(parts[3])
        opcode = {'VADD': OP_VADD, 'VSUB': OP_VSUB, 'VMUL': OP_VMUL, 'VDIV': OP_VDIV}[mnemonic]
        return _pred(encode_rtype(opcode, vd, vs1, vs2), predicated)

    if mnemonic == 'VMOV':
        return _pred(encode_itype(OP_VMOV, _parse_vreg(parts[1]), _parse_int(parts[2])), predicated)
    if mnemonic == 'VLD':
        return _pred(encode_itype(OP_VLD, _parse_vreg(parts[1]), _parse_mem_addr(parts[2])), predicated)
    if mnemonic == 'VST':
        return _pred(encode_stype(OP_VST, _parse_vreg(parts[1]), _parse_mem_addr(parts[2])), predicated)

    # ---- SIMT ----
    if mnemonic == 'TID':      return _pred(encode_rtype(OP_TID, _parse_reg(parts[1]), 0, 0), predicated)
    if mnemonic == 'WID':      return _pred(encode_rtype(OP_WID, _parse_reg(parts[1]), 0, 0), predicated)
    if mnemonic == 'BAR':      return _pred(encode_rtype(OP_BAR, 0, 0, 0), predicated)

    # ---- Warp-level register (WREAD/WWRITE) ----
    if mnemonic == 'WREAD':
        rd = _parse_reg(parts[1])
        wreg_name = parts[2].lower()
        wreg_idx = WREG_NAMES.get(wreg_name, 0)
        return encode_rtype(OP_WREAD, rd, 0, 0, wreg_idx)
    if mnemonic == 'WWRITE':
        rs1 = _parse_reg(parts[1])
        wreg_name = parts[2].lower()
        wreg_idx = WREG_NAMES.get(wreg_name, 0)
        return encode_rtype(OP_WWRITE, 0, rs1, 0, wreg_idx)

    # ---- 谓词 (Phase 3: Predication) ----
    if mnemonic == 'SETP.EQ':
        rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2])
        return encode_rtype(OP_SETP, 0, rs1, rs2, 0)  # imm[0]=0 → EQ
    if mnemonic == 'SETP.NE':
        rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2])
        return encode_rtype(OP_SETP, 0, rs1, rs2, 1)  # imm[0]=1 → NE

    # ---- 分支 (Phase 3) ----
    if mnemonic == 'JMP':
        target_pc = labels[_parse_label(parts[1])]
        offset = target_pc - (current_pc + 1)  # 相对偏移
        return _pred(encode_rtype(OP_JMP, 0, 0, 0, offset & 0xFFF), predicated)
    if mnemonic in ('BEQ', 'BNE'):
        rs1, rs2 = _parse_reg(parts[1]), _parse_reg(parts[2])
        target_pc = labels[_parse_label(parts[3])]
        offset = target_pc - (current_pc + 1)
        opcode = OP_BEQ if mnemonic == 'BEQ' else OP_BNE
        return _pred(encode_rtype(opcode, 0, rs1, rs2, offset & 0xFFF), predicated)

    # ---- Tensor Core ----
    if mnemonic == 'MMA':
        rd = _parse_reg(parts[1])
        rs1 = _parse_reg(parts[2])
        rs2 = _parse_reg(parts[3])
        rs3 = _parse_reg(parts[4])
        return _pred(encode_mma(rd, rs1, rs2, rs3), predicated)


    # ---- Vec4 instructions (Phase 1+ extension) ----
    if mnemonic in ('V4ADD', 'V4MUL'):
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2]); rs2 = _parse_reg(parts[3])
        opcode = {'V4ADD': OP_V4ADD, 'V4MUL': OP_V4MUL}[mnemonic]
        return _pred(encode_rtype(opcode, rd, rs1, rs2), predicated)
    if mnemonic == 'V4PACK':
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2]); rs2 = _parse_reg(parts[3])
        return _pred(encode_rtype(OP_V4PACK, rd, rs1, rs2), predicated)
    if mnemonic == 'V4UNPACK':
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2])
        lane = _parse_int(parts[3])
        return _pred(encode_rtype(OP_V4UNPACK, rd, rs1, lane), predicated)

    # ---- Warp Communication (Phase 12) ----
    if mnemonic == 'SHFL':
        rd = _parse_reg(parts[1])
        rs1 = _parse_reg(parts[2])
        src_lane = _parse_int(parts[3])  # source lane id
        mode = _parse_int(parts[4]) if len(parts) > 4 else 0  # 0=idx,1=up,2=down,3=xor
        return encode_rtype(OP_SHFL, rd, rs1, src_lane, mode)
    if mnemonic == 'VOTE.ANY':
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2])
        return encode_rtype(OP_VOTE, rd, rs1, 0, 0)  # imm[0]=0 → ANY
    if mnemonic == 'VOTE.ALL':
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2])
        return encode_rtype(OP_VOTE, rd, rs1, 0, 1)  # imm[0]=1 → ALL
    if mnemonic == 'BALLOT':
        rd = _parse_reg(parts[1]); rs1 = _parse_reg(parts[2])
        return encode_rtype(OP_BALLOT, rd, rs1, 0)

    # ---- Tiling (Phase 13) ----
    if mnemonic == 'TLCONF':
        tile_m = _parse_int(parts[1])   # tile M dimension
        tile_n = _parse_int(parts[2])   # tile N dimension
        tile_k = _parse_int(parts[3])   # tile K dimension
        return encode_rtype(OP_TLCONF, tile_m, tile_n, 0, tile_k)
    if mnemonic == 'TLDS':
        smem_off = _parse_int(parts[1])  # shared memory offset
        glob_base = _parse_int(parts[2]) # global memory base
        return encode_rtype(OP_TLDS, smem_off, glob_base, 0)
    if mnemonic == 'TLSTS':
        smem_off = _parse_int(parts[1])  # shared memory offset
        glob_base = _parse_int(parts[2]) # global memory base
        return encode_rtype(OP_TLSTS, smem_off, glob_base, 0)

    # ---- Warp-Group MMA (Phase 14) ----
    if mnemonic == 'WGMMA':
        smem_a = _parse_int(parts[1])  # smem offset for A tile
        smem_b = _parse_int(parts[2])  # smem offset for B tile
        smem_c = _parse_int(parts[3])  # smem offset for C (accumulator)
        return encode_rtype(OP_WGMMA, smem_a, smem_b, smem_c)

    
    # ---- Async Copy & Events (Phase 17) ----
    if mnemonic == 'CPYASYNC':
        src = _parse_int(parts[1]); dst = _parse_int(parts[2]); size = _parse_int(parts[3])
        return encode_rtype(OP_CPYASYNC, OP_TMA_LOAD, OP_TMA_STORE, src, dst, 0, size)
    if mnemonic == 'RECORD':
        ev_id = _parse_int(parts[1])
        return encode_rtype(OP_RECORD, 0, 0, 0, ev_id)
    if mnemonic == 'WAIT':
        ev_id = _parse_int(parts[1])
        return encode_rtype(OP_WAIT, 0, 0, 0, ev_id)

    
    # ---- TMA (Phase 20) ----
    if mnemonic == 'TMA_LOAD':
        desc_id = _parse_int(parts[1]); smem_off = _parse_int(parts[2])
        start = _parse_int(parts[3]) if len(parts) > 3 else 0
        return encode_rtype(OP_TMA_LOAD, desc_id, smem_off, 0, start)
    if mnemonic == 'TMA_STORE':
        desc_id = _parse_int(parts[1]); smem_off = _parse_int(parts[2])
        start = _parse_int(parts[3]) if len(parts) > 3 else 0
        return encode_rtype(OP_TMA_STORE, desc_id, smem_off, 0, start)

    raise ValueError(f"Unknown instruction: {mnemonic}")


def _pred(word: int, predicated: bool) -> int:
    """若谓词化, 在指令字中设置 PRED_FLAG (bit 11)"""
    return word | PRED_FLAG if predicated else word

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
