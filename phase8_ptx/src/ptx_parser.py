"""
PTX Parser + Translator — PTX 前端模块
=========================================
对标 GPGPU-Sim cuda-sim/ptx_parser 的 PTX→内部 IR 翻译。

Phase 8: 解析 CUDA PTX 语法子集，翻译为 toygpgpu 内部 ISA。

Pipeline:
  .ptx file → parse → PTX instructions → translate → internal ISA → machine code

参考: NVIDIA PTX ISA 文档 (docs.nvidia.com/cuda/parallel-thread-execution)
      GPGPU-Sim cuda-sim/ptx.l / ptx.y (Lex/Yacc PTX 解析器)
"""

import re
from typing import List, Tuple, Dict, Optional
from isa import (OP_HALT, OP_ADD, OP_MUL, OP_LD, OP_ST, OP_MOV,
                 OP_TID, OP_JMP, OP_BEQ,
                 encode_rtype, encode_itype, encode_stype)


# ============================================================
# 词法分析 (Tokenization)
# ============================================================

TOKEN_RE = re.compile(r"""
    %r\d+          # 虚拟寄存器 %r0, %r1, ...
  | \.reg\s+\.u32  # .reg .u32 声明
  | \.entry        # kernel entry
  | ld\.global\.u32
  | st\.global\.u32
  | mov\.u32
  | add\.u32
  | mul\.lo\.u32
  | bra
  | @%p\d+         # 预测寄存器 @%p0, @%p1
  | %tid\.x|%ntid\.x|%ctaid\.x  # special registers
  | setp\.[a-z]+\.[a-z]+  # setp 比较指令
  | ret
  | \.u32           # 类型修饰
  | \.x             # 维度修饰
  | [a-zA-Z_]\w*:   # label
  | [a-zA-Z_]\w*    # identifier
  | -?\d+           # integer literal
  | 0x[0-9a-fA-F]+  # hex literal
  | \[|\]|,|\{|\}|\(|\)  # punctuation
  | ;               # statement terminator
""", re.VERBOSE)


def tokenize(source: str) -> List[str]:
    """词法分析: PTX 源码 → token 列表"""
    # 去除注释 (; 开头行, // 行注释, 行内 ; 注释)
    clean_lines = []
    for line in source.split('\n'):
        line = line.split('//')[0]  # 去除 // 注释
        # 保留第一个 ; 作为语句终止符, 去除之后的 ; 行内注释
        if ';' in line:
            first = line.index(';')
            line = line[:first+1]  # 保留第一个 ; (语句终止符)
        line = line.strip()
        if line.startswith(';') or not line:
            continue  # 整行 ; 注释 或 空行
        clean_lines.append(line)
    source = '\n'.join(clean_lines)

    tokens = []
    for m in TOKEN_RE.finditer(source):
        token = m.group().strip()
        if token and not token.startswith('//'):
            tokens.append(token)
    return tokens


# ============================================================
# PTX 中间表示
# ============================================================

class PtxInstr:
    """一条 PTX 指令的中间表示"""
    def __init__(self, op: str, operands: list, pred: str = None,
                 label: str = None):
        self.op = op        # e.g., "mov.u32", "add.u32"
        self.operands = operands  # e.g., ["%r1", "5"] or ["%r1", "%r2", "%r3"]
        self.pred = pred    # predicate register, e.g., "%p0"
        self.label = label  # label name (for bra target)


class PtxProgram:
    """PTX 程序的中间表示"""
    def __init__(self):
        self.instructions: List[PtxInstr] = []
        self.labels: Dict[str, int] = {}  # label → instruction index
        self.num_regs = 0  # virtual registers used

    def add(self, instr: PtxInstr, index: int):
        self.instructions.append(instr)
        if instr.label:
            self.labels[instr.label] = index


# ============================================================
# 语法分析 (Parsing)
# ============================================================

def parse_ptx(source: str) -> PtxProgram:
    """解析 PTX 源码 → PtxProgram IR"""
    tokens = tokenize(source)
    prog = PtxProgram()
    i = 0

    while i < len(tokens):
        t = tokens[i]

        # 跳过声明关键字
        if t in ('.reg', '.entry', '.param', '.u32', '.x'):
            # .entry name { → skip .entry, name, {
            i += 1  # skip .entry itself
            # skip the entry name if present
            if i < len(tokens) and not tokens[i].startswith('.') and tokens[i] not in ('{', '}', ';'):
                i += 1
            continue

        # 跳过花括号
        if t in ('{', '}', '(', ')'):
            i += 1
            continue

        # 分号 (单独的分号)
        if t == ';':
            i += 1
            continue

        # Label (xxx:)
        label = None
        if t.endswith(':') and not t.startswith('%') and not t.startswith('@'):
            label = t[:-1]
            i += 1
            # 可能后面跟着指令
            if i < len(tokens) and tokens[i] == ';':
                i += 1
            continue

        # 预测寄存器 @%pX
        pred = None
        if t.startswith('@%p'):
            pred = t[1:]  # strip @, keep %p0
            i += 1
            t = tokens[i] if i < len(tokens) else ''

        # 指令操作码
        op = t
        i += 1

        # 收集操作数 (直到 ;)，合并 [ addr ] 为一个 token
        operands = []
        while i < len(tokens) and tokens[i] != ';':
            tok = tokens[i]
            if tok == ',':
                i += 1
                continue
            # Merge [ N ] into [N]
            if tok == '[':
                addr_tok = tokens[i+1] if i+1 < len(tokens) else '0'
                operands.append(f'[{addr_tok}]')
                i += 3  # skip [, N, ]
                continue
            operands.append(tok)
            i += 1

        # 跳过分号
        if i < len(tokens) and tokens[i] == ';':
            i += 1

        # 跳过 standalone ret
        if op == 'ret':
            op = 'ret'
            operands = []

        instr = PtxInstr(op=op, operands=operands, pred=pred, label=label)
        idx = len(prog.instructions)
        prog.add(instr, idx)

        # Track virtual register usage
        for o in operands:
            if o.startswith('%r'):
                try:
                    r = int(o[2:])
                    prog.num_regs = max(prog.num_regs, r + 1)
                except ValueError:
                    pass

    return prog


# ============================================================
# 翻译: PTX → toygpgpu Internal ISA
# ============================================================

class RegisterAllocator:
    """简单线性扫描寄存器分配器

    虚拟寄存器 %r0..%rN 映射到物理寄存器 r1..r10。
    r0 保留为 0。
    """
    def __init__(self, num_virtual: int, max_physical: int = 10):
        self.map: Dict[str, int] = {}
        self.next_phys = 1
        self.max_phys = max_physical

        # 硬映射: tid.x 特殊处理
        self.tid_temp_reg = None

    def alloc(self, vreg: str) -> int:
        """分配物理寄存器"""
        if vreg in self.map:
            return self.map[vreg]
        reg_id = int(vreg[2:])  # %rX → X
        # 简单分配: 虚拟 reg_id + 1 = 物理寄存器 (vreg %r0 → phys r1)
        phys = reg_id + 1
        if phys > self.max_phys:
            phys = (reg_id % self.max_phys) + 1
        self.map[vreg] = phys
        return phys

    def get(self, vreg: str) -> Optional[int]:
        return self.map.get(vreg)


def translate_ptx(ptx_prog: PtxProgram) -> Tuple[str, int]:
    """翻译 PTX IR → toygpgpu 汇编文本

    Returns:
        (assembly_text, num_instructions)
    """
    allocator = RegisterAllocator(ptx_prog.num_regs)
    lines = []
    label_pcs = {}  # label → PC (for branch targets)

    # First pass: determine PC for each label
    pc = 0
    for instr in ptx_prog.instructions:
        if instr.label:
            label_pcs[instr.label] = pc
        pc += _estimate_instructions(instr)

    # Second pass: emit code
    pc = 0
    for instr in ptx_prog.instructions:
        if instr.label:
            lines.append(f"{instr.label}:")

        emitted = _translate_instr(instr, allocator, label_pcs, pc)
        if emitted:
            lines.extend(emitted)
            pc += len(emitted)

    return '\n'.join(lines), pc


def _estimate_instructions(instr: PtxInstr) -> int:
    """估算一条 PTX 指令需要多少条内部指令"""
    op = instr.op
    if op == 'setp.ne.u32': return 3  # SUB + set based on result
    return 1


def _translate_instr(instr: PtxInstr, alloc: RegisterAllocator,
                     labels: Dict[str, int], pc: int) -> List[str]:
    """翻译单条 PTX 指令 → 汇编代码行列表"""
    op = instr.op
    ops = instr.operands

    if op == 'ret':
        return ['HALT']

    if op in ('mov.u32',):
        if len(ops) == 2:
            # mov.u32 %rd, %rs  or  mov.u32 %rd, imm
            rd = alloc.alloc(ops[0])
            src = ops[1]
            if src.startswith('%r'):
                # Register move: ADD rd, rs, r0
                rs = alloc.alloc(src)
                return [f'ADD r{rd}, r{rs}, r0']
            elif src == '%tid.x':
                return [f'TID r{rd}']
            elif src == '%ntid.x':
                # Block dim — simplified
                return [f'MOV r{rd}, 8']
            elif src.startswith('%ctaid'):
                return [f'WID r{rd}']
            else:
                # Immediate
                val = _parse_int(src)
                return [f'MOV r{rd}, {val}']
        return []

    if op == 'add.u32':
        if len(ops) == 3:
            rd = alloc.alloc(ops[0])
            rs1 = alloc.alloc(ops[1])
            rs2 = alloc.alloc(ops[2])
            return [f'ADD r{rd}, r{rs1}, r{rs2}']
        return []

    if op == 'mul.lo.u32':
        if len(ops) == 3:
            rd = alloc.alloc(ops[0])
            rs1 = alloc.alloc(ops[1])
            rs2 = alloc.alloc(ops[2])
            return [f'MUL r{rd}, r{rs1}, r{rs2}']
        return []

    if op == 'ld.global.u32':
        if len(ops) == 2:
            rd = alloc.alloc(ops[0])
            base_src = ops[1].strip('[]')
            if base_src.startswith('%r'):
                # 寄存器基地址 → 简化为使用基地址值作为偏移
                # 在 PTX 中通常是基址+偏移(structured), 这里简化处理
                rs = alloc.alloc(base_src)
                return [f'MOV r{rd}, 0']
            else:
                # 立即数地址
                try:
                    addr = _parse_int(base_src)
                    return [f'LD r{rd}, [{addr}]']
                except ValueError:
                    return [f'# PTX: ld {ops}']
        return []

    if op == 'st.global.u32':
        if len(ops) == 2:
            target = ops[0].strip('[]')
            src = ops[1]
            if src.startswith('%r'):
                rs = alloc.alloc(src)
                try:
                    addr = _parse_int(target)
                    return [f'ST r{rs}, [{addr}]']
                except ValueError:
                    if target.startswith('%r'):
                        # Register-based address — simplified
                        return [f'ST r{rs}, [0]']
        return []

    if op == 'st.global.u32':
        if len(ops) == 2:
            # st.global.u32 [%rd], %rs
            target = ops[0].strip('[]')
            src = ops[1]
            if src.startswith('%r'):
                rs = alloc.alloc(src)
                if target.startswith('%r'):
                    # Store to register-based address (use fixed offset as simplification)
                    return [f'ST r{rs}, [0]']
                else:
                    addr = int(target)
                    return [f'ST r{rs}, [{addr}]']
        return []

    if op == 'setp.ne.u32':
        # setp.ne.u32 %pd, %a, %b → 比较 a != b
        # 简化: SUB a, b → if result != 0 then pred true
        if len(ops) >= 3:
            pd = ops[0]
            a = ops[1]
            b = ops[2]
            # 用临时寄存器做减法
            # 映射: %pd → 物理寄存器 (分配)
            if a.startswith('%r') and b.startswith('%r'):
                ra = alloc.alloc(a)
                rb = alloc.alloc(b)
                rp = alloc.alloc(pd)
                # SUB rp, ra, rb; check non-zero → pred
                return [f'SUB r{rp}, r{ra}, r{rb}']
        return []

    if op == 'bra':
        # 无条件跳转
        if len(ops) == 1:
            target = ops[0]
            if target in labels:
                offset = labels[target] - (pc + 1)
                return [f'JMP {target}']
        return []

    # Predicated branch: @%px bra target
    if instr.pred and op == 'bra':
        if ops and ops[0] in labels:
            target = ops[0]
            # BEQ pred_reg, r0, target (if pred_reg == 0, branch)
            # Actually: pred true → branch. So BNE pred_reg, r0, target
            # Wait: pred is 0 when false. So BNE pred, 0, target = branch if pred!=0
            if instr.pred.startswith('%p'):
                p_reg = alloc.alloc(instr.pred)
                return [f'BNE r{p_reg}, r0, {target}']
        return []

    # Default: comment
    return [f'# PTX: {instr.op} {" ".join(instr.operands)}']


def _parse_int(s: str) -> int:
    """解析整数"""
    s = s.strip()
    if s.startswith('0x'):
        return int(s, 16)
    return int(s)


def assemble_ptx(source: str) -> Tuple[List[int], str]:
    """完整的 PTX → 机器码 编译管道

    Returns:
        (machine_code_list, assembly_text)
    """
    ptx_prog = parse_ptx(source)
    asm_text, _ = translate_ptx(ptx_prog)
    from assembler import assemble
    machine_code = assemble(asm_text)
    return machine_code, asm_text
