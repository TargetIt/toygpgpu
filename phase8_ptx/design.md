# Phase 8: PTX Frontend / PTX 前端 — Design Document / 设计文档

> **对应 GPGPU-Sim**: `cuda-sim/ptx_parser.cc`, `cuda-sim/ptx.l / ptx.y`
> **参考**: NVIDIA PTX ISA 文档 (docs.nvidia.com/cuda/parallel-thread-execution), RISC-V RV32I 指令格式

## 1. Introduction / 架构概览

```
           ┌─────────────────────────────────────────────────────────────────┐
           │                     PTX Compilation Pipeline                     │
           │  .ptx file → Tokenize → Parse → PTX IR → Translate → .asm      │
           │                                                  ↓              │
           │                                           Assemble → Machine Code│
           │                                                  ↓              │
           │                                           SIMT Core Execute     │
           └─────────────────────────────────────────────────────────────────┘
                                                                                
           ┌─────────────────────────────────────────────────────────────────┐
           │                     Module Architecture                          │
           │  ┌─────────────┐  ┌──────────┐  ┌──────────────┐               │
           │  │ ptx_parser  │  │ assembler│  │  simt_core   │               │
           │  │ .py         │→│ .py      │→│  .py         │               │
           │  │ (tokenize,  │  │ (two-pass│  │ (execute     │               │
           │  │  parse,     │  │  encode)  │  │  pipeline)   │               │
           │  │  translate) │  └──────────┘  └──────────────┘               │
           │  └─────────────┘                 ┌──────────────┐               │
           │  ┌─────────────┐                 │learning_     │               │
           │  │    isa.py   │←────────────────│console.py    │               │
           │  │ (opcodes,   │                 │(interactive  │               │
           │  │  encode/    │                 │ debugger)    │               │
           │  │  decode)    │                 └──────────────┘               │
           │  └─────────────┘                                               │
           └─────────────────────────────────────────────────────────────────┘
```

Phase 8 是 toygpgpu 的最高级编程接口。它实现了一个 PTX (Parallel Thread Execution) 前端编译器，可将 NVIDIA CUDA 风格的 PTX 汇编代码编译为内部 ISA 并执行。这是真正 GPU 编程模型的抽象层次——开发者编写 PTX，编译器将其翻译为硬件指令。

Phase 8 is the highest-level programming interface of toygpgpu. It implements a PTX (Parallel Thread Execution) frontend compiler that translates NVIDIA CUDA-style PTX assembly into the internal ISA for execution. This is the abstraction level of real GPU programming — developers write PTX, the compiler translates it to hardware instructions.

## 2. Motivation / 设计动机

Real NVIDIA GPUs compile PTX (Parallel Thread Execution) to SASS (Streaming Assembly) — the final machine code. PTX is the intermediate representation used by CUDA at the compiler level. In GPGPU-Sim, the `ptx_parser.cc` module parses PTX instructions and translates them into the simulator's internal representation.

在真实 NVIDIA GPU 中，PTX 被编译为 SASS（最终机器码）。PTX 是 CUDA 在编译器层面使用的中间表示。GPGPU-Sim 中的 `ptx_parser.cc` 模块解析 PTX 指令并将其翻译为模拟器内部表示。

Before Phase 8, the project could only run hand-written `.asm` files with direct opcode mnemonics. This phase adds:
- A PTX lexer/tokenizer supporting `.entry`, `.reg`, special registers (`%tid.x`, `%ntid.x`, `%ctaid.x`)
- A parser producing `PtxProgram` / `PtxInstr` IR
- A translator mapping PTX operations (mov.u32, add.u32, ld.global.u32, etc.) to internal ISA
- A register allocator mapping virtual `%r0..%rN` to physical `r1..r10`

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Lexical Analysis / 词法分析

Tokenization uses regex-based pattern matching. The `TOKEN_RE` pattern matches:

```
%r\d+          → virtual register
\.reg\.u32     → register declaration
\.entry        → kernel entry point
ld\.global\.u32 → global load
st\.global\.u32 → global store
mov\.u32/add\.u32/mul\.lo\.u32 → operations
@%p\d+         → predicate register
%tid\.x|%ntid\.x|%ctaid\.x → special registers
[a-zA-Z_]\w+:  → label
-?\d+|0x[0-9a-fA-F]+ → literals
\[|\]|,|\{|\}|\(|\)|; → punctuation
```

Comment handling (FIXED in Phase 8):
```python
def tokenize(source):
    # Strip // line comments
    # Strip inline ; comments (keep first ; as terminator, discard rest)
    # Skip ;-only lines
    for line in source.split('\n'):
        line = line.split('//')[0]          # remove // comments
        if ';' in line:
            first = line.index(';')
            line = line[:first+1]           # keep first ; as terminator
        line = line.strip()
        if line.startswith(';') or not line:
            continue
        clean_lines.append(line)
```

### 3.2 Parsing / 语法解析

The parser converts token lists to `PtxProgram` IR:

```
parse_ptx(tokens):
  for each token:
    skip .entry .reg .param .u32 .x
    skip { } ( ) ;
    if token ends with ':' → label (store mapping)
    if token starts with '@%p' → predicate (strip @)
    collect opcode
    collect operands until ';'
      merge [ N ] → [N]
      skip commas
    create PtxInstr(op, operands, pred, label)
    add to PtxProgram
```

**Key data structures:**

```python
class PtxInstr:
    op: str          # e.g., "mov.u32", "add.u32"
    operands: list   # e.g., ["%r1", "5"] or ["%r1", "%r2", "%r3"]
    pred: str        # predicate register, e.g., "%p0"
    label: str       # label name for branch targets

class PtxProgram:
    instructions: list[PtxInstr]
    labels: dict[str, int]      # label → instruction index
    num_regs: int               # virtual register count
```

### 3.3 Translation / 指令翻译

Two-pass translator:

**Pass 1**: Determine PC for each label (estimate instruction expansion).
**Pass 2**: Emit internal ISA assembly.

```python
def translate_instr(instr, alloc, labels, pc):
    match instr.op:
        'ret'                         → HALT
        'mov.u32' rd, %tid.x          → TID rd
        'mov.u32' rd, %ntid.x         → MOV rd, 8
        'mov.u32' rd, %ctaid.x        → WID rd
        'mov.u32' rd, imm             → MOV rd, imm
        'mov.u32' rd, %rs             → ADD rd, rs, r0
        'add.u32' rd, rs1, rs2        → ADD rd, rs1, rs2
        'mul.lo.u32' rd, rs1, rs2     → MUL rd, rs1, rs2
        'ld.global.u32' rd, [addr]    → LD rd, [addr]
        'st.global.u32' [addr], rs    → ST rs, [addr]
        'setp.ne.u32' pd, a, b        → SUB rp, ra, rb
        'bra' label                   → JMP label
        '@%px bra' label              → BNE rp, r0, label
```

### 3.4 Register Allocation / 寄存器分配

Simple linear-scan allocator mapping virtual `%rN` to physical `r(N+1)`, with r0 reserved as zero:

```python
class RegisterAllocator:
    def alloc(self, vreg: str) -> int:
        reg_id = int(vreg[2:])           # %rX → X
        phys = reg_id + 1                # %r0 → r1, %r1 → r2, ...
        if phys > self.max_phys:         # wrap around if > 10
            phys = (reg_id % self.max_phys) + 1
        self.map[vreg] = phys
        return phys
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Purpose / 用途 |
|------|---------------|
| `ptx_parser.py` | Tokenize, parse PTX, translate to ISA, full `assemble_ptx()` pipeline |
| `isa.py` | Opcode definitions (OP_ADD, OP_MOV, OP_LD, etc.), encode/decode |
| `assembler.py` | Two-pass assembler: label resolution, opcode encoding |
| `simt_core.py` | SIMT core: pipeline, warp execution, memory, branching |
| `warp.py` | Warp + Thread classes, register file, SIMT stack, IBuffer |
| `alu.py` | Scalar ALU operations (add, sub, mul, div) |
| `vec4_alu.py` | Vec4 8-bit SIMD operations (pack, add, mul, unpack) |
| `memory.py` | Global memory (word-addressable) |
| `cache.py` | L1 data cache with cache lines |
| `register_file.py` | Per-thread register file (16 x 32-bit) |
| `scoreboard.py` | Scoreboard: WAW/RAW hazard detection, pipeline latency tracking |
| `scheduler.py` | Warp scheduler: round-robin warp selection |
| `ibuffer.py` | I-Buffer: per-warp instruction buffer (2 entries), decouple fetch/issue |
| `simt_stack.py` | SIMT stack: branch divergence/reconvergence management |
| `operand_collector.py` | Operand collector: multi-bank register file access |
| `thread_block.py` | Thread block: shared memory allocation |
| `shared_memory.py` | Shared memory (low-latency on-chip scratchpad) |
| `gpu_sim.py` | Top-level GPU simulator, kernel launch, performance counters |
| `learning_console.py` | Interactive debugger with step/run/breakpoint, PTX source display |

### 4.2 Key Data Structures / 关键数据结构

```python
class PtxInstr:
    op: str              # PTX opcode string
    operands: list[str]  # operand list
    pred: str | None     # predicate register (e.g., "%p0")
    label: str | None    # label name

class PtxProgram:
    instructions: list[PtxInstr]
    labels: dict[str, int]
    num_regs: int

class RegisterAllocator:
    map: dict[str, int]  # virtual → physical register
    max_phys: int        # max physical registers (default 10)
    next_phys: int       # next free physical register
```

### 4.3 Module Interfaces / 模块接口

```python
# ptx_parser.py
def tokenize(source: str) -> list[str]
def parse_ptx(source: str) -> PtxProgram
def translate_ptx(prog: PtxProgram) -> tuple[str, int]  # (asm_text, num_instrs)
def assemble_ptx(source: str) -> tuple[list[int], str]  # (machine_code, asm_text)

# learning_console.py (new PTX commands)
def print_ptx_info(ptx_text, ptx_asm_text)  # Show PTX source + ISA translation
```

## 5. Functional Processing Flow / 功能处理流程

Execution flow for `01_vector_add.ptx`:

```
=== Step-by-step Pipeline Trace ===

Step 1: Load and parse PTX source
  Input: 01_vector_add.ptx
  ├─ mov.u32 %r0, %tid.x
  ├─ ld.global.u32 %r1, [0]
  ├─ ld.global.u32 %r2, [8]
  ├─ add.u32 %r3, %r1, %r2
  ├─ st.global.u32 [16], %r3
  └─ ret

Step 2: Translate to internal ISA
  ┌─────────────────────────────────────────────┐
  │ PTX                        ISA              │
  ├─────────────────────────────────────────────┤
  │ mov.u32 %r0, %tid.x   →   TID r1            │
  │ ld.global.u32 %r1, [0] →   LD r2, [0]       │
  │ ld.global.u32 %r2, [8] →   LD r3, [8]       │
  │ add.u32 %r3, %r1, %r2  →   ADD r4, r2, r3  │
  │ st.global.u32 [16], %r3 →   ST r4, [16]     │
  │ ret                     →   HALT             │
  └─────────────────────────────────────────────┘

Step 3: Execute (warp_size=4, A=[10,20,30,40], B=[1,2,3,4])
  Cycle 0: FETCH PC=0 → IBuffer | ISSUE: TID r1 → regs[0..3]=0,1,2,3
  Cycle 1: FETCH PC=1 → IBuffer | ISSUE: LD r2, [0] → mem[0]=10 broadcast
  Cycle 2: FETCH PC=2 → IBuffer | ISSUE: LD r3, [8] → mem[8]=1 broadcast
  Cycle 3: FETCH PC=3 → IBuffer | ISSUE: ADD r4, r2, r3
  Cycle 4: FETCH PC=4 → IBuffer | ISSUE: ST r4, [16]
  Cycle 5: FETCH PC=5 → IBuffer | ISSUE: HALT → warp done

Step 4: Final memory state
  mem[16] = 11  → C[0] = A[0] + B[0] = 10 + 1
  mem[17] = 22  → C[1] = A[1] + B[1] = 20 + 2
  mem[18] = 33  → C[2] = A[2] + B[2] = 30 + 3
  mem[19] = 44  → C[3] = A[3] + B[3] = 40 + 4
```

## 6. Comparison with Phase 7 / 与 Phase 7 的对比

| Aspect / 方面 | Phase 7 | Phase 8 | Change / 变化 |
|---------------|---------|---------|---------------|
| **Program Input** | Direct `.asm` assembly with r0..rN registers | `.ptx` + `.asm` with `%r0..%rN`, `.entry`, `.reg` | Added PTX frontend compiler |
| **Parser** | None (assembler only) | `ptx_parser.py`: tokenize, parse, translate | NEW module |
| **IR** | None | `PtxProgram`, `PtxInstr` classes | NEW data structures |
| **Register Allocation** | Manual by user | `RegisterAllocator`: virtual → physical mapping | NEW algorithm |
| **Special Registers** | None | `%tid.x`, `%ntid.x`, `%ctaid.x` → TID/MOV/WID | NEW support |
| **Predication** | None | `@%p0` predicate parsing and translation | NEW feature |
| **Comment Handling** | `#` only | `#`, `//`, inline `;` comment handling (FIXED) | Improved robustness |
| **Console** | Basic step/run/reg | `ptx` command showing PTX source + ISA translation | NEW command |
| **IBuffer.peek()** | Bug: peek returns last entry | Fixed: selects smallest PC entry | FIXED |
| **Memory Format** | N/A | `.ptx` files with bilingual Chinese+English comments | NEW |
| **Test Programs** | `.asm` only | `01_vector_add.ptx`, `02_scale.ptx`, `03_mov_imm.ptx` | NEW PTX tests |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **Limited PTX subset**: Only supports `mov.u32`, `add.u32`, `mul.lo.u32`, `ld.global.u32`, `st.global.u32`, `setp.ne.u32`, `bra`, `ret`. No float, 64-bit, or atomics support.
2. **Register allocation is basic**: Linear-scan with simple `vreg_id + 1` mapping. No spilling or live-range analysis.
3. **Predicated branches**: PTX `@%p0 bra` predication uses `BNE` (assumes `SUB` result non-zero → true). This is a simplification that may not handle all PTX predication semantics.
4. **Memory addressing**: `ld.global.u32` with register-based address is simplified to base-offset only. No full scatter/gather support.
5. **PTX type system**: Only `.u32` is supported. No `.f32`, `.s32`, `.f64`, or vector types.
6. **No PTX assembler optimizations**: The translator is a direct 1:1 mapping. No instruction scheduling or register renaming.
7. **Virtual register count tracking**: `num_regs` tracks max register number seen, but doesn't account for non-contiguous register usage patterns.
