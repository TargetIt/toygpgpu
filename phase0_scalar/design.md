# Phase 0: Scalar Processor — Design Document / 设计文档

> **对应 GPGPU-Sim**: gpgpu_sim (顶层) + shader_core_ctx (流水线) + simd_function_unit 的 SP (Streaming Processor) 单管线
> **参考**: RISC-V RV32I 指令格式 (riscv.org), TinyGPU (github.com/deaneeth/tinygpu), GPGPU-Sim shader.cc

## 1. Introduction / 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU (cpu.py)                              │
│                                                             │
│  ┌──────────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ RegisterFile │    │   ALU    │    │     Memory       │  │
│  │ (reg.py)     │    │ (alu.py) │    │ (memory.py)      │  │
│  │ 16 x 32-bit  │◄──►│ + - * /  │    │ 256 words x 32b  │  │
│  │ r0 hardwired │    │ s32 int  │    │ (bytearray 1KB)  │  │
│  │   to 0       │    └────┬─────┘    └────────┬─────────┘  │
│  └──────────────┘         │                    │           │
│                           │                    │           │
│  ┌─────────────────────────────────────────┐   │           │
│  │          ISA / Decoder (isa.py)          │   │           │
│  │  Instruction Fetch → Decode → Execute   │───┘           │
│  │  (single-cycle, no pipeline overlap)    │               │
│  └──────────────────┬──────────────────────┘               │
│                     │                                      │
│  ┌──────────────────▼──────────────────────┐               │
│  │       Assembler (assembler.py)           │               │
│  │    Text assembly → 32-bit machine code  │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ learning_console │  │    --trace mode  │                │
│  │ .py (interactive)│  │ (auto step+diff) │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

Phase 0 is the foundation of the entire toygpgpu project. It implements a minimal scalar processor with a custom 32-bit RISC-style ISA, a single ALU, 16 general-purpose registers (r0 hardwired to zero), and a flat memory model. The CPU executes a fetch-decode-execute loop in a single cycle per instruction, with no pipelining, no branch prediction, and no parallelism. This provides the baseline execution model onto which all subsequent GPU features (SIMD, SIMT, branch divergence, etc.) are layered.

Phase 0 是整个 toygpgpu 项目的基础。它实现了最简标量处理器，包含自定义 32 位 RISC 风格 ISA、单 ALU、16 个通用寄存器（r0 硬连线为 0）和平坦内存模型。CPU 以单周期每指令执行取指-译码-执行循环，无流水线重叠、无分支预测、无并行性。这为后续所有 GPU 特性（SIMD、SIMT、分支发散等）提供了基准执行模型。

## 2. Motivation / 设计动机

Before we can build a GPU simulator, we need a working CPU simulator. Real GPUs execute thousands of threads, but each individual thread is still a scalar processor running sequential instructions. The Streaming Processor (SP) in GPGPU-Sim is exactly this: each of the 32 SPs in a warp executes scalar integer/floating-point operations.

在设计 GPU 模拟器之前，需要一个可工作的 CPU 模拟器。真实 GPU 执行数千个线程，但每个单独线程仍然是执行顺序指令的标量处理器。GPGPU-Sim 中的 SP（Streaming Processor）正是如此：warp 中 32 个 SP 各自执行标量整数/浮点运算。

**What Phase 0 provides:**
- A complete, testable instruction set with assembler
- Register file with r0=0 convention (like RISC-V x0)
- Memory load/store operations
- Single-cycle execution model for easy debugging
- Interactive learning console for step-by-step inspection

**GPGPU-Sim context**: In GPGPU-Sim, `shader_core_ctx::cycle()` implements a 6-stage pipeline (fetch/decode/issue/read_operand/execute/writeback). Phase 0 collapses all these into a single `step()` call. The `simd_function_unit` class in GPGPU-Sim handles SP/SFU/INT execution; Phase 0 only implements the INT (integer) portion.

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Core Execution Loop / 核心执行循环

```
while True:
    raw_word = program[pc]     # Fetch (instruction memory read)
    pc += 1                    # Advance PC (before execute, like classic RISC)
    instr = decode(raw_word)   # Decode (extract opcode, rd, rs1, rs2, imm)
    execute(instr)             # Execute (read operands, compute, writeback)
    if instr.opcode == HALT:
        break
```

The PC advances *before* execution, following the classic RISC convention where the PC points to the *next* instruction during execution. This simplifies branch offset calculations in later phases.

### 3.2 Instruction Encoding / 指令编码

Each instruction is a fixed 32-bit word:

```
 31    24 23   20 19   16 15   12 11         0
+--------+-------+-------+-------+------------+
| opcode |  rd   |  rs1  |  rs2  | imm/addr   |
|  8-bit | 4-bit | 4-bit | 4-bit |   12-bit   |
+--------+-------+-------+-------+------------+
```

- **R-type** (ADD/SUB/MUL/DIV): rd = rs1 op rs2
- **I-type** (MOV/LD): rd = immediate or mem[imm]
- **S-type** (ST): mem[imm] = rs1
- **HALT**: opcode only, other fields ignored

The 12-bit immediate is sign-extended to 32 bits in the decode stage.

### 3.3 r0 Hardwired to Zero / r0 硬连线为 0

Following the RISC-V x0 convention, register r0 is permanently tied to ground (value 0). Reads always return 0; writes are silently ignored. This provides a convenient zero source for comparisons and constant generation.

### 3.4 Trace Mode Algorithm / 追踪模式算法

The CPU's `run(trace=True)` method uses a diff-based tracing mechanism:

```
pre_step:  snapshot all non-zero registers and memory
step:      execute one instruction
post_step: compare current state with pre-step state
           report only changed values for each register/memory location
```

This produces compact, readable trace output showing only what changed each cycle, rather than dumping entire state.

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Lines | Purpose |
|------|-------|---------|
| `isa.py` | 113 | Opcode definitions, Instruction dataclass, decode/encode functions |
| `register_file.py` | 72 | 16x32-bit register file with r0=0 |
| `alu.py` | 62 | 32-bit signed integer arithmetic (ADD/SUB/MUL/DIV) |
| `memory.py` | 75 | Flat memory model, 256 words x 32-bit, byte-level storage |
| `cpu.py` | 220 | Top-level CPU: fetch-decode-execute loop, trace mode |
| `assembler.py` | 155 | Text-to-machine-code assembler |
| `learning_console.py` | 282 | Interactive debug console with step/run/info commands |

### 4.2 Key Data Structures / 关键数据结构

**RegisterFile** (register_file.py):
```
regs: list[int]  # 16 entries, regs[0] always 0
  read(reg_id)   → 32-bit value (r0 returns 0)
  write(reg_id, value)  → silently drops writes to r0
```

**ALU** (alu.py):
```
add(a, b)  → (a + b) & 0xFFFFFFFF   # 32-bit truncation
sub(a, b)  → (a - b) & 0xFFFFFFFF
mul(a, b)  → (a * b) & 0xFFFFFFFF   # low 32 bits of product
div(a, b)  → signed division, 0 on div-by-zero
```

**Memory** (memory.py):
```
data: bytearray(1024)  # 256 words x 4 bytes, little-endian
read_word(addr)  → 32-bit value from word address 0..255
write_word(addr, value)  → store 32-bit value, truncated
```

**Instruction** (isa.py):
```
opcode: int    # 8-bit operation code
rd: int        # 4-bit destination register
rs1: int       # 4-bit source register 1
rs2: int       # 4-bit source register 2
imm: int       # 12-bit, sign-extended to 32-bit
raw: int       # original 32-bit encoding
```

### 4.3 ISA Encoding / 指令编码

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x00 | HALT | - | Stop execution |
| 0x01 | ADD | R | rd = rs1 + rs2 |
| 0x02 | SUB | R | rd = rs1 - rs2 |
| 0x03 | MUL | R | rd = rs1 * rs2 (low 32) |
| 0x04 | DIV | R | rd = rs1 / rs2 (0 if rs2=0) |
| 0x05 | LD | I | rd = mem[imm] |
| 0x06 | ST | S | mem[imm] = rs1 |
| 0x07 | MOV | I | rd = sign_ext(imm) |

### 4.4 Module Interfaces / 模块接口

```python
class CPU:
    def __init__(self, memory_size: int = 256)
    def load_program(self, program: list[int])
    def step(self) -> bool       # Execute one instruction, return False on HALT
    def run(self, trace: bool = False)
    def dump_state(self) -> str

def assemble(source: str) -> list[int]   # Standalone assembler function
def decode(word: int) -> Instruction     # Standalone decoder function
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Execution Timeline / 执行时间线

Step-by-step execution of `01_basic_arith.asm`:

```asm
; Compute 5 + 3, store result to memory[100]
MOV r1, 5        ; r1 = 5
MOV r2, 3        ; r2 = 3
ADD r3, r1, r2   ; r3 = r1 + r2 = 8
ST r3, [100]     ; mem[100] = r3
HALT             ; stop
```

Trace output (via `python cpu.py --trace 01_basic_arith.asm`):

```
[Cycle 0] PC=0: MOV rd=r1 rs1=r0 rs2=r0 imm=5 | reg: r1:0->5
[Cycle 1] PC=1: MOV rd=r2 rs1=r0 rs2=r0 imm=3 | reg: r2:0->3
[Cycle 2] PC=2: ADD rd=r3 rs1=r1 rs2=r2 imm=0 | reg: r3:0->8
[Cycle 3] PC=3: ST rd=r0 rs1=r3 rs2=r0 imm=100 | mem: mem[100]:0->8
[Cycle 4] PC=4: HALT rd=r0 rs1=r0 rs2=r0 imm=0 | reg: none
[Summary] 5 cycles, 5 instructions
```

### 5.2 Learning Console Session / 学习控制台会话

```
> python learning_console.py 01_basic_arith.asm
+----------------------------------------------------------+
|     toygpgpu Learning Console -- Phase 0 (Scalar)         |
+----------------------------------------------------------+
|  Program: 5 instructions                                  |
|  Commands: Enter=step, r=run, i=info, q=quit              |
+----------------------------------------------------------+

--- Program ---
  PC  0: MOV    rd=r1 rs1=r0 rs2=r0 imm=5
  PC  1: MOV    rd=r2 rs1=r0 rs2=r0 imm=3
  PC  2: ADD    rd=r3 rs1=r1 rs2=r0 imm=0
  PC  3: ST     rd=r0 rs1=r3 rs2=r0 imm=100
  PC  4: HALT   rd=r0 rs1=r0 rs2=r0 imm=0

[0] > s
Cycle 0: PC=0 MOV rd=r1 rs1=r0 rs2=r0 imm=5 | r1:0->5
[1] > s
Cycle 1: PC=1 MOV rd=r2 rs1=r0 rs2=r0 imm=3 | r2:0->3
[2] > r
Cycle 2: PC=2 ADD rd=r3 rs1=r1 rs2=r0 imm=0 | r3:0->8
Cycle 3: PC=3 ST rd=r0 rs1=r3 rs2=r0 imm=100 | mem mem[100]:0->8
Cycle 4: PC=4 HALT rd=r0 rs1=r0 rs2=r0 imm=0 | no reg change

  [HALT] CPU halted at cycle 5
--- Final State ---
Cycles executed: 5
Instructions executed: 5
Registers:
  r0=0, r1=5, r2=3, r3=8
Memory (non-zero):
  mem[100]=8
```

## 6. Comparison with Previous Phase / 与前一版本的对比

N/A — Phase 0 is the first phase in the toygpgpu project. There is no previous version to compare against.

Phase 0 是 toygpgpu 项目的第一个阶段，没有之前的版本可供比较。

## 7. Known Issues and Future Work / 遗留问题与后续工作

**Known Limitations / 已知限制:**
- Single-threaded: only one implicit thread, no multi-threading support
- No pipelining: each instruction completes in one cycle, no overlap
- Flat memory hierarchy: no caches, no shared memory, no global/local distinction
- No SIMD: all operations are scalar 32-bit
- No branches: sequential execution only (labels parsed but not used)
- No debugger integration: limited to console output
- No instruction-level parallelism: strict in-order execution

**Future Work / 后续工作:**
- **Phase 1: SIMD** — Add vector registers and vector ALU for 8-lane SIMD execution
- **Phase 2: SIMT** — Add multi-warp support with per-thread registers and warp scheduling
- **Phase 3: SIMT Stack** — Add branch divergence handling via SIMT stack
- **Phase 4+**: Scoreboard, memory hierarchy, kernel launch, pipeline stages

**Open Questions / 开放问题:**
- Should the ISA be extended to support unsigned operations?
- Should we add a status register (flags) for condition codes?
- The 12-bit immediate range (-2048 to 2047) is restrictive for large programs

**TODOs:**
- [ ] Add support for register-indirect addressing (LD/ST with register offset)
- [ ] Consider adding a program listing command in learning_console.py
- [ ] Add unit tests for all opcodes
