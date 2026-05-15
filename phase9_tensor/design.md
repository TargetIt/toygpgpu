# Phase 9: Tensor Core MMA / 张量核心矩阵乘加 — Design Document / 设计文档

> **对应 GPGPU-Sim**: `gpgpu-sim/tensor_core.h`, `gpgpu-sim/tensor_core.cc`
> **参考**: NVIDIA Volta/V100 Tensor Core Programming Guide, NVIDIA MMA (Matrix Multiply-Accumulate) PTX Instruction

## 1. Introduction / 架构概览

```
            ┌────────────────────────────────────────────────────────┐
            │           Tensor Core MMA Data Flow                    │
            │                                                       │
            │      r1 = [a1:16][a0:16]     r2 = [b1:16][b0:16]     │
            │          └────────┬────────┘     └───────┬────────┘   │
            │                   │                      │            │
            │                   ▼                      ▼            │
            │              ┌──────────────────────────────────┐      │
            │              │          Tensor Core MMA          │     │
            │              │  r4 = a0*b0 + a1*b1 + r3          │     │
            │              │        (signed 16-bit)            │     │
            │              └──────────────┬───────────────────┘      │
            │                             │                         │
            │                             ▼                         │
            │                          r4 = result                  │
            └────────────────────────────────────────────────────────┘
                                                                      
            ┌────────────────────────────────────────────────────────┐
            │           Matrix Multiplication Scheme                  │
            │                                                       │
            │  D = A × B + C  (2×2 matrices)                        │
            │                                                       │
            │  D[0][0] = A_row0 · B_col0 + C                       │
            │  D[0][1] = A_row0 · B_col1 + C                       │
            │  D[1][0] = A_row1 · B_col0 + C                       │
            │  D[1][1] = A_row1 · B_col1 + C                       │
            │                                                       │
            │  4 MMA instructions complete a 2×2 matrix multiply    │
            └────────────────────────────────────────────────────────┘
```

Phase 9 在 toygpgpu 中引入了张量核心仿真。核心指令是 MMA (Matrix Multiply-Accumulate)，执行 2×2 元素点积并累加。16 位打包操作数存放在 32 位寄存器中，模拟了 NVIDIA Volta+ GPU 中张量核心的基本操作模式。

Phase 9 introduces tensor core simulation in toygpgpu. The core instruction is MMA (Matrix Multiply-Accumulate), which performs a 2×2 element dot product with accumulation. Packed 16-bit operands are stored in 32-bit registers, mimicking the basic operation mode of tensor cores in NVIDIA Volta+ GPUs.

## 2. Motivation / 设计动机

Modern GPUs (NVIDIA Volta V100, Turing, Ampere, Hopper) have dedicated **tensor cores** that perform matrix multiply-accumulate operations at significantly higher throughput than conventional ALUs. Tensor cores are the fundamental hardware unit driving AI/ML workloads.

现代 GPU（NVIDIA Volta V100、Turing、Ampere、Hopper）拥有专用的**张量核心**，能以远高于传统 ALU 的吞吐量执行矩阵乘加运算。张量核心是驱动 AI/ML 工作负载的基础硬件单元。

GPGPU-Sim has a tensor core model that simulates:
- 4×4×4 matrix multiply (FP16, INT8, etc.)
- Warp-level MMA instructions
- Accumulation support

Phase 9 implements a simplified tensor core:
- **1×1 element dot product** (2 packed 16-bit values per operand)
- **Accumulation** with a third source register
- **Signed 16-bit** unpack with sign extension
- **Full 2×2 matrix multiply** as a composition of 4 MMA instructions

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Data Packing Format / 数据打包格式

Each 32-bit register packs two 16-bit signed integers:

```
Bit:  31              16  15               0
      ┌──────────────────┬──────────────────┐
      │       a1         │       a0         │
      │    (high)        │     (low)        │
      └──────────────────┴──────────────────┘
        signed 16-bit      signed 16-bit
```

Memory pre-load stores the packed data as 32-bit words:
```
mem[0] = 0x00030002  →  a1=3, a0=2
mem[1] = 0x00050004  →  b1=5, b0=4
```

### 3.2 MMA Computation / MMA 计算

```python
def mma(a_packed: int, b_packed: int, c: int) -> int:
    """Compute packed 16-bit dot product with accumulation."""
    # Unpack a (signed 16-bit)
    a0 = a_packed & 0xFFFF
    if a0 & 0x8000: a0 -= 0x10000
    a1 = (a_packed >> 16) & 0xFFFF
    if a1 & 0x8000: a1 -= 0x10000

    # Unpack b (signed 16-bit)
    b0 = b_packed & 0xFFFF
    if b0 & 0x8000: b0 -= 0x10000
    b1 = (b_packed >> 16) & 0xFFFF
    if b1 & 0x8000: b1 -= 0x10000

    # Dot product + accumulate
    result = a0 * b0 + a1 * b1 + c
    return result & 0xFFFFFFFF
```

### 3.3 2×2 Matrix Multiply / 2×2 矩阵乘法

To compute D = A × B + C where all matrices are 2×2:

```
Composition: 4 MMA instructions

D[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + C
D[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + C
D[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + C
D[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + C

Data layout:
  A_row0 = [a01, a00]  (packed in r1)
  A_row1 = [a11, a10]  (packed in r2)
  B_col0 = [b10, b00]  (packed in r3)
  B_col1 = [b11, b01]  (packed in r4)
```

### 3.4 Instruction Encoding / 指令编码

MMA uses a 4-register format: `rd, rs1, rs2, rs3`

```python
def encode_mma(rd: int, rs1: int, rs2: int, rs3: int) -> int:
    """Encode MMA: rs3 stored in imm[11:8]"""
    w = 0
    w |= (OP_MMA & 0xFF) << 24    # [31:24] = 0x41
    w |= (rd & 0xF) << 20         # [23:20] = rd
    w |= (rs1 & 0xF) << 16        # [19:16] = rs1
    w |= (rs2 & 0xF) << 12        # [15:12] = rs2
    w |= (rs3 & 0xF) << 8         # [11:8]  = rs3
    return w

def decode_mma_rs3(word: int) -> int:
    """Extract rs3 from MMA instruction"""
    return (word >> 8) & 0xF
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Purpose / 用途 | Change from Phase 8 |
|------|---------------|---------------------|
| `isa.py` | Added `OP_MMA = 0x41`, `encode_mma()`, `decode_mma_rs3()` | ADDED MMA opcode + helpers |
| `simt_core.py` | Added `OP_MMA` execution: unpack, dot product, accumulate | ADDED tensor core execution |
| `assembler.py` | Added `MMA rd, rs1, rs2, rs3` assembly support | ADDED assembler rule |
| `learning_console.py` | Added `mma` command: show MMA info, packed register display | NEW command |
| `ptx_parser.py` | Unchanged from Phase 8 | Same |
| `alu.py` | Scalar ALU | Same |
| `vec4_alu.py` | Vec4 8-bit SIMD | Same |
| `warp.py` | Warp/Thread with PRED, warp_regs | Same |
| `ibuffer.py` | IBuffer with peek()+reconv fix | Same |
| `memory.py` | Global memory | Same |
| `cache.py` | L1 cache | Same |
| `scoreboard.py` | Scoreboard with pipeline latency | Same |
| `scheduler.py` | Warp scheduler | Same |
| `simt_stack.py` | SIMT divergence management | Same |
| `operand_collector.py` | Multi-bank register file | Same |
| `gpu_sim.py` | Top-level GPU simulator | Same |

### 4.2 Key Data Structures / 关键数据结构

```python
# In isa.py
OP_MMA = 0x41

# encode_mma(rd, rs1, rs2, rs3) → 32-bit instruction word
# Bit layout:
#   [31:24] = 0x41 (opcode)
#   [23:20] = rd  (destination)
#   [19:16] = rs1 (packed operand A)
#   [15:12] = rs2 (packed operand B)
#   [11:8]  = rs3 (accumulator)
#   [7:0]   = unused

# decode_mma_rs3(word) → int (rs3 register ID)
```

### 4.3 ISA Encoding / 指令编码

| Mnemonic | Fields | Encoding |
|----------|--------|----------|
| `MMA rd, rs1, rs2, rs3` | op=0x41, rd[23:20], rs1[19:16], rs2[15:12], rs3[11:8] | `encode_mma(rd, rs1, rs2, rs3)` |

### 4.4 Module Interfaces / 模块接口

```python
# isa.py
def encode_mma(rd: int, rs1: int, rs2: int, rs3: int) -> int
def decode_mma_rs3(word: int) -> int

# simt_core.py (inside _execute_warp)
if op == OP_MMA:
    rs3_id = decode_mma_rs3(instr.raw)
    for t in tlist:
        a_packed = t.read_reg(instr.rs1)
        b_packed = t.read_reg(instr.rs2)
        c_val = t.read_reg(rs3_id)
        # unpack signed 16-bit × 2, dot product, accumulate

# learning_console.py
def print_mma_info(simt):  # Show MMA instructions + packed register values
```

## 5. Functional Processing Flow / 功能处理流程

### Test 01: MMA Dot Product (`01_mma_dot.asm`)

```
Pre-load: mem[0] = 0x00030002 (a1=3, a0=2)
          mem[1] = 0x00050004 (b1=5, b0=4)

Execution:
  Cycle  LD  r1, [0]      r1 = 0x00030002 = [a1=3, a0=2]
  Cycle  LD  r2, [1]      r2 = 0x00050004 = [b1=5, b0=4]
  Cycle  MOV r3, 10       r3 = 10
  Cycle  MMA r4, r1, r2, r3
         ┌─ Unpack: a0=2, a1=3, b0=4, b1=5, c=10
         ├─ Compute: 2×4 + 3×5 + 10 = 8 + 15 + 10
         └─ Result: r4 = 33
  Cycle  ST  r4, [10]     mem[10] = 33
  Cycle  HALT

Result: mem[10] = 33 = 2*4 + 3*5 + 10
```

### Test 02: 2×2 Matrix Multiply (`02_matmul_2x2.asm`)

```
Matrices:
  A = [[2, 3],   B = [[4, 5],   C = 1
       [1, 4]]        [6, 7]]

Memory pre-load:
  mem[0] = packed [3, 2]   (A_row0: a01=3, a00=2)
  mem[1] = packed [4, 1]   (A_row1: a11=4, a10=1)
  mem[2] = packed [6, 4]   (B_col0: b10=6, b00=4)
  mem[3] = packed [7, 5]   (B_col1: b11=7, b01=5)

Execution:
  MMA r6, r1, r3, r5  → r6 = 2×4 + 3×6 + 1 = 27  → ST mem[10]
  MMA r7, r1, r4, r5  → r7 = 2×5 + 3×7 + 1 = 32  → ST mem[11]
  MMA r8, r2, r3, r5  → r8 = 1×4 + 4×6 + 1 = 29  → ST mem[12]
  MMA r9, r2, r4, r5  → r9 = 1×5 + 4×7 + 1 = 34  → ST mem[13]

Result: D = [[27, 32], [29, 34]]
```

### Test 03: Negative MMA (`03_negative_mma.asm`)

```
mem[0] = packed [7, 1]    (a1=7, a0=1)
mem[1] = packed [2, 3]    (b1=2, b0=3)
r3 = 10
MMA r4, r1, r2, r3
Result: 1×3 + 7×2 + 10 = 3 + 14 + 10 = 27 → mem[10]
```

## 6. Comparison with Phase 8 / 与 Phase 8 的对比

| Aspect / 方面 | Phase 8 | Phase 9 | Change / 变化 |
|---------------|---------|---------|---------------|
| **ALU Type** | Scalar ALU only (ADD, SUB, MUL, DIV) | + Tensor Core MMA (matrix multiply-accumulate) | NEW instruction class |
| **Opcode** | 0x00–0x32 | NEW: `OP_MMA = 0x41` | NEW opcode |
| **Register Operands** | 2 source + 1 dest (R-type) | 3 source + 1 dest (4-register MMA) | Extended encoding |
| **Data Width** | 32-bit full register | 16-bit packed × 2 within 32-bit register | Sub-word SIMD |
| **Computation** | Element-wise arithmetic | Dot product (multiply + add across elements) | New operation model |
| **Assembly Format** | `ADD rd, rs1, rs2` | `MMA rd, rs1, rs2, rs3` | 4-operand instruction |
| **Encoding Helper** | `encode_rtype`, `encode_itype`, `encode_stype` | NEW: `encode_mma`, `decode_mma_rs3` | New encoding family |
| **Console Display** | PTX source | `mma` command: packed register display | NEW command |
| **Programming Model** | Per-element operations | Matrix-level operations (composition) | Higher abstraction |
| **Limitation** | N/A | 12-bit MOV can't encode 32-bit packed values | Memory pre-load required |
| **Test Programs** | 3 PTX tests | 3 ASM tests (dot, matmul, negative) | Expanded coverage |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **Single MMA precision**: Only 16-bit signed integer. No FP16, INT8, INT4, BF16, or TF32 support.
2. **1×1 dot product only**: Real NVIDIA tensor cores support 4×4×4 matrix multiply (16 elements). Our MMA handles only 2 elements per operand.
3. **No warp-level MMA**: Real tensor core instructions are warp-wide (all 32 threads participate). Our MMA is per-thread scalar.
4. **Manual packing**: Packed data must be pre-loaded from memory. No automatic packing support.
5. **No fragmentation**: No support for tiled matrix multiplication or shared memory buffering of tiles.
6. **No saturation/clipping**: Real tensor cores handle INT8 saturation. Ours just masks to 32 bits.
7. **No pipelined tensor operations**: Real tensor cores pipeline matrix multiply across cycles. Ours completes in one cycle.
