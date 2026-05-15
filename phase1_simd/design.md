# Phase 1: SIMD Vector Processor — Design Document / 设计文档

> **对应 GPGPU-Sim**: simd_function_unit (多 lane 并行执行), shader_core_ctx 的 SIMD 执行模型
> **参考**: RISC-V V 扩展向量指令集, GPGPU-Sim shader.cc 中 simd_function_unit, TinyGPU 向量单元

## 1. Introduction / 架构概览

```
Phase 0 (Scalar Only)              Phase 1 (SIMD Vector Extension)
┌─────────────────────┐            ┌──────────────────────────────────┐
│  RegisterFile        │            │  Scalar RegFile    Vector RegFile│
│  16 x 32-bit         │            │  r0-r15 (16)       v0-v7 (8)    │
│  (single thread)     │            │  每个 32-bit       每个 VLEN x   │
└──────────┬──────────┘            │                    32-bit       │
           │                       └──────┬───────────┬──────────────┘
           ▼                              │           │
┌─────────────────────┐            ┌──────▼────┐ ┌────▼──────────┐
│  ALU (1 lane)       │            │  Scalar   │ │  Vector ALU   │
│  + - * /            │            │   ALU     │ │  VLEN lanes   │
└─────────────────────┘            │  (same)   │ │  (8 lanes)    │
                                   └───────────┘ └───────────────┘
                                   ┌───────────────────────────────┐
                                   │   Vec4 ALU (SWAR)             │
                                   │   4 x 8-bit SIMD in 32-bit    │
                                   │   V4PACK/V4ADD/V4MUL/V4UNPACK │
                                   └───────────────────────────────┘
                                   ┌──────────────────────────────────┐
                                   │ learning_console.py (vreg cmd)    │
                                   │ trace mode (vector register diff) │
                                   └──────────────────────────────────┘
```

Phase 1 extends the Phase 0 scalar CPU with SIMD (Single Instruction, Multiple Data) vector processing capability. It adds a vector register file (8 registers, each VLEN=8 lanes wide) and a vector ALU that operates on all lanes simultaneously. This demonstrates the fundamental data-parallel execution model used in GPUs: one instruction operates on multiple data elements at once.

Phase 1 在 Phase 0 标量 CPU 基础上扩展了 SIMD（单指令多数据）向量处理能力。它增加了向量寄存器堆（8 个寄存器，每个 VLEN=8 个 lane 宽）和并行操作所有 lane 的向量 ALU。这演示了 GPU 中使用的基本数据并行执行模型：一条指令同时操作多个数据元素。

Additionally, Phase 1 introduces the **Vec4 ALU** — a SWAR (SIMD Within A Register) unit that packs 4 x 8-bit values into a single 32-bit register and performs byte-level SIMD operations. This mimics GPU shader packed types like `float4` or `uchar4`.

## 2. Motivation / 设计动机

Real GPUs achieve massive throughput through SIMD parallelism. In GPGPU-Sim, each warp executes instructions in SIMD fashion: 32 threads share a PC and execute the same instruction on their own registers. Phase 1 captures this by making the "thread" concept explicit as a "lane" of a vector register.

真实 GPU 通过 SIMD 并行实现巨大的吞吐量。在 GPGPU-Sim 中，每个 warp 以 SIMD 方式执行指令：32 个线程共享 PC 并在各自的寄存器上执行同一条指令。Phase 1 通过将"线程"概念显式表示为向量寄存器的"lane"来捕捉这一点。

**What Phase 1 enables:**
- Data-parallel operations on 8-element vectors with a single instruction
- Mixed scalar-vector computation (scalar for addresses, vectors for data)
- Vec4 packed SIMD for understanding SWAR techniques used in shader code
- Comparison between scalar and vector execution models

**GPGPU-Sim context**: The `simd_function_unit` in GPGPU-Sim contains multiple function units (SP for single-precision, SFU for special functions, DP for double-precision, INT for integer). Each function unit processes all active threads of a warp in parallel. Phase 1's VectorALU is a simplified version of this: one function unit operating on VLEN lanes.

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 SIMD Lane Operation / SIMD Lane 操作

Each vector instruction operates on VLEN lanes independently:

```
VADD vd, vs1, vs2:
  for i in 0..VLEN-1:
    vd[i] = vs1[i] + vs2[i]    # Each lane is independent
```

All lanes execute the same operation in lockstep. This is the defining characteristic of SIMD — and also of GPU warp execution.

### 3.2 Vector Load/Store / 向量加载/存储

VLD loads VLEN consecutive words from memory into a vector register. VST stores them back:

```
VLD v1, [base_addr]:
  for i in 0..VLEN-1:
    v1[i] = mem[base_addr + i]    # Strided contiguous load

VST v1, [base_addr]:
  for i in 0..VLEN-1:
    mem[base_addr + i] = v1[i]    # Strided contiguous store
```

This is analogous to GPU coalesced memory access where consecutive threads access consecutive addresses.

### 3.3 Broadcast / 广播

VMOV broadcasts a scalar immediate to all lanes of a vector register:

```
VMOV v1, imm:
  for i in 0..VLEN-1:
    v1[i] = sign_ext(imm)       # Same value to all lanes
```

### 3.4 Vec4 SWAR Algorithm / Vec4 SWAR 算法

The Vec4 ALU packs 4 x 8-bit values into one 32-bit register and performs operations "across" the bytes using masking and shifting — no separate vector hardware needed:

```
Pack:  byte0 = a[7:0], byte1 = a[15:8], byte2 = b[7:0], byte3 = b[15:8]
Add:   for each byte lane i: result[i] = (a[i] + b[i]) & 0xFF
Mul:   for each byte lane i: result[i] = (a[i] * b[i]) & 0xFF
Unpack: extract lane i from packed word, zero-extend to 32-bit
```

This is a classic SWAR technique: exploiting the existing 32-bit datapath to do 4-way 8-bit SIMD without adding hardware lanes.

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Lines | Purpose |
|------|-------|---------|
| `isa.py` | 139 | Adds vector opcodes (0x11-0x17) and Vec4 opcodes (0x25-0x28) |
| `register_file.py` | 72 | Scalar 16x32-bit register file (unchanged from Phase 0) |
| `alu.py` | 62 | Scalar ALU (unchanged from Phase 0) |
| `memory.py` | 75 | Expanded memory (1024 words from 256 words) |
| `cpu.py` | 313 | CPU extended with vector execution paths, trace mode |
| `assembler.py` | 217 | Extended with vector and Vec4 instruction parsing |
| **`vector_register_file.py`** | 79 | **NEW**: 8 x VLEN x 32-bit vector register file |
| **`vector_alu.py`** | 43 | **NEW**: VLEN-lane parallel ALU |
| **`vec4_alu.py`** | 66 | **NEW**: 4x8-bit SWAR ALU (V4PACK/V4ADD/V4MUL/V4UNPACK) |
| **`learning_console.py`** | 365 | **NEW**: Interactive console with `vreg` command, vector trace |

**Total: ~1200 lines** (vs ~900 in Phase 0)

### 4.2 Key Data Structures / 关键数据结构

**VectorRegisterFile** (vector_register_file.py):
```
vlen: int          # Number of lanes (default 8)
num_regs: int      # Number of vector registers (8: v0-v7)
regs: list[list[int]]  # regs[reg_id][lane] -> 32-bit value

read(reg_id)          -> list[int]      # Returns all VLEN lane values
read_lane(reg_id, lane) -> int          # Returns single lane
write(reg_id, values)                   # Write all lanes at once
write_lane(reg_id, lane, value)          # Write single lane
broadcast(reg_id, value)                # Write same value to all lanes
```

**VectorALU** (vector_alu.py):
```
vlen: int
vadd(a: list[int], b: list[int]) -> list[int]   # Per-lane add
vsub(a: list[int], b: list[int]) -> list[int]   # Per-lane subtract
vmul(a: list[int], b: list[int]) -> list[int]   # Per-lane multiply
vdiv(a: list[int], b: list[int]) -> list[int]   # Per-lane divide (0 on div-by-0)
```

Each vector ALU method iterates over lanes and delegates to the scalar `ALU` static methods.

**Vec4ALU** (vec4_alu.py):
```
pack(a: int, b: int) -> int     # Pack 4 bytes from a,b into 32-bit
add(a: int, b: int) -> int      # 4 x 8-bit SIMD add
mul(a: int, b: int) -> int      # 4 x 8-bit SIMD mul
unpack(word: int, lane: int) -> int  # Extract byte lane
```

### 4.3 ISA Encoding / 指令编码

**New Vector Opcodes (Phase 1):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x11 | VADD | VR | vd[i] = vs1[i] + vs2[i] |
| 0x12 | VSUB | VR | vd[i] = vs1[i] - vs2[i] |
| 0x13 | VMUL | VR | vd[i] = vs1[i] * vs2[i] |
| 0x14 | VDIV | VR | vd[i] = vs1[i] / vs2[i] |
| 0x15 | VLD | VI | vd[i] = mem[imm + i] |
| 0x16 | VST | VS | mem[imm + i] = vs[i] |
| 0x17 | VMOV | VI | vd[i] = sign_ext(imm) (broadcast) |

**New Vec4 Opcodes (Phase 1+):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x25 | V4PACK | R | rd = pack(rs1[7:0], rs1[15:8], rs2[7:0], rs2[15:8]) |
| 0x26 | V4ADD | R | rd[i] = rs1[i] + rs2[i] for i in 0..3 (8-bit SIMD) |
| 0x27 | V4MUL | R | rd[i] = rs1[i] * rs2[i] for i in 0..3 (8-bit SIMD) |
| 0x28 | V4UNPACK | R | rd = (rs1 >> (rs2*8)) & 0xFF, extract byte lane |

All vector opcodes reuse the same 32-bit instruction format. The `rd`/`rs1`/`rs2` fields refer to vector registers (v0-v7) when the opcode is a vector instruction, and scalar registers (r0-r15) otherwise. The assembler and execution logic handle this multiplexing.

### 4.4 Module Interfaces / 模块接口

```python
class CPU:  # Extended from Phase 0
    def __init__(self, vlen: int = 8, memory_size: int = 1024)
    # ... same methods as Phase 0, plus:
    # _execute() extended to handle vector and Vec4 opcodes
    # run(trace=True) shows both scalar and vector register diffs

class VectorALU:
    def __init__(self, vlen: int = 8)
    def vadd(self, a: List[int], b: List[int]) -> List[int]
    def vsub(self, a: List[int], b: List[int]) -> List[int]
    def vmul(self, a: List[int], b: List[int]) -> List[int]
    def vdiv(self, a: List[int], b: List[int]) -> List[int]

class Vec4ALU:
    @staticmethod def pack(a: int, b: int) -> int
    @staticmethod def add(a: int, b: int) -> int
    @staticmethod def mul(a: int, b: int) -> int
    @staticmethod def unpack(word: int, lane: int) -> int
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Vector Add Demo / 向量加法演示

Running `01_vector_add.asm` with `--trace`:

```asm
; C[i] = A[i] + B[i], i = 0..7
; A at mem[0..7], B at mem[8..15], C at mem[16..23]
VLD  v1, [0]       ; v1 = mem[0..7]   (A)
VLD  v2, [8]       ; v2 = mem[8..15]  (B)
VADD v3, v1, v2    ; v3 = A + B
VST  v3, [16]      ; mem[16..23] = C
HALT
```

Trace output (4 instructions for 8 pairs of data):

```
[Cycle 0] PC=0: VLD vd=v1 vs1=v0 vs2=v0 imm=0 | vreg: v1:[0,0,0,0,0,0,0,0]->[1,2,3,4,5,6,7,8]
[Cycle 1] PC=1: VLD vd=v2 vs1=v0 vs2=v0 imm=8 | vreg: v2:[0,0,0,0,0,0,0,0]->[10,20,30,40,50,60,70,80]
[Cycle 2] PC=2: VADD vd=v3 vs1=v1 vs2=v2 imm=0 | vreg: v3:[0,0,0,0,0,0,0,0]->[11,22,33,44,55,66,77,88]
[Cycle 3] PC=3: VST vd=v0 vs1=v3 vs2=v0 imm=16 | mem: mem[16]=11, mem[17]=22, mem[18]=33, mem[19]=44
[Cycle 4] PC=4: HALT
[Summary] 5 cycles, 5 instructions
```

Note how `VADD v3, v1, v2` performs 8 additions in a single instruction — this is the power of SIMD.

### 5.2 Vec4 Demo / Vec4 演示

Running `06_vec4_demo.asm` demonstrates packing two `vec4` values, adding and multiplying them component-wise, then unpacking individual lanes:

```asm
; v1 = (3, 5, 7, 9), v2 = (1, 2, 3, 4)
V4PACK r7, r5, r6    ; r7 = packed(9,7,5,3) = 0x09070503
V4PACK r12, r8, r9   ; r12 = packed(4,3,2,1) = 0x04030201
V4ADD r13, r7, r12   ; r13 = (4,7,10,13) = 0x0D0A0704
V4MUL r14, r7, r12   ; r14 = (3,10,21,36) = 0x24150A03
V4UNPACK r1, r13, 0  ; r1 = lane 0 = 4
```

This demonstrates how GPU shaders handle packed data types without dedicated vector units.

### 5.3 Learning Console Session / 学习控制台会话

```
> python learning_console.py 01_vector_add.asm
+----------------------------------------------------------+
|     toygpgpu Learning Console -- Phase 1 (SIMD Vector)     |
+----------------------------------------------------------+
|  Program: 5 instructions                                   |
|  VLEN:    8 lanes                                          |
|  Commands: Enter=step, r=run, i=info, q=quit              |
+----------------------------------------------------------+

[0] > vreg    (show vector registers before any execution)
v0: [0, 0, 0, 0, 0, 0, 0, 0]
v1: [0, 0, 0, 0, 0, 0, 0, 0]
...
[0] > s
Cycle 0: PC=0 VLD vd=v1 vs1=v0 vs2=v0 imm=0 | v1[0]:0->1 v1[1]:0->2 ...
[1] > s
Cycle 1: PC=1 VLD vd=v2 vs1=v0 vs2=v0 imm=8 | v2[0]:0->10 v2[1]:0->20 ...
[2] > i
--- Current State ---
PC=3, RUNNING, Instructions=3, VLEN=8
  Scalar registers: r0=0
  Vector registers:
    v1: [1, 2, 3, 4, 5, 6, 7, 8]
    v2: [10, 20, 30, 40, 50, 60, 70, 80]
  Memory: (all zero)
[2] > r
...
  [HALT] CPU halted at cycle 5
```

## 6. Comparison with Phase 0 / 与 Phase 0 的对比

| Aspect | Phase 0 (Scalar) | Phase 1 (SIMD) | Change |
|--------|------------------|----------------|--------|
| Files | 7 source files | 10 source files | +3: vector_register_file.py, vector_alu.py, vec4_alu.py |
| Opcodes | 8 (0x00-0x07) | 19 (0x00-0x07, 0x11-0x17, 0x25-0x28) | +11: 7 vector + 4 Vec4 |
| Key Feature | Single scalar thread | 8-lane SIMD + Vec4 SWAR | Vector parallelism |
| Register Types | Scalar only (r0-r15) | Scalar (r0-r15) + Vector (v0-v7) | Dual register files |
| Memory | 256 words | 1024 words | 4x larger |
| Execution Unit | ALU (1 lane) | ALU + VectorALU + Vec4ALU | 3 execution units |
| Console | Basic scalar debug | + vreg command, vector trace | Enhanced debugging |
| Max Throughput | 1 op/cycle | 8 ops/cycle (vector) + scalar | 8x peak throughput |

**What changed:**
- `cpu.py`: extended `_execute()` with vector and Vec4 instruction handling
- `isa.py`: added vector opcodes (0x11-0x17) and Vec4 opcodes (0x25-0x28)
- `assembler.py`: added `_parse_vreg()`, vector and Vec4 instruction parsing
- `memory.py`: expanded from 256 to 1024 words
- `learning_console.py`: new file with `vreg` command, vector register diff in trace

**What stayed the same:**
- Scalar ISA (Phase 0 programs run unchanged)
- RegisterFile and ALU classes
- Core fetch-decode-execute loop
- Instruction encoding format (32-bit fixed-length)

## 7. Known Issues and Future Work / 遗留问题与后续工作

**Known Limitations / 已知限制:**
- Fixed VLEN=8: not configurable at runtime, cannot change lane count per program
- No predicated vector operations: all lanes always execute (no active mask)
- No gather/scatter: VLD/VST only support contiguous strided access
- Vec4 operations are SWAR-based, not true hardware SIMD (but functionally equivalent)
- No mixed-width vector operations (e.g., promoting 8-bit to 32-bit)
- Vector register file uses Python list-of-lists — not cycle-accurate for hardware modeling

**What Phase 2 will add:**
- Multi-warp SIMT execution: Thread and Warp classes, WarpScheduler, SIMTCore
- Per-thread scalar register files (replaces vector registers with per-lane scalar regs)
- TID/WID instructions for thread identification
- BAR (barrier synchronization) instruction
- Round-robin warp scheduling

**What Phase 3 will add:**
- Branch divergence handling via SIMT stack (JMP/BEQ/BNE)
- Predication (SETP/@p0)
- Warp-level uniform registers (WREAD/WWRITE)

**Open Questions / 开放问题:**
- Should Vec4 operations be kept in Phase 2+ SIMT or removed?
  - Currently kept: Vec4 operates on scalar registers which per-thread regs support
- Vector operations are NOT carried forward to Phase 2+ (SIMT uses per-thread scalar regs instead). Is this the right design choice?
  - Yes: GPUs use per-thread scalar registers, not vector registers, for SIMT execution

**TODOs:**
- [ ] Add test for VLD/VST with address ranges that cross VLEN boundaries
- [ ] Consider adding VSHUFFLE (permute lanes) for completeness
- [ ] Add Vec4 test cases for saturation arithmetic (clamp to 0-255)
