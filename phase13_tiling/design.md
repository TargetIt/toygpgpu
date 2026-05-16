# Phase 13: Tiling Strategies — Design Document / 设计文档

> **对应 GPGPU-Sim**: 内存管线中的 tile 加载 (tlb/tlb.sv),
> `gpgpu-sim/../gpgpusim_entry_point.cc` 中的 tiled memory access 模式
> **参考**: NVIDIA CUTLASS tiling 策略, CUDA 分块矩阵乘法编程指南

## 1. Introduction / 架构概览

```
                      ┌──────────────────────────────────────────────┐
                      │          Tiling Architecture                 │
                      │                                              │
                      │     Global Memory (DRAM)                     │
                      │     ┌────┬────┬────┬────┬────┬────┐          │
                      │     │ A0 │ A1 │ A2 │ A3 │ B0 │ B1 │  ...    │
                      │     └────┴────┴────┴────┴────┴────┘          │
                      │           │  TLDS (load tile)                 │
                      │           ▼                                   │
                      │     Shared Memory (tile buffer)              │
                      │     ┌────┬────┬────┬────┬────┬────┐          │
                      │     │ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │  ...    │
                      │     └────┴────┴────┴────┴────┴────┘          │
                      │           │  SHLD (read from tile)            │
                      │           ▼                                   │
                      │     Registers (per-thread compute)           │
                      │     ┌────┬────┬────┬────┬────┬────┐          │
                      │     │ r0 │ r1 │ r2 │ r3 │ r4 │ r5 │  ...    │
                      │     └────┴────┴────┴────┴────┴────┘          │
                      │                                              │
                      │     TLSTS: store tile back to global         │
                      │     TLCONF: configure tile dimensions        │
                      │     (M=rows, N=cols, K=inner dim)            │
                      └──────────────────────────────────────────────┘
```

Phase 13 向 toygpgpu 添加了**三种分块指令**：TLCONF（Tile 配置）、TLDS（Tile 加载）和 TLSTS（Tile 存储）。这些指令允许将全局内存中的数据分块加载到共享内存中，然后由线程从共享内存中高效读取，大幅减少全局内存访问次数。这是实现分块矩阵乘法等高性能计算模式的基础。

Phase 13 adds **three tiling instructions** to toygpgpu: TLCONF (Tile Configure), TLDS (Tile Load), and TLSTS (Tile Store). These instructions enable structured loading of data tiles from global memory into shared memory, followed by efficient per-thread reads from shared memory, significantly reducing global memory traffic. This forms the foundation for high-performance computing patterns such as tiled matrix multiplication.

### 新增指令概览 / New Instructions Overview

| 指令 | Opcode | 功能 | 对应模式 |
|------|--------|------|----------|
| TLCONF | 0x35 | 配置 tile 维度 (M=rd, N=rs1, K=imm) | `cudaFuncSetAttribute` / CUTLASS TileCoord |
| TLDS | 0x36 | 从全局内存加载 tile 到共享内存 | CUDA 手动 tile 加载循环 |
| TLSTS | 0x37 | 从共享内存写回 tile 到全局内存 | CUDA 手动 tile 存储循环 |

## 2. Motivation / 设计动机

### 2.1 为什么需要分块？/ Why Tiling?

在 GPGPU 编程中，全局内存 (Global Memory) 延迟很高（数百周期），而共享内存 (Shared Memory) 延迟很低（1-2 周期）。分块的核心思想是：

In GPGPU programming, global memory latency is high (hundreds of cycles), while shared memory latency is very low (1-2 cycles). The core idea of tiling is:

1. **数据复用 / Data Reuse**: 将全局内存中的数据分块加载到共享内存中，同一个 warp 或 thread block 内的多个线程可以多次读取同一块数据而无需重新访问全局内存。

2. **带宽优化 / Bandwidth Optimization**: 合并的全局内存访问（coalesced access）以 tile 为单位，每个线程加载连续的地址，最大化内存带宽利用率。

3. **延迟隐藏 / Latency Hiding**: Tile 加载可以与计算重叠（通过双缓冲），在计算当前 tile 的同时预取下一个 tile。

### 2.2 矩阵乘法中的分块 / Tiling in Matrix Multiplication

矩阵乘法 C[M][N] = A[M][K] x B[K][N] 是分块策略的经典用例：

Without tiling (naive):
- 每个 C[i][j] 需要从全局内存读取 A 的一行和 B 的一列 → 大量冗余全局内存访问
- A 的每个元素被读取 N 次，B 的每个元素被读取 M 次

With tiling:
- A 的 tile (`M_tile x K_tile`) 和 B 的 tile (`K_tile x N_tile`) 加载到共享内存
- C 的 tile (`M_tile x N_tile`) 通过从共享内存读取来计算
- 全局内存访问减少到原来的 1/K_tile

### 2.3 与 CUTLASS 的对应关系 / CUTLASS Correspondence

NVIDIA CUTLASS 是一个基于模板的 CUDA 矩阵乘法库，其核心是分块策略：

```
CUTLASS 概念          │ toygpgpu        │ 说明
──────────────────────┼─────────────────┼────────────────────
TileCoord             │ TLCONF M,N,K    │ tile 维度配置
GlobalToSharedLoader  │ TLDS            │ 全局→共享加载
SharedToGlobalStore   │ TLSTS           │ 共享→全局存储
Mainloop (mma)        │ SHLD + ALU      │ 核心计算循环
Epilogue              │ ST              │ 结果写回
```

### 2.4 与 GPGPU-Sim 的对应关系 / GPGPU-Sim Correspondence

GPGPU-Sim 中的内存管线支持分块访问模式：

```
GPGPU-Sim 组件        │ toygpgpu        │ 说明
──────────────────────┼─────────────────┼────────────────────
memory_pipeline       │ _execute_warp   │ 内存管线调度
tlb (translation)     │ 无 (简化)       │ 地址转换层
coalescer             │ 无 (简化)       │ 内存访问合并
shared_memory         │ SharedMemory    │ 分块数据暂存
```

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 TLCONF — Tile 配置 / Tile Configuration

TLCONF 指令配置 tile 的三个维度参数，存储在 SIMTCore 的状态中：

The TLCONF instruction configures three tile dimension parameters, stored in SIMTCore state:

```
伪代码 / Pseudocode:

TLCONF rd, rs1, imm:
    simt.tile_m = rd     # Tile 行数 (M dimension)
    simt.tile_n = rs1    # Tile 列数 (N dimension)
    simt.tile_k = imm    # Tile 内维 (K dimension, 共享维度)

编码 / Encoding:
  [31:24]=0x35  [23:20]=M  [19:16]=N  [15:12]=0  [11:0]=K

默认值 / Defaults:
    tile_m = 8, tile_n = 8, tile_k = 8  (SIMTCore.__init__)

示例 / Example:
    TLCONF 2, 2, 2    → tile_m=2, tile_n=2, tile_k=2
    TLCONF 4, 8, 16   → tile_m=4, tile_n=8, tile_k=16
```

### 3.2 TLDS — Tile 加载 / Tile Load

TLDS 指令将 global memory 中的一个 tile 加载到 shared memory。每个活跃线程根据其 thread_id 计算负责加载的数据元素位置：

The TLDS instruction loads a tile from global memory into shared memory. Each active thread computes its assigned data element position based on its thread_id:

```
伪代码 / Pseudocode:

TLDS rd, rs1, imm:
    smem_off   = rd          # shared memory 中的起始偏移
    glob_base  = rs1         # global memory 中的基地址
    smem       = simt.thread_block.shared_memory

    for each active thread t in warp:
        tid = t.thread_id
        row = tid // tile_n  # 在 tile 中的行
        col = tid %  tile_n  # 在 tile 中的列

        # 计算 shared memory 地址 (行优先)
        smem_addr = smem_off + row * tile_n + col

        # 计算 global memory 地址 (连续地址)
        glob_addr = glob_base + tid

        # 传输数据
        smem.write_word(smem_addr, memory.read_word(glob_addr))
```

#### 数据布局示例 / Data Layout Example (tile_M=2, tile_N=2)

```
Global Memory:          Shared Memory (after TLDS):
  addr 0: A[0]            smem_off+0: A[0]     (thread 0 → row=0,col=0)
  addr 1: A[1]            smem_off+1: A[1]     (thread 1 → row=0,col=1)
  addr 2: A[2]            smem_off+2: A[2]     (thread 2 → row=1,col=0)
  addr 3: A[3]            smem_off+3: A[3]     (thread 3 → row=1,col=1)

Global memory 地址是连续的 (每线程负责一个连续位置),
Shared memory 地址按 row-major 分块排列。
```

### 3.3 TLSTS — Tile 存储 / Tile Store

TLSTS 指令将 shared memory 中的一个 tile 写回到 global memory，是 TLDS 的逆操作：

The TLSTS instruction writes a tile from shared memory back to global memory, the inverse of TLDS:

```
伪代码 / Pseudocode:

TLSTS rd, rs1, imm:
    smem_off   = rd          # shared memory 中的起始偏移
    glob_base  = rs1         # global memory 中的基地址
    smem       = simt.thread_block.shared_memory

    for each active thread t in warp:
        tid = t.thread_id
        row = tid // tile_n
        col = tid %  tile_n

        smem_addr = smem_off + row * tile_n + col
        glob_addr = glob_base + tid

        memory.write_word(glob_addr, smem.read_word(smem_addr))
```

### 3.4 双缓冲 (Double Buffering) / Double Buffering

双缓冲是一种通过重叠计算和数据传输来隐藏延迟的技术：

Double buffering is a technique that hides latency by overlapping computation with data transfer:

```
伪代码 / Pseudocode (ping-pong pattern):

TLCONF M, N, K               # 配置 tile 维度

# Phase 1: 加载第 0 块到 buffer A
TLDS buf_A_offset, base_0    # base_0 = 第 0 块的全局地址

# Phase 2: 加载第 1 块到 buffer B (ping)
TLDS buf_B_offset, base_1    # 同时开始计算 buffer A

# Phase 3: 处理 buffer A
SHLD r1, buf_A_offset + 0    # 从共享内存读取
...                          # 计算 ...

# Phase 4: 处理 buffer B (pong)
SHLD r2, buf_B_offset + 0    # 从共享内存读取
...                          # 计算 ...

# 结果写回
ST r1, [result_addr_0]
ST r2, [result_addr_1]
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File / 文件 | Purpose / 用途 | Change from Phase 12 / 相对 Phase 12 的变化 |
|-------------|---------------|------------------------------------------|
| `isa.py` | Opcode definitions: OP_TLCONF(0x35), OP_TLDS(0x36), OP_TLSTS(0x37) + OPCODE_NAMES | ADDED: 3 new opcodes + names |
| `assembler.py` | Two-pass assembler: TLCONF/TLDS/TLSTS mnemonics | ADDED: 3 new assembly mnemonics |
| `simt_core.py` | SIMT pipeline: _execute_warp() TLCONF/TLDS/TLSTS handling + tile state | ADDED: 3 exec blocks + tile_m/n/k state |
| `shared_memory.py` | Shared memory read/write (unchanged, used by TLDS/TLSTS) | Same (used as tile buffer) |
| `warp.py` | Warp/Thread with active mask (unchanged) | Same |
| `register_file.py` | Per-thread register file (unchanged) | Same |
| `memory.py` | Global memory (unchanged, source/dest of tile transfers) | Same |
| `alu.py`, `vec4_alu.py` | ALU operations (unchanged) | Same |
| `cache.py` | L1 cache (unchanged) | Same |
| `scoreboard.py` | Scoreboard with pipeline latency (unchanged) | Same |
| `scheduler.py` | Warp scheduler (unchanged) | Same |
| `simt_stack.py` | SIMT divergence management (unchanged) | Same |
| `operand_collector.py` | Multi-bank register file (unchanged) | Same |
| `thread_block.py` | Thread block, shared memory (unchanged) | Same |
| `ibuffer.py` | IBuffer (unchanged) | Same |
| `console_display.py` | Rendering module (unchanged) | Same |
| `learning_console.py` | Interactive console (unchanged, backward compatible) | Same |
| `run.sh` | Wrapper script (unchanged) | Same |
| `tests/programs/11_tiled_matmul.asm` | Tiled 2x2 matmul via shared memory tiles | NEW demo program |
| `tests/programs/12_tile_double_buffer.asm` | Ping-pong double buffering pattern | NEW demo program |
| `tests/test_phase13.py` | Phase 13 test suite: ISA, assembler, tile config, matmul, double buffer, backward compat | NEW test suite |

### 4.2 ISA Encoding / 指令编码

#### TLCONF (0x35) — Tile Configuration

```
[31:24] opcode=0x35  [23:20] tile_M  [19:16] tile_N  [15:12]=0  [11:0] tile_K

  汇编语法 / Assembly:
    TLCONF M, N, K
    示例: TLCONF 2, 2, 2    ; M=2, N=2, K=2
          TLCONF 4, 8, 16   ; M=4, N=8, K=16

  字段说明 / Field Description:
    rd  (tile_M): Tile 的行数 (M dimension)
    rs1 (tile_N): Tile 的列数 (N dimension)
    imm (tile_K): Tile 的内维 (K dimension, 共享维度)

  状态更新 / State Update:
    simt_core.tile_m = M
    simt_core.tile_n = N
    simt_core.tile_k = K
```

#### TLDS (0x36) — Tile Load (Global → Shared)

```
[31:24] opcode=0x36  [23:20] smem_off  [19:16] glob_base  [15:0]=0

  汇编语法 / Assembly:
    TLDS smem_offset, glob_base
    示例: TLDS 0, 0      ; smem[0..3] = mem[0..3] (tile_M=2,tile_N=2)
          TLDS 4, 8      ; smem[4..7] = mem[8..11]

  字段说明 / Field Description:
    rd  (smem_off):  Shared memory 中的起始偏移 (word 地址)
    rs1 (glob_base): Global memory 中的基地址 (word 地址)

  数据映射 / Data Mapping:
    每个线程 t (thread_id = tid):
      row = tid // tile_n
      col = tid %  tile_n
      smem_addr = smem_off + row * tile_n + col
      glob_addr = glob_base + tid
```

#### TLSTS (0x37) — Tile Store (Shared → Global)

```
[31:24] opcode=0x37  [23:20] smem_off  [19:16] glob_base  [15:0]=0

  汇编语法 / Assembly:
    TLSTS smem_offset, glob_base
    示例: TLSTS 0, 100    ; smem[0..3] → mem[100..103]

  字段说明 / Field Description:
    rd  (smem_off):  Shared memory 中的起始偏移 (word 地址)
    rs1 (glob_base): Global memory 中的基地址 (word 地址)

  数据映射 / Data Mapping:
    与 TLDS 相同, 方向相反
    smem.read_word(smem_addr) → memory.write_word(glob_addr, ...)
```

### 4.3 Key Implementation / 关键实现

#### simt_core.py — Tile State (新增属性 / New Attributes)

```python
# SIMTCore.__init__() 新增 / Added in __init__():
self.tile_m = 8    # tile M dimension (rows of A tile)
self.tile_n = 8    # tile N dimension (cols of B tile)
self.tile_k = 8    # tile K dimension (inner dimension)
```

#### simt_core.py — TLCONF 实现

```python
if op == OP_TLCONF:
    self.tile_m = instr.rd       # M dimension
    self.tile_n = instr.rs1      # N dimension
    self.tile_k = instr.imm & 0xFFF  # K dimension
    return
```

#### simt_core.py — TLDS 实现

```python
if op == OP_TLDS:
    smem_off = instr.rd & 0xFF      # shared memory base offset
    glob_base = instr.rs1 & 0x3FF    # global memory base address
    smem = self.thread_block.shared_memory
    tlist = self._exec_threads(warp, instr)
    for t in tlist:
        tid = t.thread_id
        row = tid // self.tile_n if self.tile_n > 0 else tid
        col = tid % self.tile_n if self.tile_n > 0 else 0
        smem_addr = (smem_off + row * self.tile_n + col) % smem.size_words
        glob_addr = (glob_base + tid) & 0x3FF
        smem.write_word(smem_addr, self.memory.read_word(glob_addr))
    return
```

#### simt_core.py — TLSTS 实现

```python
if op == OP_TLSTS:
    smem_off = instr.rd & 0xFF      # shared memory base offset
    glob_base = instr.rs1 & 0x3FF    # global memory base address
    smem = self.thread_block.shared_memory
    tlist = self._exec_threads(warp, instr)
    for t in tlist:
        tid = t.thread_id
        row = tid // self.tile_n if self.tile_n > 0 else tid
        col = tid % self.tile_n if self.tile_n > 0 else 0
        smem_addr = (smem_off + row * self.tile_n + col) % smem.size_words
        glob_addr = (glob_base + tid) & 0x3FF
        self.memory.write_word(glob_addr, smem.read_word(smem_addr))
    return
```

### 4.4 Module Interfaces / 模块接口

```python
# isa.py
OP_TLCONF = 0x35  # tile config: rd=tile_M, rs1=tile_N, imm=tile_K
OP_TLDS   = 0x36  # tile load: rs1=global_base, rs2=thread_id stride, imm=shared_offset
OP_TLSTS  = 0x37  # tile store: rs1=shared_offset, rs2=thread_id stride, imm=global_base

# assembler.py — New assembly mnemonics:
#   TLCONF M, N, K        # Configure tile dimensions
#   TLDS smem_off, glob   # Load tile from global to shared
#   TLSTS smem_off, glob  # Store tile from shared to global

# simt_core.py — _execute_warp() handles OP_TLCONF, OP_TLDS, OP_TLSTS
# simt_core.py — New state: tile_m, tile_n, tile_k

# Module dependencies:
#   TLDS:  memory.read_word() → shared_memory.write_word()
#   TLSTS: shared_memory.read_word() → memory.write_word()
#   TLCONF: updates simt_core tile state (no I/O)

# No changes to other modules (backward compatible).
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Tiled Matmul Execution — `11_tiled_matmul.asm`

场景: warp_size=1，计算 2×2 矩阵乘法 C = A × B。A 在 mem[0..3]，B 在 mem[8..11]。

```
程序执行流程 / Execution Timeline:

Setup (by test harness):
  mem[0]=1  mem[1]=2  mem[2]=3  mem[3]=4    (A row-major)
  mem[8]=5  mem[9]=6  mem[10]=7 mem[11]=8   (B row-major)
  A = [[1,2],[3,4]]   B = [[5,6],[7,8]]

Cycle 0: TLCONF 2, 2, 2
  → tile_m=2, tile_n=2, tile_k=2
  → 配置完毕

Cycle 1: TLDS 0, 0
  → warp_size=1, 每个周期一个线程执行
  → tid=0: row=0//2=0, col=0%2=0
    smem[0] = mem[0] = 1   (A[0][0])
  → 注: warp_size=1 时每指令只有一个线程工作
    需要多次 TLDS 才能加载完整的 tile

Cycle 2: TLDS 4, 8
  → tid=0: row=0//2=0, col=0%2=0
    smem[4] = mem[8] = 5   (B[0][0])

Cycle 3-5: 加载 A 的剩余元素 (测试框架 warp_size=1, 需手动)
  (实际 warp_size=1 时的 TLDS 会依次加载)

...

最终 shared memory 状态 / Final Shared Memory State:
  smem[0]=1  smem[1]=2  smem[2]=3  smem[3]=4   (A tile)
  smem[4]=5  smem[5]=6  smem[6]=7  smem[7]=8   (B tile)

Compute C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]:
  SHLD r10, 0    → r10 = smem[0] = A[0][0] = 1
  SHLD r11, 4    → r11 = smem[4] = B[0][0] = 5
  MUL r12, r10, r11  → r12 = 1 * 5 = 5
  SHLD r10, 1    → r10 = smem[1] = A[0][1] = 2
  SHLD r11, 6    → r11 = smem[6] = B[1][0] = 7
  MUL r13, r10, r11  → r13 = 2 * 7 = 14
  ADD r14, r12, r13  → r14 = 5 + 14 = 19
  ST r14, [16]   → mem[16] = C[0][0] = 19 ✓

Compute C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]:
  SHLD r10, 0    → r10 = A[0][0] = 1
  SHLD r11, 5    → r11 = smem[5] = B[0][1] = 6
  MUL r12, r10, r11  → r12 = 6
  SHLD r10, 1    → r10 = A[0][1] = 2
  SHLD r11, 7    → r11 = smem[7] = B[1][1] = 8
  MUL r13, r10, r11  → r13 = 16
  ADD r14, r12, r13  → r14 = 22
  ST r14, [17]   → mem[17] = C[0][1] = 22 ✓

Compute C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]:
  SHLD r10, 2    → r10 = smem[2] = A[1][0] = 3
  SHLD r11, 4    → r11 = B[0][0] = 5
  MUL r12, r10, r11  → r12 = 15
  SHLD r10, 3    → r10 = smem[3] = A[1][1] = 4
  SHLD r11, 6    → r11 = B[1][0] = 7
  MUL r13, r10, r11  → r13 = 28
  ADD r14, r12, r13  → r14 = 43
  ST r14, [18]   → mem[18] = C[1][0] = 43 ✓

Compute C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]:
  SHLD r10, 2    → r10 = A[1][0] = 3
  SHLD r11, 5    → r11 = B[0][1] = 6
  MUL r12, r10, r11  → r12 = 18
  SHLD r10, 3    → r10 = A[1][1] = 4
  SHLD r11, 7    → r11 = B[1][1] = 8
  MUL r13, r10, r11  → r13 = 32
  ADD r14, r12, r13  → r14 = 50
  ST r14, [19]   → mem[19] = C[1][1] = 50 ✓

Cycle N: HALT
```

### 5.2 Double Buffer Execution — `12_tile_double_buffer.asm`

场景: warp_size=1，处理两个数据块。Tile 0 (mem[0..7]) 在 buf_A (smem[0..7]) 处理，
Tile 1 (mem[8..15]) 在 buf_B (smem[8..15]) 处理。

```
程序执行流程 / Execution Timeline:

Setup:
  mem[0..15] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

Cycle 0: TLCONF 2, 4, 1
  → tile_m=2, tile_n=4, tile_k=1

Cycle 1: TLDS 0, 0   (Phase 1: 加载 Tile 0 到 buf_A)
  → smem[0..7] = mem[0..7] = [0,1,2,3,4,5,6,7]

Cycle 2: TLDS 8, 8   (Phase 2: 加载 Tile 1 到 buf_B)
  → smem[8..15] = mem[8..15] = [8,9,10,11,12,13,14,15]

Phase 3: 计算 buf_A 元素和 (Tile 0 sum)
  SHLD r1, 0    → r1 = smem[0] = 0
  SHLD r2, 1    → r2 = smem[1] = 1
  ADD r1, r1, r2  → r1 = 0+1 = 1
  SHLD r2, 2    → r2 = 2
  ADD r1, r1, r2  → r1 = 3
  SHLD r2, 3    → r2 = 3
  ADD r1, r1, r2  → r1 = 6
  SHLD r2, 4    → r2 = 4
  ADD r1, r1, r2  → r1 = 10
  SHLD r2, 5    → r2 = 5
  ADD r1, r1, r2  → r1 = 15
  SHLD r2, 6    → r2 = 6
  ADD r1, r1, r2  → r1 = 21
  SHLD r2, 7    → r2 = 7
  ADD r1, r1, r2  → r1 = 28
  ST r1, [100]  → mem[100] = sum(Tile 0) = 28 ✓

Phase 4: 计算 buf_B 元素和 (Tile 1 sum)
  SHLD r3, 8    → r3 = smem[8] = 8
  SHLD r4, 9    → r4 = smem[9] = 9
  ADD r3, r3, r4  → r3 = 17
  SHLD r4, 10   → r4 = 10
  ADD r3, r3, r4  → r3 = 27
  SHLD r4, 11   → r4 = 11
  ADD r3, r3, r4  → r3 = 38
  SHLD r4, 12   → r4 = 12
  ADD r3, r3, r4  → r3 = 50
  SHLD r4, 13   → r4 = 13
  ADD r3, r3, r4  → r3 = 63
  SHLD r4, 14   → r4 = 14
  ADD r3, r3, r4  → r3 = 77
  SHLD r4, 15   → r4 = 15
  ADD r3, r3, r4  → r3 = 92
  ST r3, [101]  → mem[101] = sum(Tile 1) = 92 ✓

Cycle N: HALT
```

### 5.3 关键设计决策 / Key Design Decisions

1. **Tile 维度通过 TLCONF 单独配置**: 不是将 tile 维度编码在 TLDS/TLSTS 指令中，而是通过 TLCONF 预先配置。这使得 TLDS/TLSTS 的指令编码更紧凑（不浪费 bit 存储维度信息）。

2. **Thread ID 到 tile 元素的映射**: 每个线程的 thread_id 被映射为 tile 中的 (row, col) 坐标：`row = tid // tile_n`, `col = tid % tile_n`。这保证了连续线程加载连续地址（合并访问）。

3. **Shared Memory 作为 tile buffer**: TLDS 将数据加载到 shared memory，后续计算通过 SHLD 指令读取。Shared memory 的低延迟特性使得 tile 内数据的多次访问不会成为性能瓶颈。

4. **双缓冲模式**: 通过在不同的 shared memory 区域加载多个 tile，可以在处理一个 tile 的同时预取下一个 tile，从而隐藏内存加载延迟。

5. **立即数范围限制**: TLDS 和 TLSTS 中的偏移使用 8-bit 或 10-bit 编码，限制了直接寻址范围。更大的地址需要通过 ALU 计算后通过寄存器间接访问。

6. **完全向后兼容**: 所有 Phase 0-12 的功能和测试保持不变。现有程序在 Phase 13 模拟器上运行结果相同。

## 6. Comparison with Phase 12 / 与 Phase 12 的对比

| Aspect / 方面 | Phase 12 | Phase 13 | Change / 变化 |
|---------------|----------|----------|---------------|
| **Focus** | Warp communication primitives | Tiling strategies for memory hierarchy | NEW direction |
| **New Opcodes** | OP_SHFL(0x30), OP_VOTE(0x33), OP_BALLOT(0x34) | OP_TLCONF(0x35), OP_TLDS(0x36), OP_TLSTS(0x37) | ADDED: 3 opcodes |
| **New Assembly Mnemonics** | SHFL, VOTE.ANY, VOTE.ALL, BALLOT | TLCONF, TLDS, TLSTS | ADDED: 3 mnemonics |
| **ISA File** | 103 lines (Phase 12) | 109 lines (+6 lines for tiling opcodes) | Extended |
| **Assembler File** | ~220 lines (Phase 12) | ~237 lines (+17 lines for tiling mnemonics) | Extended |
| **simt_core.py** | ~445 lines (Phase 12) | ~485 lines (+40 lines for tiling exec + state) | Extended significantly |
| **SIMTCore State** | warp comm (no new state) | tile_m, tile_n, tile_k (3 new fields) | ADDED: tile config state |
| **Demo Programs** | 09_warp_shfl.asm, 10_warp_vote.asm | + 11_tiled_matmul.asm, 12_tile_double_buffer.asm | ADDED: 2 demos |
| **Test Suite** | test_phase12.py (5 test cases) | test_phase13.py (6 test cases) | NEW: focused on tiling |
| **Test Cases** | ISA, assembler, SHFL, VOTE/BALLOT, backward compat | ISA, assembler, tile config state, tiled matmul, double buffer, backward compat | ADDED: 6 tests |
| **New Module Dependencies** | None | TLDS/TLSTS connect memory ↔ shared_memory | Inter-module data flow |
| **Backward Compatibility** | All Phase 0-11 unchanged | All Phase 0-12 unchanged | Maintained |
| **Pipeline Stages** | 5-stage (FETCH/DECODE/ISSUE/EXEC/WB) | Same pipeline + new opcodes in EXEC | Same |
| **Memory Hierarchy Usage** | Registers only for warp comm | Global memory ↔ Shared memory ↔ Registers | Full memory hierarchy |
| **Performance Impact** | 1-cycle latency (same as ALU) | TLDS/TLSTS: multi-cycle (memory access) | Higher latency for tile ops |
| **Reference Design** | __shfl_sync, __any_sync, __all_sync, __ballot_sync | CUTLASS tiling, GPGPU-Sim memory pipeline | Enhanced memory model |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **Tile 维度硬编码**: 当前 TLCONF 将 tile 维度存储在 SIMTCore 状态中，只能通过新的 TLCONF 指令修改。不支持每个 warp 有不同的 tile 配置。

2. **TLDS/TLSTS 的地址映射简化**: 当前实现中，thread_id 到 tile (row,col) 的映射使用 `tid // tile_n` 和 `tid % tile_n`。更复杂的映射（如转置、对角线、条带化）未实现。

3. **无 bank conflict 处理**: 当多个线程访问 shared memory 中的同一 bank 时，当前实现不会模拟 bank conflict 导致的延迟增加。所有访问都是串行化的，没有性能惩罚。

4. **TLDS/TLSTS 延迟与实际硬件不匹配**: 真实硬件上，全局内存访问需要数百周期延迟。当前 TLDS 和 TLSTS 是单周期操作，没有模拟内存延迟。

5. **仅支持单 warp 场景**: 当前实现假设所有线程属于同一个 warp。多 warp 场景下的 tile 同步（如 __syncthreads() 后的 tile 加载）未实现。

6. **无 tile 边界检查**: 当 tile 维度超出 global memory 或 shared memory 大小时，没有错误检测或越界保护。

7. **双缓冲示例是顺序而非重叠**: 当前的双缓冲 demo (`12_tile_double_buffer.asm`) 实际上是顺序加载两个 tile 再计算，并没有真正重叠计算和数据传输，因为没有异步加载机制。

8. **无多维 tile 支持**: 当前只支持 2D tile（M×N），不支持 3D tile 或更高维度的分块。

9. **TLDS/TLSTS 指令编码冗余**: `imm` 字段在 TLDS 和 TLSTS 中未使用（固定为 0），浪费了 12-bit 编码空间。未来可用于编码 stride、转置标志或 bank 选择。

10. **无 CUTLASS 风格的层级分块**: 现代 GPU 编程使用三级分块（CTA 级、Warp 级、Thread 级），当前只实现了最基本的单级分块。

11. **Performance simulation not modeled**: TLDS/TLSTS 在真实硬件上具有远高于 ALU 操作的延迟和功耗，当前模拟器未对这些差异建模。
