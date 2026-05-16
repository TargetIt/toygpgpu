# Phase 14: CuTile Programming Model — Design Document / 设计文档

> **对应 GPGPU-Sim**: GPU 编程模型中的 tile 抽象 (CUTLASS TileCoord, Triton DSL)
> **参考**: NVIDIA CUTLASS (github.com/NVIDIA/cutlass), Triton DSL (github.com/openai/triton),
> PyTorch 的 tile 编程抽象

## 1. Introduction / 架构概览

```
                        ┌──────────────────────────────────────────────────┐
                        │           CuTile Programming Model              │
                        │                                                  │
                        │  CuTile DSL (High-Level)                        │
                        │  ┌──────────────────────────────────────┐       │
                        │  │  tile M=2, N=2, K=2                  │       │
                        │  │  kernel matmul(A, B, C) {            │       │
                        │  │      load A -> smem[0]               │       │
                        │  │      load B -> smem[4]               │       │
                        │  │      mma smem[0], smem[4] -> smem[8] │       │
                        │  │      store smem[8] -> C              │       │
                        │  │  }                                   │       │
                        │  └──────────────┬───────────────────────┘       │
                        │                 │ cutile_parser.py              │
                        │                 ▼                               │
                        │  toygpgpu Assembly (Generated Code)             │
                        │  ┌──────────────────────────────────────┐       │
                        │  │ TLCONF 2,2,2 / TLDS / SHLD / MUL/ADD │       │
                        │  │ -- or --                             │       │
                        │  │ TLCONF 2,2,2 / TLDS / WGMMA / TLSTS  │       │
                        │  └──────────────┬───────────────────────┘       │
                        │                 │                               │
                        │                 ▼                               │
                        │  SIMTCore Execution Engine                      │
                        │  ┌──────────────────────────────────────┐       │
                        │  │ FETCH → DECODE → ISSUE → EXEC → WB   │       │
                        │  │ WGMMA: smem[A] × smem[B] → smem[C]  │       │
                        │  └──────────────────────────────────────┘       │
                        └──────────────────────────────────────────────────┘
```

Phase 14 向 toygpgpu 添加了**CuTile 编程模型**，这是一种面向 tile 的高级抽象层。CuTile 提供：

1. **CuTile DSL**: 一种类似 Triton/CUTLASS 的高级 tile 描述语言，自动生成 ISA 代码
2. **OP_WGMMA (0x38)**: 一种新的 warp-group MMA 指令，在共享内存 tile 上执行矩阵乘法累加

这两者共同构成了从高级 tile 描述到低级 ISA 执行的完整编程模型。

Phase 14 adds the **CuTile Programming Model**, a high-level tile-oriented abstraction layer to toygpgpu. CuTile provides:

1. **CuTile DSL**: A Triton/CUTLASS-like high-level tile description language that auto-generates ISA code
2. **OP_WGMMA (0x38)**: A new warp-group MMA instruction that performs matrix multiply-accumulate on shared memory tiles

Together they form a complete programming model from high-level tile description to low-level ISA execution.

### 新增指令 / New Instruction

| 指令 | Opcode | 功能 | 对应模式 |
|------|--------|------|----------|
| WGMMA | 0x38 | Warp-group MMA: smem[A] x smem[B] -> smem[C] | CUTLASS WarpLevelMma / Triton `tl.dot` |

### 新增文件 / New File

| 文件 | 说明 |
|------|------|
| `src/cutile_parser.py` | CuTile DSL 解析器 (~237 行)，实现 parse → generate → assemble 完整管线 |

## 2. Motivation / 设计动机

### 2.1 为什么需要 CuTile 编程模型？/ Why CuTile?

Phase 13 引入了 TLCONF/TLDS/TLSTS 等分块指令，但编写分块矩阵乘法仍然需要手动管理：
- 手动计算 shared memory 地址偏移
- 手动编写 tile 元素的加载/计算/存储循环
- 手动处理循环展开和累加逻辑

While Phase 13 introduced tiling instructions (TLCONF/TLDS/TLSTS), writing tiled matrix multiplication still requires:
- Manual shared memory address calculations
- Manual tile load/compute/store loops
- Manual loop unrolling and accumulation logic

CuTile 和 WGMMA 的设计动机 / CuTile and WGMMA are motivated by:

1. **抽象层次提升 / Higher Abstraction Level**: CuTile DSL 让程序员只需声明 `load/mma/store` 操作，由解析器自动生成完整的汇编代码。类似 CUDA 相对于 PTX 的角色。
2. **硬件加速的 MMA / Hardware-Accelerated MMA**: WGMMA 指令将矩阵乘法累加从软件循环提升为硬件指令，类似 NVIDIA GPU 上的 Tensor Core MMA 指令。
3. **代码可读性 / Code Readability**: CuTile DSL 代码比等价的汇编代码更简洁、更易理解，降低了教学门槛。
4. **自动代码生成 / Automatic Code Generation**: `cutile_parser.py` 的 `generate_asm()` 函数自动展开 M×N×K 循环，生成正确的 SHLD/MUL/ADD 序列。

### 2.2 与真实 GPU 编程模型的对应 / Correspondence to Real GPU Models

```
真实 GPU 编程模型      │ toygpgpu            │ 说明
───────────────────────┼─────────────────────┼────────────────────
CUDA C++ (cuBLAS)     │ CuTile DSL          │ 高级 tile 描述语言
PTX wgmma.mma         │ WGMMA 指令          │ 硬件级 warp-group MMA
CUTLASS Gemm           │ cutile_parser       │ tile 迭代器 + 代码生成
Triton tl.dot          │ WGMMA smem_a,smem_b │ tile 级点积指令
Tensor Cores           │ WGMMA (概念对应)    │ 共享内存 tile 级矩阵乘
```

### 2.3 与 CUTLASS 的对应关系 / CUTLASS Correspondence

NVIDIA CUTLASS 使用模板元编程实现 tile 迭代和 warp 级 MMA：

```
CUTLASS 概念               │ toygpgpu              │ 说明
───────────────────────────┼───────────────────────┼────────────────────
TileCoord                  │ TLCONF M,N,K          │ tile 维度配置
GlobalToSharedLoader       │ TLDS                  │ 全局→共享加载
WarpLevelMma               │ WGMMA（或 SHLD+ALU）  │ warp 级矩阵乘法
SharedToGlobalStore        │ TLSTS                 │ 共享→全局存储
Epilogue                   │ ST                    │ 结果写回
GemmUniversal              │ parse_cutile()        │ 高层 gemm 描述
```

### 2.4 与 GPGPU-Sim / Triton 的对应 / GPGPU-Sim / Triton Correspondence

```
概念                      │ toygpgpu                │ 说明
─────────────────────────┼─────────────────────────┼──────────────────
GPGPU-Sim memory_pipeline│ TLDS/TLSTS               │ 内存 tile 传输
Triton DSL (tl.dot)      │ CuTile DSL (mma)         │ 高级 tile DSL
Tensor Core warp-inst    │ OP_WGMMA                 │ 硬件加速 MMA
GPGPU-Sim scoreboard     │ Scoreboard（未修改）     │ 指令调度
```

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 CuTile DSL 语法与解析 / CuTile DSL Syntax and Parsing

CuTile DSL 是一种面向 tile 的小型领域特定语言，定义在 `cutile_parser.py` 中。

CuTile DSL is a small tile-oriented domain-specific language defined in `cutile_parser.py`.

#### 语法规范 / Syntax Specification

```
# Tile 声明（必需） / Tile declaration (required)
tile M=<int>, N=<int>, K=<int>

# Kernel 声明 / Kernel declaration
kernel <name>(<param>:[M,K], <param>:[K,N], <param>:[M,N]) {
    load <matrix>[0:M, 0:K] -> smem[<offset>]
    load <matrix>[0:K, 0:N] -> smem[<offset>]
    mma smem[<A_off>], smem[<B_off>] -> smem[<C_off>]
    store smem[<C_off>] -> <matrix>[0:M, 0:N]
}
```

#### 解析流程 / Parsing Flow

```
CuTile DSL 源码
    │
    ▼
_clean_source()           # 移除注释和空行
    │
    ▼
逐行匹配正则表达式:
  - tile M=N=K= → TileConfig
  - kernel (params) → 内核名和参数
  - load A -> smem → load 操作
  - store smem -> C → store 操作
  - mma smem, smem -> smem → mma 操作
    │
    ▼
CuTileKernel IR 对象
    │
    ▼
generate_asm()            # 生成 ISA 汇编代码
    │
    ▼
assemble()                # 汇编为机器码
```

#### 支持的注释 / Supported Comments

```python
# CuTile DSL 支持两种注释风格
# CuTile DSL supports two comment styles:

// C++ 风格行注释 / C++ style line comment
; 汇编风格行注释 / Assembly style line comment
```

### 3.2 WGMMA — Warp-Group MMA / WGMMA 指令

WGMMA (Warp-Group Matrix Multiply-Accumulate) 指令在共享内存 tile 上执行矩阵乘法累加：
`C[m][n] += sum_k(A[m][k] * B[k][n])`

The WGMMA (Warp-Group Matrix Multiply-Accumulate) instruction performs matrix multiply-accumulate on shared memory tiles:
`C[m][n] += sum_k(A[m][k] * B[k][n])`

```
伪代码 / Pseudocode:

WGMMA smem_a, smem_b, smem_c:
    smem = shared_memory
    M = tile_m (来自 TLCONF)
    N = tile_n (来自 TLCONF)
    K = tile_k (来自 TLCONF)

    for m in range(M):
        for n in range(N):
            acc = smem[smem_c + m * N + n]  # 已有的累加值
            for k in range(K):
                a_val = smem[smem_a + m * K + k]
                b_val = smem[smem_b + k * N + n]
                acc += a_val * b_val
            smem[smem_c + m * N + n] = acc
```

#### 数据流 / Data Flow

```
Shared Memory (before WGMMA):

  smem[smem_a]:     A tile (M x K, row-major)
    [A00] [A01] ... [A0(K-1)]
    [A10] [A11] ... [A1(K-1)]
    ...
    [A(M-1)0] ...  [A(M-1)(K-1)]

  smem[smem_b]:     B tile (K x N, row-major)
    [B00] [B01] ... [B0(N-1)]
    [B10] [B11] ... [B1(N-1)]
    ...
    [B(K-1)0] ...  [B(K-1)(N-1)]

  smem[smem_c]:     C tile (M x N, row-major, 累加器)
    [C00] [C01] ... [C0(N-1)]
    [C10] [C11] ... [C1(N-1)]
    ...
    [C(M-1)0] ...  [C(M-1)(N-1)]

After WGMMA:
  smem[smem_c]:     C[m][n] = original_C[m][n] + sum_k(A[m][k] * B[k][n])
```

#### 编码 / Encoding

```
WGMMA: [31:24]=0x38  [23:20]=smem_a  [19:16]=smem_b  [15:12]=smem_c  [11:0]=0

  汇编语法 / Assembly:
    WGMMA smem_a_off, smem_b_off, smem_c_off
    示例: WGMMA 0, 4, 8
          ; smem[0..3] (A) × smem[4..7] (B) → smem[8..11] (C)

  字段说明 / Field Description:
    rd  (smem_a): A tile 在 shared memory 中的偏移 (word 地址)
    rs1 (smem_b): B tile 在 shared memory 中的偏移 (word 地址)
    rs2 (smem_c): C tile (累加器) 在 shared memory 中的偏移 (word 地址)

  依赖 / Dependencies:
    - 需要先执行 TLCONF 设置 tile_m, tile_n, tile_k
    - 需要先执行 TLDS 将 A 和 B tile 加载到 shared memory
    - C tile 在共享内存中初始化为 0 或已有累加值
```

### 3.3 CuTile 代码生成策略 / Code Generation Strategy

`generate_asm()` 函数将高层的 CuTile 操作转换为具体的 ISA 指令序列。

The `generate_asm()` function translates high-level CuTile operations into concrete ISA instruction sequences.

#### load → TLDS

每个 `load <matrix> -> smem[offset]` 操作生成一条 TLDS 指令：

```python
# 输入: load A -> smem[0], matrix_data = {'A': {'base': 0}}
# 输出:
; Load A tile → shared memory[0]
TLDS 0, 0
```

#### store → TLSTS

每个 `store smem[offset] -> <matrix>` 操作生成一条 TLSTS 指令：

```python
# 输入: store smem[8] -> C, matrix_data = {'C': {'base': 16}}
# 输出:
; Store result from shared memory[8] → C
TLSTS 8, 16
```

#### mma → SHLD + MUL/ADD 循环展开

每个 `mma smem[A], smem[B] -> smem[C]` 操作生成完全展开的 M×N×K 三层循环：

```python
# 对于 M×N×K tile，展开为 M*N 个输出元素的计算
# 每个输出 C[i][j] 需要 K 次乘积 + (K-1) 次累加
# 使用 SHLD 从 shared memory 读取，使用 MUL/ADD 计算

; C[0][0] = sum over k
SHLD r10, <A_tile_base + 0*K + 0>   ; A[0][0]
SHLD r11, <B_tile_base + 0*N + 0>   ; B[0][0]
MUL r12, r10, r11                    ; A[0][0] * B[0][0]
; ... 更多 k 项
ADD r12, r12, r13                    ; accumulate
ST r12, [<C_tile_base + 0*N + 0>]   ; Store C[0][0]
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File / 文件 | Purpose / 用途 | Change from Phase 13 / 相对 Phase 13 的变化 |
|-------------|---------------|------------------------------------------|
| `cutile_parser.py` | CuTile DSL parser: parse_cutile() → CuTileKernel IR, generate_asm() → ISA code, assemble_cutile() → machine code | **NEW** (~237 lines) |
| `isa.py` | Opcode definitions: OP_WGMMA(0x38) + OPCODE_NAMES entry | ADDED: 1 new opcode + name |
| `assembler.py` | Two-pass assembler: WGMMA mnemonic (smem_a, smem_b, smem_c) | ADDED: 1 new assembly mnemonic |
| `simt_core.py` | SIMT pipeline: _execute_warp() WGMMA handling — smem[A]×smem[B]→smem[C] | ADDED: ~20 lines WGMMA exec block |
| `shared_memory.py` | Shared memory read/write (unchanged, used by WGMMA/TLDS/TLSTS) | Same |
| `warp.py` | Warp/Thread with active mask (unchanged) | Same |
| `register_file.py` | Per-thread register file (unchanged) | Same |
| `memory.py` | Global memory (unchanged) | Same |
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
| `tests/programs/13_cutile_matmul.cutile` | CuTile DSL matmul: tile M=2,N=2,K=2, load A→smem[0], load B→smem[4], mma→smem[8], store→C | **NEW** demo (.cutile file) |
| `tests/test_phase14.py` | Phase 14 test suite: ISA, parser, codegen, WGMMA, e2e, backward compat | **NEW** test suite (6 tests) |

### 4.2 ISA Encoding / 指令编码

#### WGMMA (0x38) — Warp-Group MMA

```
[31:24] opcode=0x38  [23:20] smem_a  [19:16] smem_b  [15:12] smem_c  [11:0]=0

  汇编语法 / Assembly:
    WGMMA smem_a_off, smem_b_off, smem_c_off
    示例: WGMMA 0, 4, 8
          ; A(smem[0..3]) × B(smem[4..7]) → C(smem[8..11])

  字段说明 / Field Description:
    rd  = smem_a:  A tile 在 shared memory 中的基地址偏移
    rs1 = smem_b:  B tile 在 shared memory 中的基地址偏移
    rs2 = smem_c:  C tile (累加器) 在 shared memory 中的基地址偏移
    imm = 0:      保留字段 (未使用)

  编码示例 / Encoding Examples:
    WGMMA 0, 4, 8    → 0x38000480  (smem_a=0, smem_b=4, smem_c=8)
    WGMMA 2, 6, 10   → 0x382260A0  (smem_a=2, smem_b=6, smem_c=10)
```

### 4.3 Key Implementation / 关键实现

#### cutile_parser.py — CuTileKernel IR (核心数据结构)

```python
class TileConfig:
    """Tile shape configuration."""
    def __init__(self, m: int = 8, n: int = 8, k: int = 8):
        self.M = m
        self.N = n
        self.K = k

class CuTileKernel:
    """Parsed CuTile kernel representation."""
    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Tuple[str, str]] = {}  # name -> (shape_str, role)
        self.tile: Optional[TileConfig] = None
        self.ops: List[Dict] = []  # [{op, matrix, smem_off}, ...]
```

#### cutile_parser.py — parse_cutile() (DSL 解析)

```python
def parse_cutile(source: str) -> CuTileKernel:
    """Parse CuTile DSL source → CuTileKernel IR."""
    kernel = None
    lines = _clean_source(source)

    for line in lines:
        # tile M=N=N=K=N → TileConfig
        m = re.match(r'tile\s+M\s*=\s*(\d+)\s*,\s*N\s*=\s*(\d+)\s*,\s*K\s*=\s*(\d+)', line, re.I)
        if m:
            kernel.tile = TileConfig(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        # kernel name(params) { → parse name and parameters
        m = re.match(r'kernel\s+(\w+)\s*\((.*?)\)\s*\{?', line, re.I)
        if m:
            kernel.name = m.group(1)
            # Parse "A:[M,K], B:[K,N], C:[M,N]"
            ...

        # load A[0:M, 0:K] -> smem[0]
        m = re.match(r'load\s+(\w+)\[.*?\]\s*->\s*smem\[(\d+)\]', line, re.I)
        if m:
            kernel.ops.append({'op': 'load', 'matrix': m.group(1), 'smem_off': ...})

        # store smem[8] -> C[0:M, 0:N]
        m = re.match(r'store\s+smem\[(\d+)\]\s*->\s*(\w+)\[.*?\]', line, re.I)
        if m:
            kernel.ops.append({'op': 'store', 'smem_off': ..., 'matrix': ...})

        # mma smem[0], smem[4] -> smem[8]
        m = re.match(r'mma\s+smem\[(\d+)\]\s*,\s*smem\[(\d+)\]\s*->\s*smem\[(\d+)\]', line, re.I)
        if m:
            kernel.ops.append({'op': 'mma', 'smem_a': ..., 'smem_b': ..., 'smem_c': ...})
```

#### cutile_parser.py — generate_asm() (代码生成)

```python
def generate_asm(kernel: CuTileKernel, matrix_data=None) -> str:
    """Generate toygpgpu assembly from CuTileKernel IR."""
    tile = kernel.tile
    lines = [f"; Tile shape: M={tile.M}, N={tile.N}, K={tile.K}"]

    # TLCONF
    lines.append(f"TLCONF {tile.M}, {tile.N}, {tile.K}")

    for op in kernel.ops:
        if op['op'] == 'load':
            lines.append(f"TLDS {op['smem_off']}, {glob_base}")
        elif op['op'] == 'store':
            lines.append(f"TLSTS {op['smem_off']}, {glob_base}")
        elif op['op'] == 'mma':
            # Expand: for i in rows, for j in cols, for k in K
            # SHLD / MUL / ADD for each term
            ...

    return '\n'.join(lines)
```

#### simt_core.py — WGMMA 执行

```python
if op == OP_WGMMA:
    smem_a = instr.rd & 0xFF    # A tile base in shared memory
    smem_b = instr.rs1 & 0xFF   # B tile base in shared memory
    smem_c = instr.rs2 & 0xFF   # C tile base in shared memory
    smem = self.thread_block.shared_memory
    M, N, K = self.tile_m, self.tile_n, self.tile_k

    # Compute C[m][n] += sum_k(A[m][k] * B[k][n])
    for m in range(M):
        for n in range(N):
            acc = smem.read_word(smem_c + m * N + n)  # existing C value
            for k in range(K):
                a_val = smem.read_word(smem_a + m * K + k)
                b_val = smem.read_word(smem_b + k * N + n)
                acc = (acc + a_val * b_val) & 0xFFFFFFFF
            smem.write_word(smem_c + m * N + n, acc)
    return
```

#### assembler.py — WGMMA 汇编器支持

```python
if mnemonic == 'WGMMA':
    smem_a = _parse_int(parts[1])   # smem offset for A tile
    smem_b = _parse_int(parts[2])   # smem offset for B tile
    smem_c = _parse_int(parts[3])   # smem offset for C (accumulator)
    return encode_rtype(OP_WGMMA, smem_a, smem_b, smem_c)
```

### 4.4 Module Interfaces / 模块接口

```python
# isa.py
OP_WGMMA = 0x38  # warp-group MMA: rd=smem_a, rs1=smem_b, rs2=smem_c, imm=0

# assembler.py — New assembly mnemonic:
#   WGMMA smem_a, smem_b, smem_c  # A×B→C tile matmul in shared memory

# cutile_parser.py — Public API:
#   parse_cutile(source: str) -> CuTileKernel
#   generate_asm(kernel, matrix_data) -> str
#   assemble_cutile(source, matrix_data) -> (machine_code, asm_text)

# simt_core.py — _execute_warp() handles OP_WGMMA
#   Reads tile_m, tile_n, tile_k from SIMTCore state (inherited from Phase 13)
#   Reads A/B tiles from shared memory, writes C tile to shared memory

# Module dependencies:
#   WGMMA:  shared_memory.read_word() × K × M × N read operations
#           shared_memory.write_word() × M × N write operations
#   No changes to other modules (backward compatible).
```

### 4.5 CuTile 完整管线 / Full CuTile Pipeline

```
CuTile DSL Source (.cutile file)
    │
    ▼
parse_cutile() ──────→ CuTileKernel IR (内存中表示)
    │                      │
    │                      ├── TileConfig (M, N, K)
    │                      ├── params dict
    │                      └── ops list [{load/store/mma}]
    │
    ▼
generate_asm() ────────→ Assembly Text (string)
    │                      │
    │                      └── TLCONF + TLDS + SHLD/MUL/ADD + TLSTS + HALT
    │
    ▼
assemble() ─────────────→ Machine Code (List[int])
    │
    ▼
load_program() + run() ─→ SIMTCore Execution
    │
    ▼
Memory Verification (test assertions)
```

#### assemble_cutile() 便捷函数 / Convenience Function

```python
def assemble_cutile(source: str, matrix_data: Dict = None) -> Tuple[List[int], str]:
    """Full CuTile pipeline: parse → generate → assemble."""
    kernel = parse_cutile(source)
    asm_text = generate_asm(kernel, matrix_data)
    machine_code = assemble(asm_text)
    return machine_code, asm_text
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 CuTile DSL Matmul — `13_cutile_matmul.cutile`

场景: 2×2 矩阵乘法 C = A × B。A 在 mem[0..3]，B 在 mem[8..11]。

CuTile 源码 / CuTile Source:

```cutile
tile M=2, N=2, K=2

kernel matmul(A:[M,K], B:[K,N], C:[M,N]) {
    load A[0:M, 0:K] -> smem[0]
    load B[0:K, 0:N] -> smem[4]
    mma smem[0], smem[4] -> smem[8]
    store smem[8] -> C[0:M, 0:N]
}
```

自动生成的汇编 / Auto-Generated Assembly:

```asm
; Auto-generated from CuTile: matmul
; Tile shape: M=2, N=2, K=2

; Configure tile dimensions
TLCONF 2, 2, 2

; Load A tile → shared memory[0]
TLDS 0, 0

; Load B tile → shared memory[4]
TLDS 4, 8

; Warp-group MMA: smem[0] × smem[4] → smem[8]
; For each C[i][j], compute sum_k(A[i][k] * B[k][j])
; C[0][0] = sum over k
SHLD r10, 0     ; A[0][0]
SHLD r11, 4     ; B[0][0]
MUL r12, r10, r11
SHLD r10, 1     ; A[0][1]
SHLD r11, 6     ; B[1][0]
MUL r13, r10, r11
ADD r12, r12, r13
ST r12, [8]     ; Store C[0][0]

; C[0][1] = sum over k
SHLD r10, 0     ; A[0][0]
SHLD r11, 5     ; B[0][1]
MUL r12, r10, r11
SHLD r10, 1     ; A[0][1]
SHLD r11, 7     ; B[1][1]
MUL r13, r10, r11
ADD r12, r12, r13
ST r12, [9]     ; Store C[0][1]

; C[1][0] = sum over k
SHLD r10, 2     ; A[1][0]
SHLD r11, 4     ; B[0][0]
MUL r12, r10, r11
SHLD r10, 3     ; A[1][1]
SHLD r11, 6     ; B[1][0]
MUL r13, r10, r11
ADD r12, r12, r13
ST r12, [10]    ; Store C[1][0]

; C[1][1] = sum over k
SHLD r10, 2     ; A[1][0]
SHLD r11, 5     ; B[0][1]
MUL r12, r10, r11
SHLD r10, 3     ; A[1][1]
SHLD r11, 7     ; B[1][1]
MUL r13, r10, r11
ADD r12, r12, r13

; Store result from shared memory[8] → C
TLSTS 8, 16
HALT
```

执行结果 / Execution Results:
```
C[0][0] = 1*5 + 2*7 = 19  ✓  (mem[16])
C[0][1] = 1*6 + 2*8 = 22  ✓  (mem[17])
C[1][0] = 3*5 + 4*7 = 43  ✓  (mem[18])
C[1][1] = 3*6 + 4*8 = 50  ✓  (mem[19])
```

### 5.2 WGMMA Direct Assembly 示例 / WGMMA Direct Assembly Example

WGMMA 指令可以直接在汇编代码中使用，跳过 CuTile DSL 解析器：

The WGMMA instruction can be used directly in assembly, bypassing the CuTile DSL parser:

```asm
; Direct WGMMA usage (without CuTile DSL)
TLCONF 2, 2, 2       ; Configure tile dimensions
TLDS 0, 0             ; Load A tile: smem[0..3] = mem[0..3] = [1,2,3,4]
TLDS 4, 8             ; Load B tile: smem[4..7] = mem[8..11] = [5,6,7,8]
WGMMA 0, 4, 8         ; Hardware MMA: C = A × B, result in smem[8..11]
TLSTS 8, 16           ; Store result to mem[16..19]
HALT

; Results:
;   smem[8]  = 1*5 + 2*7 = 19  → mem[16]
;   smem[9]  = 1*6 + 2*8 = 22  → mem[17]
;   smem[10] = 3*5 + 4*7 = 43  → mem[18]
;   smem[11] = 3*6 + 4*8 = 50  → mem[19]
```

### 5.3 两种 MMA 方式的对比 / Two MMA Approaches

| 方面 | CuTile DSL (expanded SHLD+MUL) | Direct WGMMA |
|------|-------------------------------|-------------|
| 编程层次 | 高级 (DSL) | 低级 (汇编) |
| 代码量 | 4 行 DSL + 自动生成 ~40 行 asm | 5 行 asm |
| 执行速度 | 多周期 (每项 SHLD+MUL+ADD) | 单指令 (硬件加速) |
| 灵活性 | 自动展开，正确性有保证 | 需手动管理地址 |
| 教育价值 | 展示 tile 计算展开过程 | 展示硬件加速概念 |

### 5.4 关键设计决策 / Key Design Decisions

1. **CuTile DSL 作为独立解析器模块**: `cutile_parser.py` 独立于核心模拟器模块，可以单独测试和扩展。不需要修改现有模块即可添加新 DSL 语法。

2. **WGMMA 使用共享内存而非寄存器**: 与 NVIDIA Tensor Core 不同，WGMMA 使用共享内存作为输入和输出，而非寄存器。这简化了实现，但可能增加共享内存带宽压力。

3. **CuTile DSL 生成 Phase 13 风格代码**: `generate_asm()` 生成 TLDS/SHLD/ALU 指令，不直接使用 WGMMA。这意味着 CuTile DSL 代码在旧版模拟器（Phase 13）上也能运行（假设添加了解析器）。

4. **WGMMA 累加到已有 C tile**: WGMMA 读取 C tile 的现有值并累加，而不是覆盖。这支持了分步计算和累加模式，与 CUTLASS 的 warp-level MMA 语义一致。

5. **不可变的 tile 维度来源**: WGMMA 从 SIMTCore 状态（通过 TLCONF 设置）读取 tile 维度，而非从指令字中编码。这保持了与 TLDS/TLSTS 一致的配置模式。

6. **完全向后兼容**: 所有 Phase 0-13 的功能和测试保持不变。现有程序在 Phase 14 模拟器上运行结果相同。

## 6. Comparison with Phase 13 / 与 Phase 13 的对比

| Aspect / 方面 | Phase 13 | Phase 14 | Change / 变化 |
|---------------|----------|----------|---------------|
| **Focus** | Tiling strategies for memory hierarchy | CuTile programming model / tile DSL | NEW direction |
| **New Opcodes** | OP_TLCONF(0x35), OP_TLDS(0x36), OP_TLSTS(0x37) | OP_WGMMA(0x38) | ADDED: 1 opcode |
| **New Assembly Mnemonics** | TLCONF, TLDS, TLSTS | WGMMA | ADDED: 1 mnemonic |
| **New Files** | None | `cutile_parser.py` (~237 lines) | ADDED: 1 new module |
| **ISA File** | 109 lines (Phase 13) | 112 lines (+3 lines for WGMMA) | Extended |
| **Assembler File** | ~237 lines (Phase 13) | ~247 lines (+10 lines for WGMMA) | Extended |
| **simt_core.py** | ~485 lines (Phase 13) | ~505 lines (+20 lines for WGMMA exec) | Extended |
| **Programming Model** | Manual assembly-level tiling | High-level DSL + auto code generation | Elevated abstraction |
| **Demo Programs** | 11_tiled_matmul.asm, 12_tile_double_buffer.asm | + 13_cutile_matmul.cutile (CuTile DSL) | ADDED: 1 new demo |
| **Test Suite** | test_phase13.py (6 test cases) | test_phase14.py (6 test cases) | NEW: focused on CuTile |
| **Test Categories** | ISA, assembler, tile config, tiled matmul, double buffer, backward compat | ISA, parser, codegen, WGMMA, e2e, backward compat | Shift to DSL/WGMMA focus |
| **New Module Dependencies** | TLDS/TLSTS connect memory ↔ shared_memory | cutile_parser depends on assembler module | New dependency chain |
| **Backward Compatibility** | All Phase 0-12 unchanged | All Phase 0-13 unchanged | Maintained |
| **Pipeline Stages** | 5-stage (FETCH/DECODE/ISSUE/EXEC/WB) | Same pipeline + WGMMA in EXEC | Same |
| **Memory Hierarchy Usage** | Global memory ↔ Shared memory ↔ Registers | Same + WGMMA adds pure SMEM-to-SMEM compute | SMEM compute path |
| **Performance Model** | TLDS/TLSTS: multi-cycle (memory access) | WGMMA: single instruction, M×N×K operations | HW-accelerated MMA |
| **Reference Design** | CUTLASS tiling, GPGPU-Sim memory pipeline | CUTLASS WarpLevelMma, Triton DSL, Tensor Cores | Higher abstraction |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **WGMMA 是单指令但多周期执行**: 当前 WGMMA 在 EXEC 阶段一次性完成所有 M×N×K 次乘加操作，没有分步或流水线化。真实硬件上的 Tensor Core MMA 是高度流水线化的。

2. **CuTile DSL 不支持 WGMMA 代码生成**: 当前 `generate_asm()` 为 mma 操作生成 SHLD/MUL/ADD 展开而非 WGMMA 指令。需要增加一个"优化模式"来直接生成 WGMMA。

3. **不支持异步 WGMMA**: 真实 GPU 上的 `wgmma.mma_async` 指令是异步的 — 启动后程序可以继续执行其他指令，数据准备完成后结果可用。当前 WGMMA 是同步的。

4. **无多维 tile 索引**: CuTile DSL 只支持 `A[0:M, 0:K]` 这种简单的全 tile 范围语法。不支持 `A[2:4, 5:7]` 等子范围切片语法。

5. **无 tile 边界检查**: 当 tile 维度超出 shared memory 大小时，WGMMA 和 CuTile DSL 没有错误检测。

6. **仅支持单 warp 场景**: CuTile DSL 和 WGMMA 都假设所有操作在一个 warp 内完成。多 warp 协作的 tile 计算未实现。

7. **不支持非方形 tile**: WGMMA 支持所有 M×N×K 组合，但未针对非方形 tile（如 M=32, N=8, K=16）进行优化或对齐检查。

8. **CuTile DSL 不支持 else/条件**: 高级 DSL 中的条件分支和循环控制未实现。所有操作都是顺序执行的。

9. **无 CUTLASS 风格的层级分块**: CUTLASS 使用三级分块（CTA → Warp → Thread），CuTile 只实现了最顶层（CTA/tile 级）。

10. **WGMMA 无寄存器重载**: 在 NVIDIA GPU 上，`wgmma.fence` 和 `wgmma.commit_group` 等指令用于管理异步 WGMMA 的数据依赖。当前实现无此同步原语。

11. **Matrix data 传递依赖调用者**: `generate_asm()` 和 `assemble_cutile()` 的 `matrix_data` 参数需要调用者提供全局地址映射，CuTile DSL 本身不包含地址信息。

12. **DSL 语法固定不可扩展**: CuTile DSL 的语法解析使用硬编码的正则表达式，不支持用户自定义操作或扩展。要添加新操作需要修改解析器核心代码。
