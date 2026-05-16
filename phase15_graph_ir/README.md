## Quick Start

```bash
# Phase 15: Compute Graph IR demo (build + validate + export)
python -c "
import sys; sys.path.insert(0, 'src')
from graph_ir import build_example_graph
g = build_example_graph()
print('Validate:', g.validate())
print('Topo order:', g.topological_order())
print()
print('--- DOT ---')
print(g.to_dot())
print()
print('--- JSON ---')
print(g.to_json())
"

# Phase 15: Cycle detection demo
python -c "
import sys; sys.path.insert(0, 'src')
from graph_ir import ComputeGraph
g = ComputeGraph('cyclic')
a = g.add_kernel('A', block_dim=(4,))
b = g.add_kernel('B', block_dim=(4,), dependencies=[a])
g.nodes[a].dependencies.append(b)  # introduce cycle
print(g.validate())  # (False, 'Cycle detected at node 0')
"

# Phase 15: JSON round-trip demo
python -c "
import sys; sys.path.insert(0, 'src')
from graph_ir import build_example_graph
import json
g = build_example_graph()
j = g.to_json()
g2 = ComputeGraph.from_json(j)
print(g2.validate())
print('Reconstructed nodes:', len(g2.nodes))
"

# Phase 15: Learning console graph command
python src/learning_console.py --graph-info

# Run all Phase 14 tests (CuTile parser + WGMMA)
python tests/test_phase14.py

# Run all Phase 13 tests (Tiling)
python tests/test_phase13.py

# Run all Phase 12 tests (Warp Communication)
python tests/test_phase12.py

# CuTile DSL: parse and run the CuTile matmul demo
python -c "
import sys; sys.path.insert(0, 'src')
from cutile_parser import assemble_cutile
from simt_core import SIMTCore
src = open('tests/programs/13_cutile_matmul.cutile').read()
md = {'A':{'base':0},'B':{'base':8},'C':{'base':16}}
code, asm = assemble_cutile(src, md)
simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
for a,v in [(0,1),(1,2),(2,3),(3,4),(8,5),(9,6),(10,7),(11,8)]: simt.memory.write_word(a, v)
simt.load_program(code); simt.run()
for a in [16,17,18,19]: print(f'mem[{a}]={simt.memory.read_word(a)}')
"

# Run the tiled matrix multiply demo (Phase 13)
python src/learning_console.py tests/programs/11_tiled_matmul.asm

# Run the double buffer demo (Phase 13)
python src/learning_console.py tests/programs/12_tile_double_buffer.asm

# Run the SHFL (warp shuffle) demo (Phase 12)
python src/learning_console.py tests/programs/09_warp_shfl.asm

# Run the VOTE & BALLOT demo (Phase 12)
python src/learning_console.py tests/programs/10_warp_vote.asm

# Run existing Phase 11 demos (backward compatible)
python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4
python src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5
```

## New in Phase 15

- **Compute Graph IR** (`src/graph_ir.py`, ~213 lines): A DAG-based intermediate representation for computation graphs, inspired by NVIDIA CUDA Graphs, XLA HLO, and MLIR.
  - **GraphNode**: DAG node with `node_id`, `op_type`, `name`, `params`, `dependencies` — supports 5 types: `kernel`, `memcpy`, `barrier`, `input`, `output`
  - **ComputeGraph**: Graph manager with node addition, DAG validation (DFS cycle detection), topological sort (Kahn's algorithm), and serialization
- **DAG Validation**: `validate()` returns `(bool, msg)` — checks empty graph, missing dependencies, and cycles via three-color DFS (White-Gray-Black)
- **Topological Sort**: `topological_order()` returns node IDs in Kahn BFS order — all dependencies appear before dependents
- **JSON Serialization**: `to_json()` / `from_json()` — round-trip safe, human-readable, cross-language compatible
- **DOT Visualization**: `to_dot()` — Graphviz format with node shapes (kernel=box, memcpy=ellipse, barrier=diamond, input=invhouse, output=house)
- **Example graph**: `build_example_graph()` creates a pipeline: kernel_A → kernel_B → barrier → kernel_C
- **Zero dependencies**: `graph_ir.py` uses Python stdlib only — no dependency on other toygpgpu modules
- **No ISA changes**: Phase 15 is a pure software IR layer — no new opcodes, instructions, or pipeline stages
- **Full backward compatibility**: All existing Phase 0-14 programs and tests run unchanged
- **CUDA Graphs reference**: `ComputeGraph` ~ `cudaGraphCreate`, `add_kernel` ~ `cudaGraphAddKernelNode`, `validate` ~ `cudaGraphInstantiate`, `to_json` ~ `cudaGraphExportToJSON`

## New in Phase 14

- **CuTile Programming Model**: A high-level tile-oriented DSL for describing matrix multiply computations concisely
  - **CuTile DSL Parser** (`src/cutile_parser.py`): Parse `.cutile` files into IR and auto-generate toygpgpu assembly
  - **CuTile DSL Syntax**: `tile M=<int>, N=<int>, K=<int>`, `kernel name(params) { load / mma / store }` operations
  - **Full pipeline**: CuTile DSL → CuTileKernel IR → toygpgpu Asm → Machine Code → Execution
- **OP_WGMMA** (opcode 0x38): Warp-group Matrix Multiply-Accumulate instruction operating on shared memory tiles
  - Syntax: `WGMMA smem_a, smem_b, smem_c` — computes `smem[C] += A × B` using tile_m/n/k from TLCONF
  - Single-instruction hardware-accelerated MMA compared to Phase 13's manual SHLD+MUL+ADD loops
- **New demo**: `13_cutile_matmul.cutile` — CuTile DSL description of 2×2 matrix multiplication
- **Test suite**: 6 test cases covering ISA encoding, CuTile parser, code generation, WGMMA execution, end-to-end DSL pipeline, and backward compatibility with Phase 0-13 features
- **Two programming approaches**: (1) High-level CuTile DSL with auto-generated code, (2) Direct WGMMA assembly for compact hardware-accelerated execution
- **CUTLASS/Triton reference**: CuTile DSL inspired by NVIDIA CUTLASS and OpenAI Triton programming models
- **Full backward compatibility**: All existing Phase 0-13 programs run unchanged

# Phase 14: CuTile 编程模型

在 Phase 0-13 的基础上，增加了 CuTile 编程模型，提供高级 tile DSL 解析器和 WGMMA 硬件加速指令。

## 新增功能

| 功能 | 说明 |
|------|------|
| CuTile DSL 解析器 (`cutile_parser.py`) | 高级 tile 描述语言 → 自动生成 ISA 汇编代码 |
| WGMMA 指令 (opcode 0x38) | Warp-group MMA: smem[A] × smem[B] → smem[C] |

## CuTile DSL 语法

```cutile
tile M=2, N=2, K=2

kernel matmul(A:[M,K], B:[K,N], C:[M,N]) {
    load A[0:M, 0:K] -> smem[0]
    load B[0:K, 0:N] -> smem[4]
    mma smem[0], smem[4] -> smem[8]
    store smem[8] -> C[0:M, 0:N]
}
```

## WGMMA 指令

```
WGMMA smem_a, smem_b, smem_c
  smem[C] += smem[A] × smem[B]
  使用 TLCONF 配置的 tile_m, tile_n, tile_k
```

## 运行

```bash
# Phase 14 测试 (CuTile + WGMMA)
python3 tests/test_phase14.py

# CuTile DSL 端到端执行
python3 -c "
import sys; sys.path.insert(0, 'src')
from cutile_parser import assemble_cutile; from simt_core import SIMTCore
src = open('tests/programs/13_cutile_matmul.cutile').read()
code, _ = assemble_cutile(src, {'A':{'base':0},'B':{'base':8},'C':{'base':16}})
s = SIMTCore(1, 1, 256)
for a,v in [(0,1),(1,2),(2,3),(3,4),(8,5),(9,6),(10,7),(11,8)]: s.memory.write_word(a, v)
s.load_program(code); s.run()
for a in [16,17,18,19]: print(f'mem[{a}]={s.memory.read_word(a)}')
"

# WGMMA 直接汇编执行
# 编辑: TLCONF 2,2,2; TLDS 0,0; TLDS 4,8; WGMMA 0,4,8; TLSTS 8,16; HALT
```

## 对标

| 真实系统 | toygpgpu |
|---------|----------|
| CUTLASS GemmUniversal | `parse_cutile()` + `generate_asm()` |
| CUTLASS WarpLevelMma | `WGMMA` 指令 |
| Triton `tl.dot` | `mma smem[a], smem[b] -> smem[c]` |
| NVIDIA Tensor Core wgmma | `OP_WGMMA (0x38)` |
| CUDA C++ vs PTX | CuTile DSL vs 汇编 |

## New in Phase 13

- **Tiling Strategies**: 3 new instructions for tiled data movement between global memory and shared memory
  - **TLCONF** (opcode 0x35): Configure tile dimensions (M=rd, N=rs1, K=imm) — sets SIMTCore tile state
  - **TLDS** (opcode 0x36): Load tile from global memory to shared memory — data flows: DRAM -> SRAM
  - **TLSTS** (opcode 0x37): Store tile from shared memory to global memory — data flows: SRAM -> DRAM
- **SIMTCore tile state**: 3 new fields (`tile_m`, `tile_n`, `tile_k`) track the active tile configuration for TLDS/TLSTS operations
- **2 new demo programs**: `11_tiled_matmul.asm` (2x2 matrix multiply via shared memory tiles, A and B tiles loaded separately) and `12_tile_double_buffer.asm` (ping-pong pattern with two tile buffers in shared memory)
- **Test suite**: 6 test cases covering ISA encoding, assembler, tile config state, tiled matmul, double buffer, and backward compatibility with Phase 0-12 features
- **Memory hierarchy utilization**: Full global → shared → register memory hierarchy demonstrated, in contrast to Phase 12 which only used registers
- **CUTLASS/GPGPU-Sim reference**: Tiling approach inspired by NVIDIA CUTLASS tile iterators and GPGPU-Sim memory pipeline tile loading
- **Full backward compatibility**: All existing Phase 0-12 programs run unchanged

# Phase 13: 分块策略

在 Phase 0-12 的基础上，增加了三种分块指令，支持将全局内存数据分块加载到共享内存进行处理。

## 新增指令

| 指令 | Opcode | 功能 | 对应模式 |
|------|--------|------|----------|
| TLCONF | 0x35 | 配置 tile 维度 (M, N, K) | CUTLASS TileCoord |
| TLDS | 0x36 | 全局内存 → 共享内存加载 | 手动 tile 加载 |
| TLSTS | 0x37 | 共享内存 → 全局内存写回 | 手动 tile 存储 |

## 运行

```bash
# 分块矩阵乘法演示
python3 src/learning_console.py tests/programs/11_tiled_matmul.asm

# 双缓冲演示
python3 src/learning_console.py tests/programs/12_tile_double_buffer.asm

# 测试
python3 tests/test_phase13.py
```

## New in Phase 12

- **Warp Communication primitives**: 3 new instructions for warp-level collaboration without shared memory
  - **SHFL** (opcode 0x30): Cross-thread register read with IDX/UP/DOWN/XOR modes — equivalent to CUDA `__shfl_sync()`
  - **VOTE** (opcode 0x33): Warp-wide ANY/ALL reduction — equivalent to CUDA `__any_sync()` / `__all_sync()`
  - **BALLOT** (opcode 0x34): Bitmask of threads with non-zero predicate — equivalent to CUDA `__ballot_sync()`
- **2 new demo programs**: `09_warp_shfl.asm` (4 SHFL modes + store to memory) and `10_warp_vote.asm` (ANY/ALL/BALLOT with even/odd thread conditions)
- **Test suite**: 5 test cases covering ISA encoding, assembler, SHFL execution, VOTE/BALLOT execution, and backward compatibility with Phase 0-11 features
- **Zero pipeline changes**: New instructions execute in the existing EXEC stage with 1-cycle latency
- **Full backward compatibility**: All existing Phase 0-11 programs (demo_basic.asm, demo_divergence.asm) run unchanged
- **Educational focus**: Primitive names (SHFL, VOTE, BALLOT) chosen for clarity over CUDA/PTX naming conventions

# Phase 12: Warp 级协作原语

在 Phase 0-11 的基础上，增加了三种 Warp 级通信指令，使同一 warp 内的线程可以直接交换数据，无需经过 shared memory。

## 新增指令

| 指令 | Opcode | 功能 | CUDA 等价 |
|------|--------|------|-----------|
| SHFL | 0x30 | 跨线程寄存器读取 (4种模式) | `__shfl_sync()` |
| VOTE.ANY | 0x33 | 任意线程非零? | `__any_sync()` |
| VOTE.ALL | 0x33 | 全部线程非零? | `__all_sync()` |
| BALLOT | 0x34 | 非零线程 bitmask | `__ballot_sync()` |

## 运行

```bash
# Warp Shuffle 演示
python3 src/learning_console.py tests/programs/09_warp_shfl.asm

# VOTE & BALLOT 演示
python3 src/learning_console.py tests/programs/10_warp_vote.asm

# 测试
python3 tests/test_phase12.py
```

# Phase 11: 交互式学习控制台

GDB 风格的 GPU 流水线单步调试器。每周期显示完整内部状态。

## 特性
- 五级流水线: FETCH/DECODE/ISSUE/EXEC/WB
- 寄存器变化追踪 (old→new 含符号解释)
- Scoreboard 倒计时可视化
- I-Buffer 槽位状态
- SIMT Stack 栈内容
- 内存变化 delta
- ANSI 颜色标注
- 断点支持

## 使用

```bash
# 交互式单步
python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

# 自动播放
python3 src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5
```

## 交互命令
- `Enter` 单步 | `r` 自动运行 | `b 5` 断点 | `q` 退出
- `reg/sb/ib/stack/m` 查看状态

## 对标
GDB `stepi` 模式 + GPGPU-Sim 无等价物（教学定制）
