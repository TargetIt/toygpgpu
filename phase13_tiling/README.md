## Quick Start

```bash
# Run the tiled matrix multiply demo (Phase 13)
python src/learning_console.py tests/programs/11_tiled_matmul.asm

# Run the double buffer demo (Phase 13)
python src/learning_console.py tests/programs/12_tile_double_buffer.asm

# Run all Phase 13 tests
python tests/test_phase13.py

# Run the SHFL (warp shuffle) demo (Phase 12)
python src/learning_console.py tests/programs/09_warp_shfl.asm

# Run the VOTE & BALLOT demo (Phase 12)
python src/learning_console.py tests/programs/10_warp_vote.asm

# Run all Phase 12 tests
python tests/test_phase12.py

# Run existing Phase 11 demos (backward compatible)
python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4
python src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5
```

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
