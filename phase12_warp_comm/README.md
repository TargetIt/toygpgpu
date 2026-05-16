## Quick Start

```bash
# Run the SHFL (warp shuffle) demo
python src/learning_console.py tests/programs/09_warp_shfl.asm

# Run the VOTE & BALLOT demo
python src/learning_console.py tests/programs/10_warp_vote.asm

# Run all Phase 12 tests
python tests/test_phase12.py

# Run existing Phase 11 demos (backward compatible)
python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4
python src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5
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
