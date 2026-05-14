## Quick Start

Interactive debugging with learning_console.py (kernel launch and GTO scheduler display):

```bash
# Interactive debugging with GTO scheduler
python src/learning_console.py tests/programs/01_gto_schedule.asm --num-warps 4 --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_gto_schedule.asm --trace --warp-size 4

# Multi-block kernel demo
python src/learning_console.py tests/programs/02_multi_block.asm --num-warps 4 --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive kernel debugger with GTO scheduler state display, multi-warp scheduling visualization
- **PRED/Predication**: `@p0` predicate support across all instruction types in kernel execution
- **vec4/float4 instructions**: V4PACK, V4ADD, V4MUL, V4UNPACK support in assembler and execution
- **Warp-level registers**: WREAD/WWRITE for warp-uniform register access during kernel execution
- **Trace mode**: `--trace` for batch execution with scheduler cycle-by-cycle tracking

# Phase 6: Kernel Launch & 调度

GPU 顶层：Kernel Launch + GTO 调度 + 性能计数器。

## 新增
- `GPUSim` 顶层类: `launch_kernel(program, grid_dim, block_dim)`
- GTO 调度策略 (Greedy Then Oldest)
- PerfCounters: IPC, stall rate, active cycles
- Multi-block kernel 执行

## 运行

```bash
cd phase6_kernel && bash run.sh
```

## 对标 GPGPU-Sim
`gpgpu_sim` 顶层 + GTO scheduler
