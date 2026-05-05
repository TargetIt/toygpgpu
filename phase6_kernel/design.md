# Phase 6 设计: Kernel Launch

## GTO Scheduler
- 跟踪每个 warp 最近一次 issue 时间 (`_warp_last_issue`)
- 选择 oldest pending warp: `min(candidates, key=lambda w: _warp_last_issue[w.warp_id])`
- 对标 GPGPU-Sim 的 GTO (Greedy Then Oldest)

## Kernel Launch
- `launch_kernel(program, grid_dim, block_dim)`
- total_blocks = product(grid_dim)
- num_warps_per_block = total_threads // warp_size
- 为每个 block 创建独立的 SIMTCore

## PerfCounters
- IPC = total_instructions / total_cycles
- stall_scoreboard: scoreboard stall cycles
- stall_barrier: barrier wait cycles
- active_cycles: cycles with instruction issued
