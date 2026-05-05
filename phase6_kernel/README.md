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
