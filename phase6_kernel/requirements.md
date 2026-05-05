# Phase 6: Kernel Launch & Scheduling — 需求

## 1. 目标

补齐 GPU 编程模型的最后一环：**Kernel Launch（grid/block 两级并行）** 和 **GTO 调度策略**。

## 2. 新增功能

### FR-01: GTO Scheduler
- 对标 GPGPU-Sim 默认调度策略 GTO (Greedy Then Oldest)
- 优先执行有 oldest 指令的 warp
- Warp 优先级队列

### FR-02: Kernel Launch
- `launch_kernel(program, grid_dim, block_dim)` API
- 自动创建 grid 中所有 blocks
- 每个 block 包含 warps（block_dim / warp_size）

### FR-03: 性能计数器
- IPC (Instructions Per Cycle)
- Stall reasons (scoreboard / barrier / branch / none)
- Warp occupancy

### FR-04: GPU 顶层
- `GPUSim` 类聚合所有组件
- 一键 launch + run + report

## 3. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | GTO 策略选择 oldest warp |
| AC-02 | Multi-block kernel 正确执行 |
| AC-03 | Performance report 输出 |
| AC-04 | Phase 0-5 测试保持通过 |
