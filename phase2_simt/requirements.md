# Phase 2: SIMT 核心 (Warp) — 需求分解

## 1. 目标

从 Phase 1 的纯 SIMD 向量处理器，升级为 **SIMT (Single Instruction, Multiple Threads)** 核心。

对标 GPGPU-Sim 中 `shader_core_ctx` 的 warp 管理和 `shd_warp_t` 的线程状态。

## 2. 核心概念变化

```
Phase 1 (SIMD):             Phase 2 (SIMT):
┌─────────────────┐        ┌─────────────────────┐
│ Vector reg v0   │        │ Warp 0               │
│  lane0=10       │        │  Thread 0: RF(16)    │
│  lane1=20       │   →    │  Thread 1: RF(16)    │
│  ...            │        │  ...                 │
│ All lanes share │        │  Thread 7: RF(16)    │
│ one PC           │        │  Share one PC       │
└─────────────────┘        └─────────────────────┘
```

关键差异：每个线程现在有**独立的寄存器堆**（不是共享向量寄存器）。

## 3. 功能需求

### FR-01: Thread 抽象
- 每个 thread 有独立 RegisterFile (16×32bit, r0=0)
- 每个 thread 有 thread_id
- 同 warp 内 threads 共享一个 PC

### FR-02: Warp 抽象
- WARP_SIZE 个线程组成一个 warp（默认 8）
- Warp 内所有 active 线程执行同一条指令
- 每个 warp 有 active_mask（活跃线程掩码）

### FR-03: Warp 调度器
- 支持 Round-Robin 调度（对应 GPGPU-Sim LRR）
- 每个周期选择一个 warp 执行一条指令
- 后续可扩展 GTO（Greedy Then Oldest）

### FR-04: 线程感知指令

| 指令 | 功能 |
|------|------|
| TID rd | rd = thread_id (0..WARP_SIZE-1) |
| WID rd | rd = warp_id (0..NUM_WARPS-1) |
| BAR | 同步屏障：等所有 active threads 到达 |

### FR-05: 执行模型
- 每周期取一个 warp，发一条指令
- 该 warp 的所有 active 线程并行执行
- SIMD 执行单元共享（线程串行化执行 but 功能并行）

### FR-06: 内存层次
- 全局内存：所有 warp/thread 共享
- 无 cache（Phase 5 加入）
- 访存按 thread_id 顺序串行（简化，无 coalescing）

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | 多线程独立执行：同程序不同数据，输出正确 |
| AC-02 | TID/WID 指令正确返回线程/warp ID |
| AC-03 | BAR 屏障同步正确 |
| AC-04 | Round-Robin 调度：warp 轮流执行 |
| AC-05 | Phase 0/1 测试保持通过（向后兼容） |
| AC-06 | 向量加法 kernel 在多线程下正确 |
