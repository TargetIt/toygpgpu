# Phase 2 实验报告

## 2026-05-15: Feature Additions

### learning_console.py 交互式调试体验
- Phase 2 调试器支持 warp 级调试，可观察多 warp 并发执行
- 新增命令支持：
  - `warps`: 显示所有 warp 的状态（active/suspended/completed）、当前 PC 和 active mask
  - `threads <wid>`: 显示指定 warp 内所有线程的寄存器状态（r0-r15 per thread）
  - `scheduler`: 显示 warp scheduler 的当前状态和调度策略（Round-Robin）
- 单步调试可逐 warp 观察指令执行，验证 SIMT 模型中不同 warp 的独立执行路径
- trace 模式支持 warp 级输出，每个 trace 记录包含 wid、tid 和 active mask

### trace 模式验证
- Phase 2 trace 输出扩展包含以下字段：
  - `Cycle`: 当前周期
  - `Warp`: warp ID（0..NUM_WARPS-1）
  - `ActiveMask`: 当前活跃线程掩码（bitmask 格式）
  - `PC`: warp 的程序计数器
  - `Insn`: 指令助记符
  - `TID/WID`: 线程 ID 和 warp ID 指令执行
  - `Barrier`: BAR 指令的同步状态
- 多 warp 并发 trace 验证：Round-Robin 调度下 warp 0 和 warp 1 交替执行
- BAR 指令 trace 验证：所有线程到达 barrier 前 warp 进入等待状态，全部到达后恢复执行

### warp 寄存器行为测试
- 验证 SIMT 模型中每线程独立寄存器堆的正确性：同一 warp 内不同线程的 r1 寄存器互不干扰
- 测试 TID 指令正确返回线程在 warp 内的索引（0..7）
- 测试 WID 指令正确返回 warp 的全局 ID
- 验证 LD/ST 地址生成：base_addr + thread_id 产生正确的每线程独立内存访问模式

---

## 1. 实验概述

**目标**: 从 SIMD 向量处理器升级为 SIMT（Single Instruction Multiple Threads）核心。

**对标**: GPGPU-Sim 的 `shader_core_ctx` + `shd_warp_t` + `scheduler_unit`。

## 2. 新增/修改模块

| 文件 | 行数 | 说明 |
|------|------|------|
| warp.py | 110 | Thread + Warp 类 (对标 shd_warp_t) |
| scheduler.py | 55 | WarpScheduler (对标 scheduler_unit) |
| simt_core.py | 165 | SIMTCore (对标 shader_core_ctx) |
| isa.py | +6 | TID/WID/BAR 操作码 |
| assembler.py | +10 | TID/WID/BAR 解析 |

## 3. 架构对比

| 概念 | Phase 1 (SIMD) | Phase 2 (SIMT) |
|------|---------------|----------------|
| 并行单元 | Vector lane (lane 0..7) | Thread (tid 0..7) |
| 寄存器 | v0-v7 (向量, 共享) | 每线程 r0-r15 (独立) |
| PC | 全局 1 个 | 每 warp 1 个 (线程共享) |
| 调度 | 无 | Round-Robin (warp 间) |
| 线程 ID | 无 | TID/WID 指令 |
| 同步 | 无 | BAR 指令 |
| 访存 | base+fixed offset | **base + thread_id** |

## 4. SIMT 执行模型

- 每个周期选择一个 warp（Round-Robin）
- Warp 内所有 active 线程执行同一条指令
- 每线程使用自己的寄存器堆
- LD/ST 地址 = base_addr + thread_id（模仿 GPU 全局内存访问模式）

## 5. 测试结果

43/43 全部通过：
- Thread 独立性: 4/4 ✅
- Warp 管理: 8/8 ✅
- Scheduler: 4/4 ✅
- ISA: 3/3 ✅
- Assembler: 4/4 ✅
- 集成测试: 20/20 ✅
  - TID/WID 正确
  - 每线程独立计算
  - Barrier 同步
  - 多线程向量加法
  - 多 Warp 并发

## 6. 下一步

Phase 3: SIMT Stack — 分支发散与重汇聚
- 引入 SIMT Stack（后支配栈）
- Active mask 动态更新
- 支持 if/else 分支
- JMP/BEQ/BNE 控制流指令
