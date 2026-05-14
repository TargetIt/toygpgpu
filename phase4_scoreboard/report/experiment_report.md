# Phase 4 实验报告

## 2026-05-15: Feature Additions

### learning_console.py 交互式调试体验
- Phase 4 调试器新增 Scoreboard 状态查看命令：
  - `scoreboard`: 显示所有 warp 的 scoreboard 状态，包括每个寄存器的 pending write 信息
  - `stalls`: 显示当前被 scoreboard stall 的 warp 列表和 stall 原因（RAW/WAW）
  - `pipeline`: 显示流水线中正在执行的指令及其剩余周期
- 单步调试时可逐 cycle 观察 scoreboard 的 reserve/release 过程
- 直观展示 RAW 冒险检测：当目的寄存器的写操作尚未完成时，依赖该寄存器的后续指令被 stall

### trace 模式验证
- Phase 4 trace 输出扩展包含流水线状态字段：
  - `Cycle`: 当前周期
  - `Warp`: warp ID
  - `PC`: 程序计数器
  - `Insn`: 指令助记符
  - `Scoreboard`: 当前 scoreboard 保留状态（哪些寄存器正在等待写回）
  - `Stall`: 是否因数据冒险被暂停（Y/N）
  - `Pipeline`: 流水线中各阶段的指令分布
- LD/ST 指令（延迟 4 cycles）的 trace 显示正确的 scoreboard reserve → wait → release 序列
- ALU 指令（延迟 1 cycle）快速通过流水线，scoreboard 仅保留 1 cycle
- 验证 stall 解除机制：scoreboard 到期后 stall 标志自动清除，warp 恢复调度

### 流水线延迟与 scoreboard 冒险检测
- 验证 ALU 指令延迟配置：ADD/SUB/MUL/DIV 等 ALU 操作延迟 1 cycle
- 验证 LD/ST 指令延迟配置：内存访问延迟 4 cycles
- 验证分支指令延迟配置：JMP/BEQ/BNE 不写寄存器，延迟 0 cycles
- RAW 冒险测试序列：
  - `LD r1, [addr]` → 延迟 4 cycles
  - `ADD r2, r1, r1` → 检测到 r1 有 pending write，stall 直到 LD 完成
  - trace 显示 stall 期间 warp 被跳过，scoreboard 持续递减
  - stall 解除后 ADD 正常发射
- WAW 冒险测试序列：
  - `ADD r1, r2, r3` → reserve r1 1 cycle
  - `SUB r1, r4, r5` → 检测到 r1 有 pending write，stall
- 验证 r0 写保护：r0 永远不被 scoreboard reserved，写操作被静默忽略

### scoreboard 调度优化
- 验证 `select_warp` 返回 None 时的正确处理：当所有 warp 都被 scoreboard stall 时，返回 `has_active_warps()` 状态以继续推进 scoreboard
- 验证多个 warp 同时 stall 时的 Round-Robin 行为：stall 解除的 warp 在后续 cycle 中按顺序恢复调度

---

## 1. 实验概述

**目标**: 实现 Scoreboard 记分板，模拟流水线延迟和寄存器数据冒险检测。

**对标**: GPGPU-Sim `scoreboard` 类。

## 2. 新增模块

| 文件 | 行数 | 说明 |
|------|------|------|
| scoreboard.py | 95 | Scoreboard 类 + PIPELINE_LATENCY 配置 |

## 3. 设计要点

### 3.1 冒险检测
- **RAW**: 源寄存器有 pending write → stall
- **WAW**: 目的寄存器有 pending write → stall
- r0 永远不 reserved（硬连线 0）

### 3.2 流水线延迟
- ALU: 1 cycle
- LD/ST: 4 cycles
- 分支: 0 cycles（无目的寄存器）

### 3.3 执行流程
每个 cycle:
1. advance all scoreboards (减延迟，到期清除)
2. clear scoreboard_stalled if clean
3. scheduler selects warp (跳过 stalled)
4. scoreboard check before issue (RAW/WAW)
5. execute → reserve rd with pipeline latency
6. if stalled: skip warp, advance scoreboard next cycle

### 3.4 修复: select_warp 返回 None 时的处理
当所有 warp 都被 scoreboard stall 时，select_warp 返回 None。修改为 `return has_active_warps()` 以继续推进 scoreboard，等待延迟到期。

## 4. 测试结果

15/15 全部通过。
