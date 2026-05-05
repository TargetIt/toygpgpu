# Phase 4 实验报告

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
