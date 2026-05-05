# Phase 4: Scoreboard — 记分板与数据冒险

## 1. 目标

引入 GPGPU-Sim 的 **Scoreboard（记分板）** 机制，检测并处理 **RAW（Read After Write）** 和 **WAW（Write After Write）** 寄存器数据冒险。

对标 GPGPU-Sim `scoreboard` 类 (gpgpu-sim/scoreboard.h)。

## 2. 功能需求

### FR-01: Scoreboard
- 每个 warp 一个 scoreboard
- 跟踪每个寄存器的"正在写"状态（pending write reservation）
- 记录每个 pending write 还需要的周期数

### FR-02: 流水线延迟
- ALU 指令（ADD/SUB/MUL/DIV）：1 周期
- 访存指令（LD/ST）：4 周期
- 其他指令（MOV/TID/WID/BAR）：1 周期

### FR-03: 冒险检测
- **RAW（写后读）**: 源寄存器有 pending write → stall warp
- **WAW（写后写）**: 目的寄存器有 pending write → stall warp

### FR-04: Warp 调度
- 被 scoreboard stall 的 warp 不参与调度
- 每周期推进 scoreboard 状态（递减延迟计数）
- 延迟到期时清除 reservation

## 3. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | RAW 冒险检测正确（写后读到旧值被阻止） |
| AC-02 | WAW 冒险检测正确（两次写同一 reg 有序） |
| AC-03 | 流水线延迟后结果正确（LD 4 周期后数据可用） |
| AC-04 | 无冒险程序正常运行 |
| AC-05 | Phase 0/1/2/3 测试保持通过 |
