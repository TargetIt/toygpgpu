# Phase 10 设计: 可视化工具链

## TraceCollector
- `record_exec(cycle, warp_id, pc, opcode, active, mem_addr)`
- `record_stall(cycle, warp_id, reason)`
- `export_json(filepath)`: 标准 JSON 格式

## ASCII Warp Timeline
- 每行 = 1 cycle, 每列 = 1 warp
- 显示 4-char 指令缩写
- 默认最多显示 80 cycles

## Stall Analysis
- 按 stall_reason 分组计数
- ASCII 柱状图: █ 字符, 最大宽度 40

## Memory Heatmap
- 统计每个地址的访问次数
- 密度映射: ` .:-=+*#%@` (10 级)
- 宽度可配 (默认 32 列)

## Full Report
- 组合 timeline + stall + heatmap
- 单次调用生成完整分析
