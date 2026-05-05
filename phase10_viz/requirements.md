# Phase 10: 可视化需求

## 1. 目标
为 toygpgpu 添加执行追踪和可视化分析工具。

## 2. 功能
- FR-01: Warp PC 时间线 (ASCII 甘特图)
- FR-02: Stall 原因统计分析 (柱状图)
- FR-03: 内存访问热力图 (密度字符)
- FR-04: JSON 追踪导出 (外部工具兼容)

## 3. 验收标准
- AC-01: 时间线正确展示每个 warp 的 PC 变化
- AC-02: Stall 统计包含 scoreboard/barrier 分类
- AC-03: 热力图密度与访问次数成正比
- AC-04: JSON 能被 Chrome Tracing 打开
