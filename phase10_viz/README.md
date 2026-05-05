# Phase 10: 可视化与工具链

执行追踪可视化：Warp Timeline、Stall Analysis、Memory Heatmap、JSON Export。

## 新增
- `TraceCollector`: 逐周期事件收集
- ASCII Warp Timeline: 每 warp PC 时间线
- Stall Analysis: 柱状图展示 stall 原因分布
- Memory Heatmap: 密度字符热力图
- JSON Export: Chrome Tracing 兼容格式
- `full_report()`: 一键生成完整报告

## 运行

```bash
cd phase10_viz && bash run.sh
```

## 对标 GPGPU-Sim
`AerialVision` + `stat-tool`
