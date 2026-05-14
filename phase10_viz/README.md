## Quick Start

Use learning_console.py with `--trace` flag to generate trace data for visualization:

```bash
# Generate trace data with the learning console
python src/learning_console.py ../phase9_tensor/tests/programs/01_mma_dot.asm --auto --warp-size 4

# Run visualization tests
bash run.sh

# Tracing with JSON export for Chrome Tracing
python src/learning_console.py ../phase7_pipeline/tests/programs/01_ibuffer_basic.asm --auto --warp-size 4
```

## New in this update

- **TraceCollector**: Comprehensive cycle-by-cycle event collection system with cycle/warp/PC/opcode/stall tracking
- **ASCII Warp Timeline**: Visual timeline showing each warp's PC over execution cycles
- **Stall Analysis**: Bar chart visualization showing stall cause distribution (scoreboard/barrier/ibuffer)
- **Memory Heatmap**: Density-based character heatmap of memory access patterns
- **JSON Export**: Chrome Tracing compatible format for external tool analysis (`chrome://tracing`)
- **`full_report()`**: One-call generation of complete analysis report
- **learning_console.py**: Integrated trace visualization support
- **PRED/Predication**: Predicate state tracked in trace events
- **vec4/float4 instructions**: Vec4 operations captured in trace data
- **Warp-level registers**: Warp register reads/writes tracked in execution events

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
