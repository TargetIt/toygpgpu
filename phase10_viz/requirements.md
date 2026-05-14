# Phase 10: 可视化需求

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added a visualization-oriented interactive debugger. Supports the `trace` command for detailed execution traces, the `timeline` command for ASCII-based warp PC timeline charts, and the `report` command for aggregated performance statistics.
- **Fixed IBuffer peek and reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the SIMT stack reconvergence check logic (same fix as Phase 7).
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing.

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
