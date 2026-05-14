# Phase 11: 学习控制台需求

## New Features (2026-05-15)

The following features were added after the initial release:

- **Fixed IBuffer peek and reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the SIMT stack reconvergence check logic (same fix as Phase 7).
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace/--auto mode to learning_console.py**: Added command-line flags `--trace` for verbose instruction tracing and `--auto` for unattended continuous execution mode in the learning console.
- **run.sh --trace support**: Added `--trace` flag support to the `run.sh` launch script, propagating trace mode to the underlying simulator.

## 1. 目标
为初学者提供逐周期、逐流水级的 GPU 内部状态观察工具。

## 2. 功能
- FR-01: 交互式单步执行 (Enter 键)
- FR-02: 五级流水线可视化
- FR-03: 寄存器变化追踪
- FR-04: Scoreboard/I-Buffer/SIMT Stack 实时显示
- FR-05: 断点支持
- FR-06: 自动运行模式

## 3. 验收标准
- AC-01: 单步模式每周期输出完整状态
- AC-02: 流水线、Scoreboard、I-Buffer、Stack 同时可见
- AC-03: 断点命中时暂停
- AC-04: 分支发散程序的 SIMT Stack push/pop 可见
