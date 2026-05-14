# Phase 7: 流水线解耦 — I-Buffer + Operand Collector

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added an interactive pipeline stage debugger. Supports the `ib` command to inspect I-Buffer entries per warp, the `oc` command to view operand collector bank state and conflicts, and the `sb` command for scoreboard inspection.
- **Fixed SIMT stack reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the reconvergence check logic to properly detect when all divergent paths have completed execution.
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing with pipeline stage state dumps.
- **Bilingual comments and ASCII flow diagrams**: All `.asm` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

将 Phase 6 的简化流水线（fetch→execute 合一）升级为 GPGPU-Sim 风格的多级流水线：

```
原:  FETCH+EXECUTE (1 stage)
新:  FETCH → DECODE→I-Buffer → ISSUE(check bank) → EXECUTE → WRITEBACK
```

## 2. I-Buffer (per-warp 指令缓冲)

对标 GPGPU-Sim 的 I-Buffer：fetch 和 issue 解耦。

| 属性 | 说明 |
|------|------|
| 容量 | 每 warp 2 条指令槽 |
| 分区 | 静态分区（各 warp 独立） |
| valid 位 | fetch 写入后置 1 |
| ready 位 | decode + scoreboard check 后置 1 |
| 调度 | Scheduler 从 I-Buffer 中选 ready 的 warp |

## 3. Operand Collector (banked register file)

对标 GPGPU-Sim 的 `opndcoll_rfu_t`。

| 属性 | 说明 |
|------|------|
| Bank 数 | 4 banks |
| 寄存器→bank | bank_id = reg_id % 4 |
| 读端口 | 每 bank 1 个读端口/cycle |
| Bank conflict | rs1 和 rs2 同 bank → 需 2 cycles 串行读 |
| No conflict | rs1 和 rs2 不同 bank → 1 cycle 并行读 |

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | I-Buffer fetch→decode→issue 解耦正确 |
| AC-02 | Bank conflict 检测正确 (同 bank → +1 cycle) |
| AC-03 | No conflict 时操作数 1 cycle 收集 |
| AC-04 | Phase 0-6 测试保持通过 |
