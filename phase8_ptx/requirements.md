# Phase 8: PTX Frontend — 需求

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added a PTX-aware interactive debugger. Supports the `ptx` command to display the original PTX source line corresponding to the current instruction, enabling source-level debugging.
- **Fixed IBuffer peek and reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the SIMT stack reconvergence check logic (same fix as Phase 7).
- **Fixed PTX parser inline `;` comment handling in tokenize()**: Updated the `tokenize()` function to correctly strip inline comments starting with `;` in PTX source lines, preventing parse errors on real-world PTX output.
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing.
- **Bilingual comments and ASCII flow diagrams**: All `.ptx` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

实现一个简化的 **PTX (Parallel Thread Execution)** 解析器，
将真实 CUDA PTX 子集翻译为 toygpgpu 内部 ISA。

对标 GPGPU-Sim 的 `cuda-sim/ptx_parser`。

## 2. 支持的 PTX 子集

| PTX 指令 | 内部映射 |
|----------|---------|
| mov.u32 %r, imm | MOV r, imm |
| mov.u32 %r, %tid.x | TID r |
| add.u32 %r, %a, %b | ADD r, a, b |
| mul.lo.u32 %r, %a, %b | MUL r, a, b |
| ld.global.u32 %r, [%a] | LD r, [a] |
| st.global.u32 [%a], %r | ST r, [a] |
| bra label | JMP label |
| @p bra label | BEQ p, 1, label |
| ret | HALT |

## 3. 寄存器分配
- PTX 虚拟寄存器 (%r0, %r1, ...) 映射到物理寄存器 (r1-r10)
- 线性扫描分配（简单贪心）

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | mov/add/mul/ld/st 正确翻译 |
| AC-02 | 寄存器分配正确（无冲突） |
| AC-03 | 多线程 kernel 正确运行 |
