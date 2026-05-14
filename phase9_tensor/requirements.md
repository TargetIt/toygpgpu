# Phase 9: Tensor Core — MMA 指令

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added a tensor-aware interactive debugger. Supports the `mma` command to inspect MMA (matrix multiply-accumulate) instruction operands and results, including packed 16-bit element views.
- **Fixed IBuffer peek and reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the SIMT stack reconvergence check logic (same fix as Phase 7).
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing.
- **Bilingual comments and ASCII flow diagrams**: All `.asm` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

实现简化版 Tensor Core 的矩阵乘累加 (MMA) 指令。

对标 NVIDIA Tensor Core 的 HMMA (Half-precision Matrix Multiply-Accumulate)。

## 2. MMA 指令

| 指令 | 格式 | 功能 |
|------|------|------|
| MMA | `MMA rd, rs1, rs2, rs3` | rd = sum(rs1[i]*rs2[i]) + rs3 |

- rs1: packed [a1:16][a0:16] — 两个有符号 16-bit
- rs2: packed [b1:16][b0:16] — 两个有符号 16-bit
- rs3: 32-bit 累加器
- rd: 32-bit 结果 = a0*b0 + a1*b1 + rs3

## 3. 2x2 矩阵乘

使用 4 条 MMA 指令完成 D = A×B + C:
```
A = [[a00, a01],   B = [[b00, b01],   C = [[c00, c01],
     [a10, a11]]        [b10, b11]]        [c10, c11]]
```
每条 MMA 计算一个输出元素。

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | MMA 单指令正确 (带符号扩展) |
| AC-02 | 2×2 矩阵乘结果正确 |
| AC-03 | Phase 0-8 测试保持通过 |
