# Phase 3: SIMT Stack — 分支发散与重汇聚

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added an interactive debugger with SIMT stack divergence/reconvergence tracking. Supports the `stack` command to inspect the current SIMT stack state, including reconvergence PC, taken/fallthrough masks, and active path history.
- **PRED (predication) support**: Added the `OP_SETP` instruction (opcode 0x24) for setting per-thread predicate bits, `@p0` prefix syntax for conditional execution in assembly, and per-thread predication state tracking.
- **06_predication.asm**: Added a demo program comparing the PRED (predication) approach vs. the DPC (divergent PC) approach for handling warp-level conditional execution.
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for reading/writing warp-level uniform registers (shared across all threads in a warp). Added `warp_regs` storage in the Warp class.
- **08_warp_regs.asm**: Added a demo program showcasing warp-level uniform register read/write operations.
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing with SIMT stack state dumps.
- **Bilingual comments and ASCII flow diagrams**: All `.asm` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

引入 **SIMT Stack（后支配栈）**，正确处理 GPU 中 warp 内线程的分支发散（Branch Divergence）。

对标 GPGPU-Sim 的 `simt_stack` 类（基于 IPDOM / Immediate Post-Dominator）。

## 2. 核心问题

SIMT 中所有线程共享 PC。遇到 if/else 时，不同线程走不同分支：

```c
if (tid % 2 == 0)  // 偶数线程走 A, 奇数线程走 B
    A;
else
    B;
// ← 这里所有线程重新汇合 (reconvergence)
```

**方案**: SIMT Stack 记录发散状态，分步执行各分支路径，最后汇合。

## 3. 功能需求

### FR-01: 分支指令

| 指令 | 格式 | 功能 |
|------|------|------|
| JMP  | `JMP label` | 无条件跳转 |
| BEQ  | `BEQ rs1, rs2, label` | rs1 == rs2 时跳转 |
| BNE  | `BNE rs1, rs2, label` | rs1 != rs2 时跳转 |

### FR-02: Label 支持
- Assembler 支持 `label:` 语法
- 标签解析为 PC 地址
- 分支指令引用标签名

### FR-03: SIMT Stack
每个 warp 维护一个 SIMT Stack。栈条目包含：
- `reconv_pc`: 重汇聚地址
- `orig_mask`: 发散前活跃掩码
- `taken_mask`: 走跳转路径的线程掩码
- `fallthrough_pc`: 不走跳转的线程继续执行的 PC

### FR-04: 发散执行流程
1. 遇到分支 → 计算哪些线程跳转 (taken_mask)
2. 如果有线程不走跳转 → push {reconv=PC+1, orig=old_mask, taken=taken_mask, fall=PC+1}
3. 当前 warp 执行跳转路径（active_mask = taken_mask, PC = target）
4. 到达重汇聚点时 pop → 执行 fallthrough 路径（active_mask = orig_mask - taken_mask）
5. fallthrough 到达重汇聚点 → 恢复 orig_mask, PC = reconv_pc

### FR-05: 重汇聚检测
- 当 warp.pc == stack.top().reconv_pc 时触发重汇聚
- 如果还有未执行的路径 → 切换到那条路径
- 如果都执行完了 → 恢复发散前状态

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | 无条件 JMP 正确 |
| AC-02 | BEQ/BNE 按条件分支正确 |
| AC-03 | if-then-else 模式：偶数线程走 A，奇数线程走 B |
| AC-04 | 重汇聚后所有线程继续执行 |
| AC-05 | Phase 0/1/2 测试保持通过（向后兼容） |
