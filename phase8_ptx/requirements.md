# Phase 8: PTX Frontend — 需求

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
