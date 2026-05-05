# Phase 1 实验报告

## 1. 实验概述

**目标**: 将 Phase 0 标量处理器扩展为 SIMD 向量处理器。

**对标**: GPGPU-Sim 中 `simd_function_unit` 的多 lane 并行执行。

## 2. 新增代码

| 文件 | 行数 | 说明 |
|------|------|------|
| vector_register_file.py | 80 | VLEN×8 向量寄存器堆 |
| vector_alu.py | 40 | VLEN 路并行 ALU |
| isa.py (扩展) | +10 | 7 条向量操作码 |
| cpu.py (重写) | 170 | 标量+向量双通路 |
| assembler.py (扩展) | +25 | 向量指令解析 |
| 5 个测试程序 | 150 | 向量运算 + 兼容性 |

## 3. 架构演进

Phase 0 → Phase 1 的关键变化：
1. **双寄存器堆**：标量寄存器 (r0-15) 用于地址/控制，向量寄存器 (v0-7) 用于数据并行
2. **双 ALU**：标量 ALU + VLEN 路向量 ALU
3. **新增 7 条向量指令**：VADD/VSUB/VMUL/VDIV/VLD/VST/VMOV
4. **内存扩展**：256→1024 words

## 4. 与 GPGPU-Sim 的对应

| GPGPU-Sim | Phase 1 |
|-----------|---------|
| warp 调度 (32 threads) | 简化为 VLEN=8 固定宽度 SIMD |
| simd_function_unit SP 管线 | VectorALU 并行 lane 执行 |
| 标量寄存器 (地址计算) | RegisterFile r0-r15 |
| 向量寄存器 (数据) | VectorRegisterFile v0-v7 |
| 连续内存访问 (coalesced) | VLD/VST 连续地址模式 |

## 5. 关键性能数据

| 操作 | Phase 0 (标量) | Phase 1 (SIMD) | 加速比 |
|------|---------------|---------------|--------|
| 8 对加法 | 8×4=32 条指令 | 4 条指令 | **8×** |
| 8 对乘加 | 8×6=48 条指令 | 5 条指令 | **~10×** |

## 6. 已知限制

1. 无 lane 掩码（所有 lane 始终活跃）
2. 无跨 lane 操作（shuffle/reduce）
3. 无 SVV (标量-向量-向量) 指令格式
4. VMOV 只能广播立即数，不能从标量寄存器取值

## 7. 下一步

Phase 2: SIMT 核心 — 引入 Warp
- 将向量 lane 改为"线程"（thread）
- 每线程独立 PC（但同 warp 内共享）
- 线程 ID 概念（threadIdx）
- 为分支发散/重汇聚（Phase 3）做准备
