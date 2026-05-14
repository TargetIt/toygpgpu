# Phase 1 实验报告

## 2026-05-15: Feature Additions

### vec4 复合类型操作
- 新增 vec4_demo 程序，展示如何在 SIMD 向量处理器上实现 vec4 复合类型
- vec4 加法：使用 VADD 指令对两个 vec4 的对应分量逐元素相加，VLEN=8 可同时处理两组 vec4
- vec4 点积：通过 VMUL（逐元素乘）和标量 ADD/ST 组合实现，模拟 GPU 着色器中常见的 dot product 模式
- vec4 标量乘法：VMOV 广播标量到所有 lane，然后 VMUL 执行逐元素乘法
- 关键发现：vec4 操作天然映射到 SIMD 架构，每个 lane 处理一个分量，免去标量循环开销
- 对比 Phase 0 标量实现：vec4 加法从 4 条标量指令 + 循环控制缩减为 1 条 VADD 指令

### learning_console.py 交互式调试体验
- Phase 1 调试器支持双寄存器堆查看：`regs` 同时显示标量寄存器 r0-r15 和向量寄存器 v0-v7
- 向量寄存器显示格式：每个向量寄存器展示所有 8 个 lane 的值，便于观察 SIMD 并行执行效果
- 向量指令单步调试：step 执行 VADD 时一次显示 8 个 lane 的 ALU 结果
- trace 模式增强：向量指令 trace 以表格形式展示 lane 0-7 的并行执行详情

### trace 输出分析
- 向量指令 trace 记录格式：
  - `Cycle`: 执行周期
  - `PC`: 指令地址
  - `Insn`: 向量指令助记符（如 VADD v0, v1, v2）
  - `VecALU[0-7]`: 每个 lane 的 ALU 运算输入和输出
  - `VecReg`: 向量寄存器写回详情（每个 lane 的目标值）
  - `VecMem`: VLD/VST 的连续内存访问模式
- 验证示例：VADD 执行后 trace 显示 v0[0-7] = v1[0-7] + v2[0-7]，所有 lane 结果正确
- 连续内存访问 trace 验证：VLD 从 base_addr 开始连续加载 8 个 word，地址模式为 base + lane_idx

---

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
