# Phase 1: SIMD 向量处理器

从标量扩展为 VLEN 路 SIMD 向量处理器，对标 GPGPU-Sim 多 lane 并行。

## 新增概念
- 向量寄存器堆 (8×VLEN×32bit)
- 向量 ALU (VLEN 路并行)
- 向量指令: VADD/VSUB/VMUL/VDIV/VLD/VST/VMOV
- 标量+向量混合编程

## 运行

```bash
cd phase1_simd && bash run.sh
```

## 测试
49 项测试，含向量加法 kernel (4 条指令处理 8 对数据，~10× 加速)。

## 对标 GPGPU-Sim
`simd_function_unit` (多 lane 并行执行)
