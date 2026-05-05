# Phase 1: SIMD 向量处理器 — 需求分解

## 1. 目标

将 Phase 0 的标量处理器扩展为 **SIMD（Single Instruction, Multiple Data）向量处理器**。

对标 GPGPU-Sim 中 `simd_function_unit` 的多 lane 并行执行模型。

## 2. 核心概念

### SIMD vs 标量

```
标量: ADD r3, r1, r2   →   r3[0] = r1[0] + r2[0]

SIMD:  VADD v3, v1, v2  →   v3[0] = v1[0] + v2[0]
                             v3[1] = v1[1] + v2[1]
                             v3[2] = v1[2] + v2[2]
                             ... (LANES 路并行)
```

### 对标 GPGPU-Sim

GPGPU-Sim 中 warp 内 32 个线程执行同一条指令 = SIMD 宽度为 32。Phase 1 的 SIMD = 把 warp 简化为纯向量（无线程 ID、无分支发散）。

## 3. 功能需求

### FR-01: 向量寄存器堆
- VLEN 路向量寄存器，每路 32-bit
- 8 个向量寄存器 (v0-v7)，每个 VLEN×32bit
- 支持单周期全 lane 读写

### FR-02: 向量 ALU
- 支持 VLEN 路并行的 ADD/SUB/MUL/DIV
- 每路独立运算，无 lane 间通信
- 运算语义与 Phase 0 ALU 一致

### FR-03: 向量指令

| 指令 | 格式 | 功能 |
|------|------|------|
| VADD | VADD vd, vs1, vs2 | vd[i] = vs1[i] + vs2[i] (i=0..VLEN-1) |
| VSUB | VSUB vd, vs1, vs2 | vd[i] = vs1[i] - vs2[i] |
| VMUL | VMUL vd, vs1, vs2 | vd[i] = vs1[i] * vs2[i] |
| VDIV | VDIV vd, vs1, vs2 | vd[i] = vs1[i] / vs2[i] |
| VLD  | VLD vd, addr | vd[i] = mem[addr + i] (连续加载) |
| VST  | VST vs, addr | mem[addr + i] = vs[i] (连续存储) |
| VMOV | VMOV vd, imm | vd[i] = imm (广播到所有 lane) |

### FR-04: 标量/向量混合
- Phase 0 标量指令（ADD/SUB/...）继续有效
- 标量寄存器 r0-r15 用于地址计算
- 向量寄存器 v0-v7 用于数据并行运算

### FR-05: 配置参数
- VLEN（向量长度）默认为 8，可通过配置文件修改
- 内存扩展为 1024 words (4KB)

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | VADD/VSUB/VMUL/VDIV 全 lane 正确运算 |
| AC-02 | VLD/VST 连续访存正确 |
| AC-03 | VMOV 广播正确 |
| AC-04 | 标量+向量混合程序正确运行 |
| AC-05 | 向量加法 kernel（C[i]=A[i]+B[i]）正确 |
| AC-06 | Phase 0 全部测试仍然通过（向后兼容） |
| AC-07 | 一键运行通过所有测试 |
