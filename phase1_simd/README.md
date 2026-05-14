## Quick Start

Interactive debugging with learning_console.py (SIMD-aware with vector register display):

```bash
# Interactive single-step debugging
python src/learning_console.py tests/programs/01_vector_add.asm

# Batch trace mode
python src/learning_console.py tests/programs/01_vector_add.asm --trace

# Explore vec4 composite data type demo
python src/learning_console.py tests/programs/06_vec4_demo.asm

# Run with custom max cycles
python src/learning_console.py tests/programs/02_vector_mul.asm --max-cycles 300
```

## New in this update

- **learning_console.py**: Interactive SIMD vector debugger with vector register inspection (`vreg` command), per-lane change tracking
- **Interactive commands**: `Enter`=step, `r`=run, `i`=info, `m`=memory, `reg`=scalar registers, `vreg`=vector registers, `q`=quit
- **vec4/float4 composite data type**: V4PACK, V4ADD, V4MUL, V4UNPACK instructions for 4x8-bit SIMD within a register (SWAR)
- **Vec4ALU module**: Dedicated ALU for pack/add/mul/unpack operations on 4-byte packed data
- **Trace mode**: `--trace` flag for batch vector execution with detailed cycle output

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
