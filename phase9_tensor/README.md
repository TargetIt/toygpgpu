## Quick Start

Interactive debugging with learning_console.py (Tensor Core MMA display):

```bash
# Interactive debugging with MMA dot product
python src/learning_console.py tests/programs/01_mma_dot.asm --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_mma_dot.asm --trace --warp-size 4

# 2x2 matrix multiply demo
python src/learning_console.py tests/programs/02_matmul_2x2.asm --warp-size 4

# Negative test
python src/learning_console.py tests/programs/03_negative_mma.asm --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive Tensor Core debugger with MMA instruction display, packed operand visualization
- **PRED/Predication**: `@p0` predicate support for MMA instructions
- **vec4/float4 instructions**: V4PACK, V4ADD, V4MUL, V4UNPACK alongside MMA operations
- **Warp-level registers**: WREAD/WWRITE for warp-uniform register access
- **Trace mode**: `--trace` for batch execution with MMA operation tracking

# Phase 9: Tensor Core (MMA)

简化版 Tensor Core — 2 元素点积+累加指令。

## 新增
- `MMA rd, rs1, rs2, rs3` — rd = sum(rs1[i]*rs2[i]) + rs3
- rs1/rs2 打包 16-bit 有符号值 [hi:lo]
- 4 条 MMA 完成 2×2 矩阵乘 D=A×B+C

## 编码
rs3 编码在 imm[11:8] 字段，突破标准 3 操作数限制。

## 运行

```bash
cd phase9_tensor && bash run.sh
```

## 对标 GPGPU-Sim
`Tensor Core HMMA` (Half-precision Matrix Multiply-Accumulate)
