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
