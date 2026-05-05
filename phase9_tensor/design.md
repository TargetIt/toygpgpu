# Phase 9 设计: Tensor Core MMA

## MMA 指令编码
- opcode: 0x41
- rd: destination [23:20]
- rs1: packed operand A [19:16]
- rs2: packed operand B [15:12]
- rs3: accumulator [11:8] (复用 imm 字段)

## 数据格式
- rs1 = [a1:16][a0:16] (lo in bits 15:0, hi in bits 31:16)
- 16-bit 有符号整数
- result = a0*b0 + a1*b1 + rs3

## 2×2 矩阵乘
```
D[0][0] = A_row0 · B_col0 + C
D[0][1] = A_row0 · B_col1 + C
D[1][0] = A_row1 · B_col0 + C
D[1][1] = A_row1 · B_col1 + C
```
4 条 MMA 指令完成。

## 限制
- 12-bit MOV 直接载入无法编码 32-bit 打包值
- 解决: 通过 memory pre-load (LD) 加载打包数据
