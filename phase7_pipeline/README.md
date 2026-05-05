# Phase 7: 流水线解耦 (I-Buffer + Operand Collector)

Fetch/Decode/Issue 多级流水线 + Banked Register File。

## 新增
- I-Buffer: per-warp 2 指令槽 (valid/ready), FIFO consume
- Operand Collector: 4-bank 寄存器堆, bank conflict 检测
- 流水线: advance→issue→fetch (逆序执行)
- I-Buffer 分支/重汇聚时 flush

## 运行

```bash
cd phase7_pipeline && bash run.sh
```

## 对标 GPGPU-Sim
`shader_core_ctx::fetch/decode/issue` + `opndcoll_rfu_t`
