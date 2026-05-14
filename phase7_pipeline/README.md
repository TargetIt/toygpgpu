## Quick Start

Interactive debugging with learning_console.py (pipeline stage and I-Buffer display):

```bash
# Interactive debugging with I-Buffer visualization
python src/learning_console.py tests/programs/01_ibuffer_basic.asm --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_ibuffer_basic.asm --trace --warp-size 4

# Bank conflict demo
python src/learning_console.py tests/programs/02_bank_conflict.asm --warp-size 4

# Branch I-Buffer flush demo
python src/learning_console.py tests/programs/03_branch_ibuffer.asm --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive pipeline debugger with I-Buffer slot state display, pipeline stage visualization (fetch/decode/issue/exec/wb)
- **Interactive commands**: `Enter`=step, `r`=run, `i`=state, `m`=memory, `reg`=registers, `ib`=I-Buffer, `sb`=scoreboard, `stack`=SIMT Stack, `q`=quit
- **PRED/Predication**: `@p0` predicate support through the pipeline, proper I-Buffer handling of predicated instructions
- **vec4/float4 instructions**: V4PACK, V4ADD, V4MUL, V4UNPACK support in pipeline execution
- **Warp-level registers**: WREAD/WWRITE for accessing warp-uniform registers
- **Trace mode**: `--trace` for batch execution with pipeline stage tracking

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
