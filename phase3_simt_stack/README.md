## Quick Start

Interactive debugging with learning_console.py (branch divergence and SIMT Stack display):

```bash
# Interactive single-step debugging with divergence tracking
python src/learning_console.py tests/programs/03_divergence.asm --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/03_divergence.asm --trace --warp-size 4

# Predication demo
python src/learning_console.py tests/programs/06_predication.asm --warp-size 4

# Nested divergence demo
python src/learning_console.py tests/programs/04_nested.asm --warp-size 4

# Warp register demo
python src/learning_console.py tests/programs/08_warp_regs.asm --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive SIMT Stack debugger with divergence tracking, push/pop detection, reconvergence display
- **Interactive commands**: `Enter`=step, `r`=run, `i`=state, `m`=memory, `reg`=registers, `wreg`=warp regs, `stack`=SIMT Stack, `q`=quit
- **PRED/Predication**: `@p0` predicate flag (bit 31 of instruction word), `SETP` instruction, per-thread `pred` register for branch-free conditional execution
- **Warp-level registers**: WREAD/WWRITE instructions with warp-uniform registers (`wid`, `ntid`)
- **Trace mode**: `--trace` for batch execution with cycle-by-cycle trace output
- **Branch divergence visualization**: Taken/not-taken mask display, stack depth tracking, reconvergence point detection

# Phase 3: SIMT Stack (分支发散)

实现 SIMT Stack 处理 warp 内分支发散与重汇聚。对标 GPGPU-Sim `simt_stack`。

## 新增概念
- JMP/BEQ/BNE 分支指令
- Two-pass 汇编器 (label 解析)
- SIMT Stack: push(发散) → pop(重汇聚)
- Active mask 分拆与恢复
- JMP merge 模式

## 运行

```bash
cd phase3_simt_stack && bash run.sh
```

## 测试
32 项测试，含 if/else 偶数/奇数线程发散、tid 分组分支。

## 对标 GPGPU-Sim
`simt_stack` (IPDOM 后支配栈)
