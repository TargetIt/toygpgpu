## Quick Start

Interactive debugging with learning_console.py (SIMT-aware with per-thread register display):

```bash
# Interactive single-step debugging
python src/learning_console.py tests/programs/01_tid_wid.asm

# Batch trace mode
python src/learning_console.py tests/programs/01_tid_wid.asm --trace

# Multi-warp demo
python src/learning_console.py tests/programs/05_multi_warp.asm --num-warps 2 --warp-size 4

# Thread vector add demo
python src/learning_console.py tests/programs/04_vector_add_mt.asm --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive SIMT debugger with per-thread register display, warp state summary, barrier tracking
- **Interactive commands**: `Enter`=step, `r`=run, `i`=info, `m`=memory, `reg`=per-thread registers, `q`=quit
- **Warp-level registers**: WREAD/WWRITE instructions for accessing warp-uniform registers (`wid`, `ntid`) shared by all threads in a warp
- **Trace mode**: `--trace` flag for batch execution with warp-level tracking
- **Per-warp state inspection**: Active mask, PC, barrier status in state display

# Phase 2: SIMT 核心 (Warp/Thread)

引入 SIMT 执行模型——多线程共享 PC 的 Warp 概念。对标 GPGPU-Sim 的 `shd_warp_t`。

## 新增概念
- Thread: 每线程独立寄存器堆
- Warp: 一组线程共享 PC + active mask
- Warp Scheduler: Round-Robin 调度
- TID/WID/BAR 指令
- base+thread_id 访存模式

## 运行

```bash
cd phase2_simt && bash run.sh
```

## 测试
43 项测试，含多线程向量加法、barrier 同步、多 warp 并发。

## 对标 GPGPU-Sim
`shd_warp_t` + `scheduler_unit` + `shader_core_ctx`
