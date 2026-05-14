## Quick Start

Interactive debugging with learning_console.py (memory hierarchy inspection):

```bash
# Interactive debugging with shared memory display
python src/learning_console.py tests/programs/01_shared_mem.asm --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_shared_mem.asm --trace --warp-size 4

# Memory coalescing demo
python src/learning_console.py tests/programs/02_coalescing.asm --warp-size 4

# Multi-warp shared memory demo
python src/learning_console.py tests/programs/04_multi_warp_shared.asm --num-warps 2 --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive memory hierarchy debugger with shared memory and cache inspection
- **PRED/Predication**: `@p0` predicate support for all memory instructions (LD, ST, SHLD, SHST)
- **vec4/float4 instructions**: V4PACK, V4ADD, V4MUL, V4UNPACK for packed data processing in memory operations
- **Warp-level registers**: WREAD/WWRITE for accessing warp-uniform data during memory transactions
- **Trace mode**: `--trace` for batch execution with memory access tracking

# Phase 5: 内存层次

GPU 风格多层内存体系。对标 GPGPU-Sim 的 `gpu-cache` + shared memory。

## 新增
- Shared Memory (片上 256 words)
- L1 Cache (直接映射, 16 lines × 4 words)
- Memory Coalescing (连续地址合并)
- Thread Block (CTA, 共享 shared memory)
- SHLD/SHST 指令

## 运行

```bash
cd phase5_memory && bash run.sh
```

## 对标 GPGPU-Sim
`gpu-cache` + Shared Memory + Memory Coalescing
