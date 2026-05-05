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
