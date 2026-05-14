# Phase 5: 内存层次 — 需求分解

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added an interactive debugger with memory hierarchy views. Supports the `cache` command to inspect L1 cache lines, tags, hit/miss counters, and the `smem` command to view shared memory contents per block.
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing with memory hierarchy state dumps.
- **Bilingual comments and ASCII flow diagrams**: All `.asm` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

从平坦内存升级为 **GPU 风格的多层内存体系**。

对标 GPGPU-Sim 的 Shared Memory、L1 Cache、Memory Coalescing、Thread Block。

## 2. 功能需求

### FR-01: Shared Memory
- 每个 Thread Block 有独立的 Shared Memory (256 words)
- `SHLD rd, addr` — 从 shared memory 加载
- `SHST rs, addr` — 存储到 shared memory
- 同 block 内所有 warp 共享

### FR-02: L1 Cache (简化)
- 直接映射 (direct-mapped)
- 16 lines × 4 words/line
- 提供 hit/miss 统计
- 自动由 LD/ST 经过

### FR-03: Memory Coalescing
- 同一 warp 内多个线程的 LD/ST 请求合并
- 连续地址访问合并为一次 transaction
- 不连续则串行处理

### FR-04: Thread Block (CTA)
- 多个 warp 组成一个 block
- 共享 shared memory
- 为未来的 grid kernel launch 做准备

### FR-05: 统计
- 打印 cache hit rate、coalescing efficiency

## 3. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | Shared memory 读写正确 |
| AC-02 | L1 Cache hit/miss 统计正确 |
| AC-03 | Coalescing 合并连续访问 |
| AC-04 | 多 warp 共享 shared memory |
| AC-05 | Phase 0-4 测试保持通过 |
