# Phase 5 设计: 内存层次

## L1 Cache
- 直接映射, 16 lines × 4 words/line (64 words = 256 bytes)
- tag = addr // (num_lines * line_size)
- index = (addr / line_size) % num_lines
- Write-through, no-write-allocate
- hit/miss 统计

## Coalescing
- 同一 warp 内所有 active 线程的 LD/ST 地址连续 → 合并为 1 次 transaction
- `_is_contiguous()` 检查地址是否 addrs[i] == addrs[i-1]+1
- 统计 coalesce_count / total_mem_reqs

## Shared Memory
- per-CTA, 256 words
- SHLD/SHST 指令: opcode 0x31/0x32
- 同 block 所有 warp 共享
