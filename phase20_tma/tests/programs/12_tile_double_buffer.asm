; ============================================================
; Phase 13: Tiling Strategies — Double Buffer Pattern
; 阶段 13：分块策略 — 双缓冲模式
;
; Purpose / 目的:
;   Demonstrate double buffering: while computing on tile buffer A,
;   preload next tile into buffer B. Then swap. This hides memory
;   latency by overlapping compute and data load.
;   演示双缓冲: 在 buffer A 上计算的同时, 预加载下一块数据到
;   buffer B, 然后交换。通过重叠计算和数据加载隐藏内存延迟。
;
; Scenario / 场景 (warp_size=1):
;   Process two tiles of data sequentially:
;   - Load Tile 0 → buf_A (smem[0..7])
;   - Load Tile 1 → buf_B (smem[8..15]) while computing Tile 0
;   - Compute Tile 0 from buf_A
;   - Compute Tile 1 from buf_B
;   - Store both results
;
; Expected Result / 预期结果:
;   mem[100] = sum of Tile 0 elements = 0+1+2+3+4+5+6+7 = 28
;   mem[101] = sum of Tile 1 elements = 8+9+...+15 = 92
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TLCONF 2, 4, 1    ← tile config
;     |
;     v
;   TLDS 0, 0         ← load Tile 0 → buf_A (smem[0..7])
;     |
;     +------+------+
;     |             |
;     v             v
;   [buffering]  TLDS 8, 8    ← load Tile 1 → buf_B (smem[8..15])
;     |             |
;     v             v
;   Compute       (done)
;   Tile 0        Tile 1
;   from buf_A    in buf_B
;     |             |
;     v             v
;   ST sum0→[100] ST sum1→[101]
;     |
;     v
;   HALT
; ============================================================

; Configure tile: 2×4 output, K=1
TLCONF 2, 4, 1           ; tile_M=2, tile_N=4, tile_K=1

; === Phase 1: Load Tile 0 into buf_A (smem offset 0) ===
TLDS 0, 0                ; smem[0..7] = mem[0..7] (Tile 0 data)

; === Phase 2: Load Tile 1 into buf_B (smem offset 8)
;    while we start computing Tile 0 ===
TLDS 8, 8                ; smem[8..15] = mem[8..15] (Tile 1 data)

; === Phase 3: Compute sum of Tile 0 from buf_A ===
SHLD r1, 0               ; r1 = smem[0] = tile0[0]
SHLD r2, 1               ; r2 = smem[1] = tile0[1]
ADD r1, r1, r2           ; r1 = tile0[0] + tile0[1]
SHLD r2, 2               ; r2 = tile0[2]
ADD r1, r1, r2           ; r1 += tile0[2]
SHLD r2, 3               ; r2 = tile0[3]
ADD r1, r1, r2           ; r1 += tile0[3]
SHLD r2, 4               ; r2 = tile0[4]
ADD r1, r1, r2           ; r1 += tile0[4]
SHLD r2, 5               ; r2 = tile0[5]
ADD r1, r1, r2           ; r1 += tile0[5]
SHLD r2, 6               ; r2 = tile0[6]
ADD r1, r1, r2           ; r1 += tile0[6]
SHLD r2, 7               ; r2 = tile0[7]
ADD r1, r1, r2           ; r1 = sum(tile0[0..7])
ST r1, [100]             ; mem[100] = sum of Tile 0

; === Phase 4: Compute sum of Tile 1 from buf_B ===
SHLD r3, 8               ; r3 = smem[8] = tile1[0]
SHLD r4, 9               ; r4 = smem[9] = tile1[1]
ADD r3, r3, r4           ; r3 = tile1[0] + tile1[1]
SHLD r4, 10              ; r4 = tile1[2]
ADD r3, r3, r4           ; r3 += tile1[2]
SHLD r4, 11              ; r4 = tile1[3]
ADD r3, r3, r4           ; r3 += tile1[3]
SHLD r4, 12              ; r4 = tile1[4]
ADD r3, r3, r4           ; r3 += tile1[4]
SHLD r4, 13              ; r4 = tile1[5]
ADD r3, r3, r4           ; r3 += tile1[5]
SHLD r4, 14              ; r4 = tile1[6]
ADD r3, r3, r4           ; r3 += tile1[6]
SHLD r4, 15              ; r4 = tile1[7]
ADD r3, r3, r4           ; r3 = sum(tile1[0..7])
ST r3, [101]             ; mem[101] = sum of Tile 1

HALT
