; ============================================================
; Phase 13: Tiling Strategies — Tiled Matrix Multiply (2×2)
; 阶段 13：分块策略 — 分块矩阵乘法
;
; Purpose / 目的:
;   Demonstrate tiled matrix multiplication using shared memory.
;   A and B are loaded into shared memory tiles via TLDS,
;   then SHLD reads them for fast multiply-accumulate.
;   演示使用 shared memory 分块的矩阵乘法。TLDS 将 A 和 B
;   加载到 shared memory, SHLD 读取做乘加运算。
;
; Algorithm / 算法 (2×2 matmul):
;   C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j]
;   Tile: A[2×2] in smem[0..3], B[2×2] in smem[4..7]
;
; Setup (by test harness, warp_size=1):
;   A = [[1,2],[3,4]]  at mem[0..3]
;   B = [[5,6],[7,8]]  at mem[8..11]
;
; Expected Result / 预期结果:
;   mem[16] = 1*5 + 2*7 = 19  (C[0][0])
;   mem[17] = 1*6 + 2*8 = 22  (C[0][1])
;   mem[18] = 3*5 + 4*7 = 43  (C[1][0])
;   mem[19] = 3*6 + 4*8 = 50  (C[1][1])
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TLCONF 2, 2, 2    ← tile config: M=2,N=2,K=2
;     |
;     v
;   TLDS 0, 0         ← load A tile → shared mem[0..3]
;   TLDS 4, 8         ← load B tile → shared mem[4..7]
;     |
;     v
;   SHLD + MUL + ADD  ← compute C element by element
;   (4 iterations for 2×2 output)    using shared memory reads
;     |
;     v
;   ST results → global memory[16..19]
;     |
;     v
;   HALT
; ============================================================

; Configure tile: 2×2 output tile, K=2 inner dimension
TLCONF 2, 2, 2           ; tile_M=2, tile_N=2, tile_K=2

; Load A[2×2] from global mem[0..3] to shared mem[0..3]
TLDS 0, 0                ; smem[0]=A[0]=1, smem[1]=A[1]=2, smem[2]=A[2]=3, smem[3]=A[3]=4

; Load B[2×2] from global mem[8..11] to shared mem[4..7]
TLDS 4, 8                ; smem[4]=B[0]=5, smem[5]=B[1]=6, smem[6]=B[2]=7, smem[7]=B[3]=8

; === Compute C[0][0] = A[0]*B[0] + A[1]*B[2] = 1*5 + 2*7 = 19 ===
SHLD r10, 0              ; r10 = smem[0] = A[0][0] = 1
SHLD r11, 4              ; r11 = smem[4] = B[0][0] = 5
MUL r12, r10, r11        ; r12 = 1 * 5 = 5
SHLD r10, 1              ; r10 = smem[1] = A[0][1] = 2
SHLD r11, 6              ; r11 = smem[6] = B[1][0] = 7
MUL r13, r10, r11        ; r13 = 2 * 7 = 14
ADD r14, r12, r13        ; r14 = 5 + 14 = 19
ST r14, [16]             ; mem[16] = C[0][0] = 19

; === Compute C[0][1] = A[0]*B[1] + A[1]*B[3] = 1*6 + 2*8 = 22 ===
SHLD r10, 0              ; r10 = A[0][0] = 1
SHLD r11, 5              ; r11 = smem[5] = B[0][1] = 6
MUL r12, r10, r11        ; r12 = 6
SHLD r10, 1              ; r10 = A[0][1] = 2
SHLD r11, 7              ; r11 = smem[7] = B[1][1] = 8
MUL r13, r10, r11        ; r13 = 16
ADD r14, r12, r13        ; r14 = 22
ST r14, [17]             ; mem[17] = C[0][1] = 22

; === Compute C[1][0] = A[2]*B[0] + A[3]*B[2] = 3*5 + 4*7 = 43 ===
SHLD r10, 2              ; r10 = smem[2] = A[1][0] = 3
SHLD r11, 4              ; r11 = smem[4] = B[0][0] = 5
MUL r12, r10, r11        ; r12 = 15
SHLD r10, 3              ; r10 = smem[3] = A[1][1] = 4
SHLD r11, 6              ; r11 = smem[6] = B[1][0] = 7
MUL r13, r10, r11        ; r13 = 28
ADD r14, r12, r13        ; r14 = 43
ST r14, [18]             ; mem[18] = C[1][0] = 43

; === Compute C[1][1] = A[2]*B[1] + A[3]*B[3] = 3*6 + 4*8 = 50 ===
SHLD r10, 2              ; r10 = A[1][0] = 3
SHLD r11, 5              ; r11 = B[0][1] = 6
MUL r12, r10, r11        ; r12 = 18
SHLD r10, 3              ; r10 = A[1][1] = 4
SHLD r11, 7              ; r11 = B[1][1] = 8
MUL r13, r10, r11        ; r13 = 32
ADD r14, r12, r13        ; r14 = 50
ST r14, [19]             ; mem[19] = C[1][1] = 50

HALT
