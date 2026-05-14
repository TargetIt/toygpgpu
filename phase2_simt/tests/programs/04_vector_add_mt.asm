; ============================================================
; Phase 2: SIMT Core — Multi-Thread Vector Addition
; 阶段 2：SIMT 核心 — 多线程向量加法
;
; Purpose / 目的:
;   Implement vector addition using SIMT threads rather
;   than SIMD vector registers. Each thread processes
;   one element: C[tid] = A[tid] + B[tid].
;   使用 SIMT 线程而非 SIMD 向量寄存器实现向量加法。
;   每个线程处理一个元素。
;
; This demonstrates the key advantage of SIMT over SIMD:
; each thread independently computes its element using
; scalar instructions, with no need for vector registers.
; 这展示了 SIMT 相对于 SIMD 的关键优势。
;
; Expected Result / 预期结果:
;   mem[16..23] = [11, 22, 33, 44, 55, 66, 77, 88]
;   (A[tid] + B[tid] for tid = 0..7)
;
; Key Concepts / 关键概念:
;   - SIMT parallelism: threads are the parallel unit, not vector lanes
;   - TID provides element index
;   - LD [base] actually loads from [base + tid] for each thread
;   - 每个线程使用标量 LD/ADD/ST 指令，效果等同于向量操作
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   初始化 A: mem[0..7] = [10, 20, 30, 40, 50, 60, 70, 80]
;   初始化 B: mem[8..15] = [1, 2, 3, 4, 5, 6, 7, 8]
;     |
;     v
;   每个线程并行执行:
;   TID r1           ← r1 = tid
;   LD r2, [0]       ← r2 = A[tid]  (mem[0+tid])
;   LD r3, [8]       ← r3 = B[tid]  (mem[8+tid])
;     |
;     v
;   ADD r4, r2, r3   ← r4 = A[tid] + B[tid]
;   ST r4, [16]      ← mem[16+tid] = C[tid]
;     |
;     v
;   HALT
;
; Thread mapping / 线程映射:
;   Thread 0: C[0] = A[0] + B[0] = 10 + 1 = 11
;   Thread 1: C[1] = A[1] + B[1] = 20 + 2 = 22
;   ...
;   Thread 7: C[7] = A[7] + B[7] = 80 + 8 = 88
; ============================================================

; === Step 1: Initialize vector A in memory / 初始化向量 A ===
; 向量 A: [10, 20, 30, 40, 50, 60, 70, 80]
MOV r1, 10
ST r1, [0]
MOV r1, 20
ST r1, [1]
MOV r1, 30
ST r1, [2]
MOV r1, 40
ST r1, [3]
MOV r1, 50
ST r1, [4]
MOV r1, 60
ST r1, [5]
MOV r1, 70
ST r1, [6]
MOV r1, 80
ST r1, [7]

; === Step 2: Initialize vector B in memory / 初始化向量 B ===
; 向量 B: [1, 2, 3, 4, 5, 6, 7, 8]
MOV r1, 1
ST r1, [8]
MOV r1, 2
ST r1, [9]
MOV r1, 3
ST r1, [10]
MOV r1, 4
ST r1, [11]
MOV r1, 5
ST r1, [12]
MOV r1, 6
ST r1, [13]
MOV r1, 7
ST r1, [14]
MOV r1, 8
ST r1, [15]

; === Step 3: Per-thread vector addition / 每个线程做向量加法 ===
; 每个线程使用自己的 tid 来索引数组元素
; Each thread uses its TID to index array elements

TID r1           ; r1 = tid (当前线程索引 / current thread index)

; 加载 A[tid] 和 B[tid] / Load A[tid] and B[tid]
; LD 的地址是基址，实际加载地址为 base + tid
; LD uses base address, actual address = base + tid
LD r2, [0]       ; r2 = mem[0 + tid] = A[tid]
LD r3, [8]       ; r3 = mem[8 + tid] = B[tid]

; 计算 C[tid] = A[tid] + B[tid]
ADD r4, r2, r3   ; r4 = A[tid] + B[tid]

; 存储结果 / Store result
ST r4, [16]      ; mem[16 + tid] = C[tid]

; 程序终止 / Terminate program
HALT
