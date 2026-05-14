; ============================================================
; Phase 2: SIMT Core — Barrier Synchronization
; 阶段 2：SIMT 核心 — 屏障同步
;
; Purpose / 目的:
;   Demonstrate the BAR (barrier) instruction. All threads
;   in a warp complete phase 1 before any thread starts
;   phase 2. This ensures cross-thread ordering.
;   演示 BAR（屏障）指令。一个 warp 中的所有线程完成
;   阶段 1 后，才能有线程开始阶段 2。确保线程间顺序。
;
; Expected Result / 预期结果:
;   Phase 1 — mem[300..307] = [0, 10, 20, 30, 40, 50, 60, 70]
;   Phase 2 — mem[310..317] = [10, 20, 30, 40, 50, 60, 70, 80]
;   (每个线程: phase1 = tid*10, phase2 = tid*10 + 10)
;
; Key Concepts / 关键概念:
;   - BAR (barrier): hardware synchronization point for all threads
;   - Guarantees all previous writes are visible after barrier
;   - Useful when threads need to share data via memory
;   - 屏障保证 barrier 之前的所有写操作在 barrier 之后对
;     其他线程可见
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1          ← r1 = tid
;   MOV r2, 10      ← r2 = 10
;     |
;     v
;   MUL r3, r1, r2  ← r3 = tid * 10
;   ST r3, [300]    ← mem[300+tid] = tid*10  (阶段 1)
;     |
;     v
;   ================
;   BAR             ← 屏障: 等待所有线程到达此处
;   ================
;     |                (所有线程的阶段 1 写操作此时均已可见)
;     v
;   ADD r3, r3, r2  ← r3 = tid*10 + 10  (阶段 2)
;   ST r3, [310]    ← mem[310+tid] = r3
;     |
;     v
;   HALT
; ============================================================

; --- Phase 1: Compute pre-barrier values / 阶段 1：屏障前计算 ---
TID r1           ; r1 = tid
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = tid * 10
ST r3, [300]     ; mem[300 + tid] = tid * 10  (存储到 phase 1 区域)

; --- Barrier synchronization point / 屏障同步点 ---
; 所有线程在此等待。直到所有线程都到达 BAR 后，
; 任何线程才能继续执行阶段 2。
; All threads wait here. None proceeds to phase 2
; until every thread in the warp has reached BAR.
BAR

; --- Phase 2: Compute post-barrier values / 阶段 2：屏障后计算 ---
; 由于 BAR 的保证，mem[300..307] 中的值对所有线程可见
; Guaranteed: all mem[300..307] writes are visible to all threads
ADD r3, r3, r2   ; r3 = tid * 10 + 10
ST r3, [310]     ; mem[310 + tid] = tid * 10 + 10 (阶段 2 结果)

; 程序终止 / Terminate program
HALT
