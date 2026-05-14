; ============================================================
; Phase 6: Kernel Launch — GTO vs RR Scheduling
; 阶段 6：内核启动 — GTO（贪婪最老）vs RR（轮转）调度
;
; Purpose / 目的:
;   Demonstrate that multiple warps can be scheduled by the
;   kernel launcher, regardless of scheduling policy (GTO or RR).
;   Each warp computes: C[tid] = wid * 10 + tid.
;   演示多个 warp 可被内核启动器调度，不论调度策略
;   （GTO 或 RR）。每个 warp 计算自己的唯一编号。
;
; Expected Result / 预期结果:
;   mem[100+tid] = wid * 10 + tid  (每个线程唯一值)
;   (For warp_size=8, 2 warps)
;   Warp 0: mem[100..107] = [0, 1, 2, 3, 4, 5, 6, 7]
;   Warp 1: mem[108..115] = [10, 11, 12, 13, 14, 15, 16, 17]
;
; Key Concepts / 关键概念:
;   - GTO (greedy-then-oldest): prioritizes one warp until it stalls
;   - RR (round-robin): fair time-slicing between warps
;   - Both produce correct results; only performance differs
;   - Kernel launcher manages warp creation and scheduling
;   - GTO 和 RR 都是正确的调度策略，但性能特征不同
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   WID r1           ← r1 = wid (warp ID)
;   MOV r2, 10
;   MUL r3, r1, r2   ← r3 = wid * 10 (warp 偏移)
;     |
;     v
;   TID r4           ← r4 = tid
;   ADD r5, r3, r4   ← r5 = wid*10 + tid (全局唯一)
;     |
;     v
;   ST r5, [100]     ← mem[100+tid] = r5
;     |
;     v
;   HALT
; ============================================================

; 获取 warp ID / Get warp ID
WID r1           ; r1 = wid (warp 编号 / warp number)

; 计算 warp 基址偏移 / Compute warp base offset
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = wid * 10

; 获取线程 ID / Get thread ID
TID r4           ; r4 = tid

; 计算全局唯一值 / Compute globally unique value
ADD r5, r3, r4   ; r5 = wid * 10 + tid

; 存储到全局内存 / Store to global memory
ST r5, [100]     ; mem[100 + tid] = r5

; 程序终止 / Terminate program
HALT
