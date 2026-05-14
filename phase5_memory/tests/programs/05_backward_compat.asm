; ============================================================
; Phase 5: Memory Subsystem — Phase 0-4 Backward Compatibility
; 阶段 5：内存子系统 — Phase 0-4 向后兼容性测试
;
; Purpose / 目的:
;   Verify that legacy programs (scalar, SIMT, scoreboard)
;   still work correctly alongside the new memory subsystem.
;   验证旧版程序在新的内存子系统下仍能正确工作。
;
; Expected Result / 预期结果:
;   mem[0] = 13   (10 + 3)
;   mem[10+tid] = tid
;
; Key Concepts / 关键概念:
;   - Memory subsystem changes must not break existing programs
;   - Caches, shared memory, coalescing are transparent
;   - 内存子系统的更改不能破坏已有程序
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10
;   MOV r2, 3
;   ADD r3, r1, r2    ← r3 = 10 + 3 = 13
;   ST r3, [0]        ← mem[0] = 13
;     |
;     v
;   TID r4            ← r4 = tid
;   ST r4, [10]       ← mem[10+tid] = tid
;     |
;     v
;   HALT
; ============================================================

; --- Scalar arithmetic / 标量算术 ---
MOV r1, 10       ; r1 = 10
MOV r2, 3        ; r2 = 3
ADD r3, r1, r2   ; r3 = 10 + 3 = 13
ST r3, [0]       ; mem[0] = 13

; --- SIMT per-thread store / SIMT 多线程存储 ---
TID r4           ; r4 = tid
ST r4, [10]      ; mem[10 + tid] = tid

; 程序终止 / Terminate program
HALT
