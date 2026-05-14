; ============================================================
; Phase 3: SIMT Stack Core — Phase 0/1/2 Backward Compatibility
; 阶段 3：SIMT 栈核心 — Phase 0/1/2 向后兼容性测试
;
; Purpose / 目的:
;   Verify that scalar, vector, and SIMT instructions
;   from previous phases still work correctly in the
;   SIMT stack pipeline.
;   验证前几个阶段的标量、向量和 SIMT 指令在此
;   SIMT 栈流水线中仍然正确工作。
;
; This tests: scalar add, TID, ST, without any branches.
; 测试内容：标量加法、TID、ST，不涉及分支。
;
; Expected Result / 预期结果:
;   mem[0] = 13        (10 + 3)
;   mem[10+tid] = tid  (每个线程写自己的 tid)
;
; Key Concepts / 关键概念:
;   - Backward compatibility ensures architectural stability
;   - No branches means no SIMT stack activity needed
;   - Legacy programs work unchanged
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10
;   MOV r2, 3
;   ADD r3, r1, r2    ← r3 = 13  (标量加法)
;   ST r3, [0]        ← mem[0] = 13
;     |
;     v
;   TID r4            ← r4 = tid (多线程部分)
;   ST r4, [10]       ← mem[10+tid] = tid
;     |
;     v
;   HALT
; ============================================================

; --- Scalar arithmetic (Phase 0 style / 标量算术) ---
MOV r1, 10       ; r1 = 10
MOV r2, 3        ; r2 = 3
ADD r3, r1, r2   ; r3 = 10 + 3 = 13
ST r3, [0]       ; mem[0] = 13

; --- SIMT per-thread store (Phase 2 style / 多线程存储) ---
TID r4           ; r4 = tid
ST r4, [10]      ; mem[10 + tid] = tid

; 程序终止 / Terminate program
HALT
