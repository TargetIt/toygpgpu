; ============================================================
; Phase 7: Pipeline — Phase 0-6 Backward Compatibility
; 阶段 7：流水线 — Phase 0-6 向后兼容性测试
;
; Purpose / 目的:
;   Verify that all previous phase features work correctly
;   with the new I-Buffer pipeline stage.
;   验证前所有阶段的特性在新 I-Buffer 流水线下正确工作。
;
; Expected Result / 预期结果:
;   mem[0] = 20   (10 + 10)
;   mem[10+tid] = tid
;
; Key Concepts / 关键概念:
;   - I-Buffer pipeline must be backward compatible
;   - All instruction types work unchanged
;   - 新的流水线阶段对程序是透明的
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10
;   ADD r2, r1, r1    ← r2 = 20
;   ST r2, [0]        ← mem[0] = 20
;     |
;     v
;   TID r3            ← r3 = tid
;   ST r3, [10]       ← mem[10+tid] = tid
;     |
;     v
;   HALT
; ============================================================

; --- Scalar arithmetic / 标量算术 ---
MOV r1, 10       ; r1 = 10
ADD r2, r1, r1   ; r2 = 10 + 10 = 20
ST r2, [0]       ; mem[0] = 20

; --- SIMT per-thread store / SIMT 线程存储 ---
TID r3           ; r3 = tid
ST r3, [10]      ; mem[10 + tid] = tid

; 程序终止 / Terminate program
HALT
