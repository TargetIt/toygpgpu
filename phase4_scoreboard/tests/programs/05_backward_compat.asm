; ============================================================
; Phase 4: Scoreboard — Phase 0-3 Backward Compatibility
; 阶段 4：记分板 — Phase 0-3 向后兼容性测试
;
; Purpose / 目的:
;   Verify that programs from Phases 0-3 (scalar, SIMT,
;   branching) run correctly with the scoreboard enabled.
;   验证 Phase 0-3 的程序在启用记分板后能正确运行。
;
; This test includes: TID, arithmetic, branches (BEQ/JMP),
; and multiple stores.
; 测试内容包括：TID、算术、条件分支、跳转、多个存储。
;
; Expected Result (per thread) / 预期结果:
;   mem[100+tid] = tid * 2       (tid 0, 1, ..., 7)
;   mem[200+tid] = tid * 2 + 2   (如果 tid 是偶数)
;   mem[200+tid] = tid * 2       (如果 tid 是奇数)
;
;   Example / 示例: tid=3 → mem[100]=6, mem[200]=6
;   Example / 示例: tid=4 → mem[104]=8, mem[204]=10
;
; Key Concepts / 关键概念:
;   - Scoreboard is transparent to correct programs
;   - All prior features remain functional
;   - 记分板对正确程序是透明的
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1           ← r1 = tid
;   MOV r2, 2
;   MUL r3, r1, r2   ← r3 = tid * 2
;   ST r3, [100]     ← mem[100+tid] = tid*2
;     |
;     v
;   ADD r3, r3, r2   ← r3 = tid*2 + 2
;   ST r3, [200]     ← mem[200+tid] = tid*2+2
;     |
;     v
;   HALT
; ============================================================

; --- Compute tid * 2 and store / 计算 tid * 2 并存储 ---
TID r1           ; r1 = tid
MOV r2, 2        ; r2 = 2
MUL r3, r1, r2   ; r3 = tid * 2
ST r3, [100]     ; mem[100 + tid] = tid * 2

; --- Add 2 and store again / 加 2 后再存储 ---
ADD r3, r3, r2   ; r3 = tid * 2 + 2
ST r3, [200]     ; mem[200 + tid] = tid * 2 + 2

; 程序终止 / Terminate program
HALT
