; ============================================================
; Phase 1: SIMD Vector Core — Phase 0 Backward Compatibility
; 阶段 1：SIMD 向量核心 — Phase 0 向后兼容性测试
;
; Purpose / 目的:
;   Verify that all scalar instructions from Phase 0 still
;   work correctly in the Phase 1 SIMD pipeline.
;   验证 Phase 0 的所有标量指令在 Phase 1 SIMD 流水线
;   中仍然正确工作。
;
; This tests: MOV, ADD, ST, LD, MUL, and r0 protection.
; 测试内容包括：MOV, ADD, ST, LD, MUL 以及 r0 保护。
;
; Expected Result / 预期结果:
;   mem[0] = 8     (5 + 3)
;   mem[1] = 42    (6 * 7)
;   mem[2] = 50    (8 + 42)
;   mem[3] = 50    (0 + 50, r0 保护检查)
;
; Key Concepts / 关键概念:
;   - Backward compatibility is critical for pipeline evolution
;   - All scalar instructions must remain functional
;   - r0 write protection must still work
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     +---> MOV r1, 5; MOV r2, 3
;     |     ADD r3, r1, r2    ← r3 = 8
;     |     ST r3, [0]        ← mem[0] = 8
;     |
;     +---> MOV r4, 6; MOV r5, 7
;     |     MUL r6, r4, r5    ← r6 = 42
;     |     ST r6, [1]        ← mem[1] = 42
;     |
;     +---> LD r7, [0]        ← r7 = 8
;     |     LD r8, [1]        ← r8 = 42
;     |     ADD r9, r7, r8    ← r9 = 50
;     |     ST r9, [2]        ← mem[2] = 50
;     |
;     +---> ADD r0, r9, r9    ← 尝试写入 r0 (被忽略)
;     |     ADD r10, r0, r9   ← r10 = 0 + 50 = 50
;     |     ST r10, [3]       ← mem[3] = 50
;     |
;     v
;   HALT
; ============================================================

; --- Scalar add / 标量加法 ---
MOV r1, 5        ; r1 = 5
MOV r2, 3        ; r2 = 3
ADD r3, r1, r2   ; r3 = 5 + 3 = 8
ST r3, [0]       ; mem[0] = 8

; --- Scalar multiply / 标量乘法 ---
MOV r4, 6        ; r4 = 6
MOV r5, 7        ; r5 = 7
MUL r6, r4, r5   ; r6 = 6 * 7 = 42
ST r6, [1]       ; mem[1] = 42

; --- Load and add / 加载并相加 ---
LD r7, [0]       ; r7 = mem[0] = 8
LD r8, [1]       ; r8 = mem[1] = 42
ADD r9, r7, r8   ; r9 = 8 + 42 = 50
ST r9, [2]       ; mem[2] = 50

; --- r0 protection check / r0 保护检查 ---
ADD r0, r9, r9   ; 尝试写入 r0 = 100 (应被硬件忽略 / should be ignored)
ADD r10, r0, r9  ; r10 = 0 + 50 = 50  (r0 始终为 0)
ST r10, [3]      ; mem[3] = 50

; 程序终止 / Terminate program
HALT
