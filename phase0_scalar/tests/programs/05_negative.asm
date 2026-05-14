; ============================================================
; Phase 0: Scalar Core — Negative Number Arithmetic
; 阶段 0：标量核心 — 负数算术运算
;
; Purpose / 目的:
;   Demonstrate that the scalar pipeline handles signed
;   integers (two's complement) correctly for MUL, ADD,
;   and DIV operations.
;   演示标量流水线正确处理有符号整数（二进制补码）
;   的乘法、加法和除法运算。
;
; Expected Result / 预期结果:
;   mem[0] = -30   (0xFFFFFFE2)  —  (-10) * 3
;   mem[1] = -15   (0xFFFFFFF1)  —  (-10) + (-5)
;   mem[2] = -5    (0xFFFFFFFB)  —  (-20) / 4
;
; Key Concepts / 关键概念:
;   - Negative immediate values are supported in MOV
;   - MUL, ADD, DIV handle signed integers
;   - Two's complement representation for negatives
;   - DIV truncates toward zero
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, -10     ← r1 = -10  (0xFFFFFFF6)
;   MOV r2, 3       ← r2 = 3
;     |
;     v
;   MUL r3, r1, r2  ← r3 = -10 * 3 = -30
;     |
;     v
;   ST r3, [0]      ← mem[0] = -30  (0xFFFFFFE2)
;     |
;     v
;   MOV r4, -10     ← r4 = -10
;   MOV r5, -5      ← r5 = -5
;     |
;     v
;   ADD r6, r4, r5  ← r6 = -10 + (-5) = -15
;     |
;     v
;   ST r6, [1]      ← mem[1] = -15
;     |
;     v
;   MOV r7, -20     ← r7 = -20
;   MOV r8, 4       ← r8 = 4
;     |
;     v
;   DIV r9, r7, r8  ← r9 = -20 / 4 = -5
;     |
;     v
;   ST r9, [2]      ← mem[2] = -5
;     |
;     v
;   HALT
; ============================================================

; === Test 1: Multiplication with negative operand / 负数乘法 ===
MOV r1, -10      ; r1 = -10  (二进制补码: 0xFFFFFFF6)
MOV r2, 3        ; r2 = 3
MUL r3, r1, r2   ; r3 = -10 * 3 = -30  (0xFFFFFFE2)
ST r3, [0]       ; mem[0] = -30

; === Test 2: Addition of two negatives / 两负数相加 ===
MOV r4, -10      ; r4 = -10
MOV r5, -5       ; r5 = -5
ADD r6, r4, r5   ; r6 = -10 + (-5) = -15  (0xFFFFFFF1)
ST r6, [1]       ; mem[1] = -15

; === Test 3: Division with negative dividend / 负数除法 ===
MOV r7, -20      ; r7 = -20
MOV r8, 4        ; r8 = 4
DIV r9, r7, r8   ; r9 = -20 / 4 = -5  (商 / quotient)
ST r9, [2]       ; mem[2] = -5

; 程序终止 / Terminate program
HALT
