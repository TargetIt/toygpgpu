; ============================================================
; Phase 0: Scalar Core — Multiply and Divide
; 阶段 0：标量核心 — 乘法和除法
;
; Purpose / 目的:
;   Demonstrate integer multiplication (MUL) and
;   integer division (DIV) instructions.
;   演示整数乘法 (MUL) 和整数除法 (DIV) 指令。
;
; Expected Result / 预期结果:
;   mem[0] = 42  (6 * 7)
;   mem[1] = 7   (42 / 6)
;   r3 = 42, r5 = 7
;
; Key Concepts / 关键概念:
;   - MUL (multiply):  rD = rA * rB
;   - DIV (divide):    rD = rA / rB (integer truncation)
;   - Register reuse: r3 result fed into DIV as dividend
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 6       ← 加载 6
;   MOV r2, 7       ← 加载 7
;     |
;     v
;   MUL r3, r1, r2  ← r3 = 6 * 7 = 42
;     |
;     v
;   ST r3, [0]      ← mem[0] = 42
;     |
;     v
;   MOV r4, 6       ← 加载除数 6
;   DIV r5, r3, r4  ← r5 = 42 / 6 = 7
;     |
;     v
;   ST r5, [1]      ← mem[1] = 7
;     |
;     v
;   HALT
; ============================================================

; === Part 1: Multiplication / 第一部分：乘法 ===

; 加载被乘数 / Load multiplicand
MOV r1, 6        ; r1 = 6

; 加载乘数 / Load multiplier
MOV r2, 7        ; r2 = 7

; 执行乘法 / Perform multiplication
MUL r3, r1, r2   ; r3 = 6 * 7 = 42  (乘积 / product)

; 存储乘积 / Store product
ST r3, [0]       ; mem[0] = 42

; === Part 2: Division / 第二部分：除法 ===

; 加载除数 / Load divisor
MOV r4, 6        ; r4 = 6

; 执行除法 / Perform division
; r3 保留上一结果 (42), 将其除以 r4 (6)
; r3 still holds 42 from MUL, divide it by r4 (6)
DIV r5, r3, r4   ; r5 = 42 / 6 = 7  (商 / quotient)

; 存储商 / Store quotient
ST r5, [1]       ; mem[1] = 7

; 程序终止 / Terminate program
HALT
