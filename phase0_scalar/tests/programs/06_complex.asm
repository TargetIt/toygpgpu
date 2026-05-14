; ============================================================
; Phase 0: Scalar Core — Complex Expression
; 阶段 0：标量核心 — 复合表达式求值
;
; Purpose / 目的:
;   Combine multiple arithmetic operations into a single
;   expression, demonstrating data-flow dependency chains.
;   将多个算术运算组合为一个表达式，演示数据流
;   依赖链。
;
; Formula / 公式:
;   result = (10 + 5) * (20 - 8) / 4
;          = 15 * 12 / 4
;          = 180 / 4
;          = 45
;
; Expected Result / 预期结果:
;   mem[0] = 45
;
; Key Concepts / 关键概念:
;   - Multi-operation expression evaluation
;   - Register-to-register data dependencies
;   - SUB (subtraction) instruction introduced
;   - Intermediate results passed through registers
;   - Expression: ((a + b) * (c - d)) / e
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10      ← a = 10
;   MOV r2, 5       ← b = 5
;     |
;     v
;   ADD r3, r1, r2  ← r3 = 10 + 5 = 15  (第一部分 / part 1)
;     |
;     v
;   MOV r4, 20      ← c = 20
;   MOV r5, 8       ← d = 8
;     |
;     v
;   SUB r6, r4, r5  ← r6 = 20 - 8 = 12  (第二部分 / part 2)
;     |
;     v
;   MUL r7, r3, r6  ← r7 = 15 * 12 = 180  (第三部分 / part 3)
;     |
;     v
;   MOV r8, 4       ← e = 4
;   DIV r9, r7, r8  ← r9 = 180 / 4 = 45  (最终结果 / final)
;     |
;     v
;   ST r9, [0]      ← mem[0] = 45
;     |
;     v
;   HALT
; ============================================================

; Step 1: 计算括号内的加法 / Compute addition in parentheses
; (10 + 5) — 先计算左半部分
MOV r1, 10       ; r1 = 10  (被加数 / augend)
MOV r2, 5        ; r2 = 5   (加数 / addend)
ADD r3, r1, r2   ; r3 = 10 + 5 = 15

; Step 2: 计算括号内的减法 / Compute subtraction in parentheses
; (20 - 8) — 再计算右半部分
MOV r4, 20       ; r4 = 20  (被减数 / minuend)
MOV r5, 8        ; r5 = 8   (减数 / subtrahend)
SUB r6, r4, r5   ; r6 = 20 - 8 = 12

; Step 3: 执行乘法 / Perform multiplication
; 15 * 12 — 将前两步的结果相乘
MUL r7, r3, r6   ; r7 = 15 * 12 = 180

; Step 4: 执行除法 / Perform division
; 180 / 4 — 最后除以常数
MOV r8, 4        ; r8 = 4   (除数 / divisor)
DIV r9, r7, r8   ; r9 = 180 / 4 = 45

; Step 5: 存储结果 / Store final result
ST r9, [0]       ; mem[0] = 45

; 程序终止 / Terminate program
HALT
