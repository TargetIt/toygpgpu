; ============================================================
; Phase 11: Console Demo — Basic Scalar Program
; 阶段 11：控制台演示 — 基本标量程序
;
; Purpose / 目的:
;   A simple scalar program designed for the interactive
;   console. Computes (5 + 3) * 3 = 24 and stores the
;   result. This is the ideal "first program" for learners
;   to step through using the console's single-step mode.
;   为交互式控制台设计的简单标量程序，适合初学者
;   使用控制台单步执行模式逐条学习。
;
; Formula / 公式:  r4 = (5 + 3) * 3 = 24
;
; Expected Result / 预期结果:
;   mem[0] = 24
;   Pipeline state / 流水线状态:
;     r1=5, r2=3, r3=8, r4=24
;
; Key Concepts / 关键概念:
;   - Minimal instruction count (single-step friendly)
;   - Demonstrates MOV, ADD, MUL, ST, HALT
;   - No SIMT/TID — purely scalar, one thread only
;   - Perfect for verifying pipeline stages visually
;   - 适合在控制台中观察每条指令对寄存器和内存的影响
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 5       ← r1 = 5
;     |
;     v
;   MOV r2, 3       ← r2 = 3
;     |
;     v
;   ADD r3, r1, r2  ← r3 = 5 + 3 = 8
;     |
;     v
;   MUL r4, r3, r2  ← r4 = 8 * 3 = 24
;     |
;     v
;   ST r4, [0]      ← mem[0] = 24
;     |
;     v
;   HALT
; ============================================================

; 加载第一个操作数 / Load first operand
MOV r1, 5        ; r1 = 5

; 加载第二个操作数 / Load second operand
MOV r2, 3        ; r2 = 3

; Step 1: 加法 — r3 = r1 + r2 / Addition
ADD r3, r1, r2   ; r3 = 5 + 3 = 8

; Step 2: 乘法 — r4 = r3 * r2 / Multiplication
MUL r4, r3, r2   ; r4 = 8 * 3 = 24

; Step 3: 存储结果到内存 / Store result to memory
ST r4, [0]       ; mem[0] = 24

; 程序终止 / Terminate program
HALT
