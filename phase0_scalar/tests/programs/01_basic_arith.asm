; ============================================================
; Phase 0: Scalar Core — Basic Arithmetic
; 阶段 0：标量核心 — 基本算术运算
;
; Purpose / 目的:
;   Demonstrate fundamental integer addition (ADD) and
;   memory store (ST) instructions in the scalar pipeline.
;   演示标量流水线中最基本的整数加法 (ADD) 和
;   内存存储 (ST) 指令。
;
; Expected Result / 预期结果:
;   mem[0] = 8  (5 + 3)
;   r1 = 5, r2 = 3, r3 = 8
;
; Key Concepts / 关键概念:
;   - MOV (move immediate): load constant into register
;   - ADD (integer addition): rD = rA + rB
;   - ST (store): write register value to global memory
;   - HALT: terminate program
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 5       ← 加载常量 5
;     |
;     v
;   MOV r2, 3       ← 加载常量 3
;     |
;     v
;   ADD r3, r1, r2  ← r3 = 5 + 3 = 8
;     |
;     v
;   ST r3, [0]      ← mem[0] = 8
;     |
;     v
;   HALT
; ============================================================

; 加载第一个操作数 / Load first operand
MOV r1, 5        ; r1 = 5  (被加数 / augend)

; 加载第二个操作数 / Load second operand
MOV r2, 3        ; r2 = 3  (加数 / addend)

; 执行加法 / Perform addition
ADD r3, r1, r2   ; r3 = 5 + 3 = 8  (和 / sum)

; 将结果写回内存 / Store result to memory
ST r3, [0]       ; mem[0] = 8

; 程序终止 / Terminate program
HALT
