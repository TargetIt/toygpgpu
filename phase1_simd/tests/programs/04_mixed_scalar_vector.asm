; ============================================================
; Phase 1: SIMD Vector Core — Mixed Scalar + Vector
; 阶段 1：SIMD 向量核心 — 标量与向量混合运算
;
; Purpose / 目的:
;   Demonstrate combining scalar and vector instructions:
;   C[i] = (A[i] + B[i]) * 3.
;   演示标量与向量指令的组合使用。
;
; Formula / 公式:
;   C[i] = (A[i] + B[i]) * 3,  i = 0..7
;
; Expected Result / 预期结果:
;   mem[20..27] = [33, 66, 99, 132, 165, 198, 231, 264]
;   ((A[i] + B[i]) * 3)
;
; Key Concepts / 关键概念:
;   - Scalar initialization (MOV/ST) + vector processing (VLD/VADD/VMUL/VST)
;   - Multi-step vector expression: add first, then multiply
;   - Register reuse: v3 holds intermediate sum, v6 holds final result
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   初始化 A: mem[0..7] = [10, 20, 30, 40, 50, 60, 70, 80]
;   初始化 B: mem[8..15] = [1, 2, 3, 4, 5, 6, 7, 8]
;     |
;     v
;   第一步: 向量加法 C'[i] = A[i] + B[i]
;   VLD v1, [0]     ← 加载 A
;   VLD v2, [8]     ← 加载 B
;   VADD v3, v1, v2 ← v3[i] = A[i] + B[i]
;     |
;     v
;   第二步: 向量缩放 C[i] = C'[i] * 3
;   VMOV v5, 3      ← 广播标量 3
;   VMUL v6, v3, v5 ← v6[i] = C'[i] * 3
;     |
;     v
;   VST v6, [20]    ← mem[20..27] = C
;     |
;     v
;   HALT
; ============================================================

; === Step 1: Initialize vector A in memory / 初始化向量 A ===
; 向量 A: [10, 20, 30, 40, 50, 60, 70, 80]
MOV r1, 10
ST r1, [0]
MOV r1, 20
ST r1, [1]
MOV r1, 30
ST r1, [2]
MOV r1, 40
ST r1, [3]
MOV r1, 50
ST r1, [4]
MOV r1, 60
ST r1, [5]
MOV r1, 70
ST r1, [6]
MOV r1, 80
ST r1, [7]

; === Step 2: Initialize vector B in memory / 初始化向量 B ===
; 向量 B: [1, 2, 3, 4, 5, 6, 7, 8]
MOV r1, 1
ST r1, [8]
MOV r1, 2
ST r1, [9]
MOV r1, 3
ST r1, [10]
MOV r1, 4
ST r1, [11]
MOV r1, 5
ST r1, [12]
MOV r1, 6
ST r1, [13]
MOV r1, 7
ST r1, [14]
MOV r1, 8
ST r1, [15]

; === Step 3: Vector add — A + B / 向量加法 ===
VLD v1, [0]      ; v1 = A[0..7]
VLD v2, [8]      ; v2 = B[0..7]
VADD v3, v1, v2  ; v3[i] = A[i] + B[i]  (中间和 / intermediate sum)

; === Step 4: Scale result by 3 / 缩放结果乘以 3 ===
VMOV v5, 3       ; v5 = [3, 3, 3, 3, 3, 3, 3, 3] (标量广播)
VMUL v6, v3, v5  ; v6[i] = (A[i] + B[i]) * 3  (最终结果 / final)

; === Step 5: Store final result / 存储最终结果 ===
VST v6, [20]     ; mem[20..27] = C[0..7]

; 程序终止 / Terminate program
HALT
