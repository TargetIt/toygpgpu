; ============================================================
; Phase 1: SIMD Vector Core — Vector Subtract and Divide
; 阶段 1：SIMD 向量核心 — 向量减法与除法
;
; Purpose / 目的:
;   Demonstrate VSUB (vector subtract) and VDIV (vector
;   divide), chaining them sequentially.
;   演示 VSUB（向量减法）和 VDIV（向量除法）指令，
;   并将它们串联使用。
;
; Formula / 公式:
;   tmp[i] = A[i] - 5     (逐元素减法)
;   C[i]   = tmp[i] / 2   (逐元素除法)
;   i = 0..7
;
; Expected Result / 预期结果:
;   mem[10..17] = [95, 75, 55, 35, 15, 5, 3, -1]  (减 5 后)
;   mem[20..27] = [47, 37, 27, 17, 7, 2, 1, -1]  (再除以 2)
;
; Key Concepts / 关键概念:
;   - VSUB: element-wise vD = vA - vB
;   - VDIV: element-wise vD = vA / vB (integer divide)
;   - Vector instruction chaining (VSUB → VST → VLD → VDIV)
;   - Mix of positive and negative results
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   初始化 A: mem[0..7] = [100, 80, 60, 40, 20, 10, 8, 4]
;     |
;     v
;   第一阶段: A[i] - 5
;   VMOV v2, 5      ← 广播标量 5
;   VLD v1, [0]     ← 加载 A
;   VSUB v3, v1, v2 ← v3[i] = A[i] - 5
;   VST v3, [10]    ← 临时结果存到 mem[10..17]
;     |
;     v
;   第二阶段: tmp[i] / 2
;   VMOV v4, 2      ← 广播标量 2
;   VLD v5, [10]    ← 加载临时结果
;   VDIV v6, v5, v4 ← v6[i] = tmp[i] / 2
;   VST v6, [20]    ← 最终结果存到 mem[20..27]
;     |
;     v
;   HALT
; ============================================================

; === Step 1: Initialize input vector A / 初始化输入向量 A ===
; 向量 A: [100, 80, 60, 40, 20, 10, 8, 4]
MOV r1, 100
ST r1, [0]
MOV r1, 80
ST r1, [1]
MOV r1, 60
ST r1, [2]
MOV r1, 40
ST r1, [3]
MOV r1, 20
ST r1, [4]
MOV r1, 10
ST r1, [5]
MOV r1, 8
ST r1, [6]
MOV r1, 4
ST r1, [7]

; === Step 2: Subtract scalar / 减去标量 ===
; v3[i] = A[i] - 5  (每个元素都减去 5)
VMOV v2, 5       ; v2 = [5, 5, 5, 5, 5, 5, 5, 5] (标量广播)
VLD v1, [0]      ; v1 = A[0..7]  ← 从内存加载
VSUB v3, v1, v2  ; v3[i] = A[i] - 5  (逐元素减法)
VST v3, [10]     ; 临时结果存到 mem[10..17]

; === Step 3: Divide by scalar / 除以标量 ===
; v6[i] = tmp[i] / 2  (每个元素都除以 2)
VMOV v4, 2       ; v4 = [2, 2, 2, 2, 2, 2, 2, 2] (标量广播)
VLD v5, [10]     ; v5 = tmp[0..7]  ← 加载减法的结果
VDIV v6, v5, v4  ; v6[i] = tmp[i] / 2  (逐元素除法)
VST v6, [20]     ; 最终结果存到 mem[20..27]

; 程序终止 / Terminate program
HALT
