; ============================================================
; Phase 1: SIMD Vector Core — Vector Multiply (Scale)
; 阶段 1：SIMD 向量核心 — 向量乘法（缩放）
;
; Purpose / 目的:
;   Demonstrate SIMD vector scaling: C[i] = A[i] * 3.
;   A vector is multiplied element-wise by a scalar broadcast.
;   演示 SIMD 向量缩放：向量每个元素与标量 3 相乘。
;
; Expected Result / 预期结果:
;   mem[10..17] = [3, 6, 9, 12, 15, 18, 21, 24]
;   (A[i] * 3 for i=0..7)
;
; Key Concepts / 关键概念:
;   - VMOV (vector move immediate): broadcast scalar to all lanes
;   - VMUL (vector multiply): element-wise vD = vA * vB
;   - Vector-scalar multiplication via broadcast
;   - 向量-标量乘法，通过 VMOV 将标量广播到所有通道
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   初始化向量 A: mem[0..7] = [1, 2, 3, 4, 5, 6, 7, 8]
;     |
;     v
;   VMOV v2, 3      ← v2 = [3, 3, 3, 3, 3, 3, 3, 3] (标量广播)
;     |
;     v
;   VLD v1, [0]     ← v1 = A[0..7]
;     |
;     v
;   VMUL v3, v1, v2 ← v3[i] = A[i] * 3  (逐元素乘法)
;     |
;     v
;   VST v3, [10]    ← 结果存回 mem[10..17]
;     |
;     v
;   HALT
; ============================================================

; === Step 1: Initialize vector A in memory / 初始化向量 A ===
; 向量 A: [1, 2, 3, 4, 5, 6, 7, 8]
MOV r1, 1
ST r1, [0]
MOV r1, 2
ST r1, [1]
MOV r1, 3
ST r1, [2]
MOV r1, 4
ST r1, [3]
MOV r1, 5
ST r1, [4]
MOV r1, 6
ST r1, [5]
MOV r1, 7
ST r1, [6]
MOV r1, 8
ST r1, [7]

; === Step 2: Create scalar broadcast vector / 创建标量广播向量 ===
; VMOV 将标量值 3 复制到向量的所有 8 个通道
; VMOV broadcasts the scalar 3 to all 8 lanes
VMOV v2, 3       ; v2 = [3, 3, 3, 3, 3, 3, 3, 3]

; === Step 3: Load vector A / 加载向量 A ===
VLD v1, [0]      ; v1 = A[0..7]  ← 从 mem[0..7] 加载

; === Step 4: Perform vector scale / 执行向量缩放 ===
; 每个元素乘以 3 / Each element multiplied by 3
VMUL v3, v1, v2  ; v3[i] = A[i] * 3  (逐元素乘法)

; === Step 5: Store result / 存储结果 ===
VST v3, [10]     ; mem[10..17] = C[0..7]

; 程序终止 / Terminate program
HALT
