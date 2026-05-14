; ============================================================
; Phase 1: SIMD Vector Core — Vector Addition
; 阶段 1：SIMD 向量核心 — 向量加法
;
; Purpose / 目的:
;   Demonstrate SIMD (Single Instruction, Multiple Data)
;   vector addition: C[i] = A[i] + B[i] for i = 0..7.
;   Each vector register holds 8×32-bit elements (256 bits).
;   演示 SIMD 向量加法。每个向量寄存器存 8 个 32 位元素。
;
; Expected Result / 预期结果:
;   mem[16..23] = [11, 22, 33, 44, 55, 66, 77, 88]
;   (A[i] + B[i] for i=0..7)
;
; Key Concepts / 关键概念:
;   - VLD (vector load):  load 8 words from memory → vec reg
;   - VADD (vector add):  element-wise vD = vA + vB
;   - VST (vector store): write vector reg → 8 memory words
;   - v1..v3 are 256-bit vector registers
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   初始化向量 A: mem[0..7] = [10, 20, 30, 40, 50, 60, 70, 80]
;   初始化向量 B: mem[8..15] = [1, 2, 3, 4, 5, 6, 7, 8]
;     |
;     v
;   VLD v1, [0]     ← v1 = A[0..7]  (从内存加载向量A)
;   VLD v2, [8]     ← v2 = B[0..7]  (从内存加载向量B)
;     |
;     v
;   VADD v3, v1, v2 ← v3[i] = A[i] + B[i]  (逐元素相加)
;     |
;     v
;   VST v3, [16]    ← 将结果向量存回 mem[16..23]
;     |
;     v
;   HALT
; ============================================================

; === Step 1: Initialize vector A in memory / 初始化向量 A ===
; 向量 A: [10, 20, 30, 40, 50, 60, 70, 80]
; 存入内存地址 0~7
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
; 存入内存地址 8~15
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

; === Step 3: SIMD vector addition / SIMD 向量加法 ===
; 一次指令完成 8 个元素的并行加法
; One instruction performs 8 parallel additions
VLD v1, [0]      ; v1 = A[0..7]  ← 从 mem[0..7] 加载到向量寄存器 v1
VLD v2, [8]      ; v2 = B[0..7]  ← 从 mem[8..15] 加载到向量寄存器 v2
VADD v3, v1, v2  ; v3[i] = A[i] + B[i]  (逐元素加法 / element-wise add)

; === Step 4: Store result vector / 存储结果向量 ===
VST v3, [16]     ; 将 v3 写回 mem[16..23]

; 程序终止 / Terminate program
HALT
