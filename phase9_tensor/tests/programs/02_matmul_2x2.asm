; ============================================================
; Phase 9: Tensor Core — 2x2 Matrix Multiply (D = A x B + C)
; 阶段 9：张量核心 — 2x2 矩阵乘法
;
; Purpose / 目的:
;   Demonstrate full 2x2 matrix multiplication using the MMA
;   instruction. Each MMA computes one element of the result
;   matrix D = A * B + C.
;   使用 MMA 指令计算完整的 2x2 矩阵乘法。
;   每个 MMA 计算结果矩阵 D = A * B + C 的一个元素。
;
; Matrix layout / 矩阵布局:
;   A = [a00 a01]  = [[2, 3],
;       [a10 a11]]     [1, 4]]
;
;   B = [b00 b01]  = [[4, 5],
;       [b10 b11]]     [6, 7]]
;
;   C = 1 (scalar broadcast / 标量广播)
;
; Expected Result / 预期结果:
;   D[0][0] = 2*4 + 3*6 + 1 = 27  → mem[10]
;   D[0][1] = 2*5 + 3*7 + 1 = 32  → mem[11]
;   D[1][0] = 1*4 + 4*6 + 1 = 29  → mem[12]
;   D[1][1] = 1*5 + 4*7 + 1 = 34  → mem[13]
;
; Key Concepts / 关键概念:
;   - Matrix A stored row-wise: A_row0 = [a01, a00]; A_row1 = [a11, a10]
;   - Matrix B stored column-wise: B_col0 = [b10, b00]; B_col1 = [b11, b01]
;   - Each MMA does a row-of-A times column-of-B dot product + C
;   - 这是张量核心加速矩阵乘法的基本模式
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   LD r1, [0]      ← r1 = A_row0 = packed [a01=3, a00=2]
;   LD r2, [1]      ← r2 = A_row1 = packed [a11=4, a10=1]
;   LD r3, [2]      ← r3 = B_col0 = packed [b10=6, b00=4]
;   LD r4, [3]      ← r4 = B_col1 = packed [b11=7, b01=5]
;   MOV r5, 1       ← r5 = C = 1
;     |
;     v
;   MMA r6, r1, r3, r5  ← D[0][0] = 2*4 + 3*6 + 1 = 27
;   ST r6, [10]
;     |
;     v
;   MMA r7, r1, r4, r5  ← D[0][1] = 2*5 + 3*7 + 1 = 32
;   ST r7, [11]
;     |
;     v
;   MMA r8, r2, r3, r5  ← D[1][0] = 1*4 + 4*6 + 1 = 29
;   ST r8, [12]
;     |
;     v
;   MMA r9, r2, r4, r5  ← D[1][1] = 1*5 + 4*7 + 1 = 34
;   ST r9, [13]
;     |
;     v
;   HALT
;
; Data layout in memory / 内存数据布局:
;   mem[0] = A_row0  = [3, 2]  (a01=3, a00=2)
;   mem[1] = A_row1  = [4, 1]  (a11=4, a10=1)
;   mem[2] = B_col0  = [6, 4]  (b10=6, b00=4)
;   mem[3] = B_col1  = [7, 5]  (b11=7, b01=5)
; ============================================================

; === Step 1: Load matrices A and B from memory / 从内存加载矩阵 ===
; A 的行已打包为 16 位元素
; Rows of A packed as 16-bit elements
LD r1, [0]       ; r1 = A_row0 = packed [a01=3, a00=2]
LD r2, [1]       ; r2 = A_row1 = packed [a11=4, a10=1]

; B 的列已打包为 16 位元素
; Columns of B packed as 16-bit elements
LD r3, [2]       ; r3 = B_col0 = packed [b10=6, b00=4]
LD r4, [3]       ; r4 = B_col1 = packed [b11=7, b01=5]

; 加载累加器常量 C / Load accumulator constant C
MOV r5, 1        ; r5 = C = 1

; === Step 2: Compute D[0][0] = A_row0 · B_col0 + C ===
MMA r6, r1, r3, r5  ; r6 = 2*4 + 3*6 + 1 = 27
ST r6, [10]         ; mem[10] = 27

; === Step 3: Compute D[0][1] = A_row0 · B_col1 + C ===
MMA r7, r1, r4, r5  ; r7 = 2*5 + 3*7 + 1 = 32
ST r7, [11]         ; mem[11] = 32

; === Step 4: Compute D[1][0] = A_row1 · B_col0 + C ===
MMA r8, r2, r3, r5  ; r8 = 1*4 + 4*6 + 1 = 29
ST r8, [12]         ; mem[12] = 29

; === Step 5: Compute D[1][1] = A_row1 · B_col1 + C ===
MMA r9, r2, r4, r5  ; r9 = 1*5 + 4*7 + 1 = 34
ST r9, [13]         ; mem[13] = 34

; 程序终止 / Terminate program
HALT
