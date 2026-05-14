; ============================================================
; Phase 9: Tensor Core — MMA with Memory Pre-load
; 阶段 9：张量核心 — 内存预加载的 MMA
;
; Purpose / 目的:
;   Demonstrate MMA using data pre-loaded from memory.
;   This test verifies the MMA instruction works correctly
;   with arbitrary packed values stored in global memory.
;   演示使用从内存预加载的数据执行 MMA 指令。
;
; Formula / 公式:
;   r4 = a0*b0 + a1*b1 + r3
;      = 1*3 + 7*2 + 10
;      = 3 + 14 + 10 = 27
;
; Expected Result / 预期结果:
;   mem[10] = 27
;
; Key Concepts / 关键概念:
;   - MMA can work with any packed data layout
;   - The accumulator (r3) can be any value
;   - Same instruction format as 01_mma_dot.asm
;   - Useful for testing different data patterns
;   - MMA 可处理任意打包数据布局
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   LD r1, [0]      ← r1 = packed [a1=7, a0=1]
;   LD r2, [1]      ← r2 = packed [b1=2, b0=3]
;   MOV r3, 10      ← r3 = 10 (累加器 / accumulator)
;     |
;     v
;   MMA r4, r1, r2, r3  ← r4 = 1*3 + 7*2 + 10 = 27
;     |
;     v
;   ST r4, [10]     ← mem[10] = 27
;     |
;     v
;   HALT
;
; Data layout / 数据布局:
;   mem[0] = packed [a1=7, a0=1]
;   mem[1] = packed [b1=2, b0=3]
; ============================================================

; 从内存加载打包的操作数 / Load packed operands from memory
LD r1, [0]       ; r1 = mem[0] = packed [a1=7, a0=1]
LD r2, [1]       ; r2 = mem[1] = packed [b1=2, b0=3]

; 加载累加器值 / Load accumulator value
MOV r3, 10       ; r3 = 10

; 执行 MMA 点积 / Perform MMA dot product
; r4 = a0*b0 + a1*b1 + r3 = 1*3 + 7*2 + 10 = 3 + 14 + 10 = 27
MMA r4, r1, r2, r3  ; r4 = 1*3 + 7*2 + 10 = 27

; 存储结果 / Store result
ST r4, [10]      ; mem[10] = 27

; 程序终止 / Terminate program
HALT
