; ============================================================
; Phase 9: Tensor Core — MMA Dot Product
; 阶段 9：张量核心 — MMA 点积
;
; Purpose / 目的:
;   Demonstrate the MMA (Matrix Multiply-Accumulate) instruction
;   for packed dot product. The operands are packed pairs stored
;   in single 32-bit registers:
;     r1 = packed [a1=3, a0=2]   (高位 = a1, 低位 = a0)
;     r2 = packed [b1=5, b0=4]   (高位 = b1, 低位 = b0)
;   MMA computes: r4 = a0*b0 + a1*b1 + r3(10) = 2*4 + 3*5 + 10 = 33
;
; 演示 MMA 指令用于打包点积。操作数是打包在单个 32 位
; 寄存器中的值对。
;
; Expected Result / 预期结果:
;   mem[10] = 33  (2*4 + 3*5 + 10 = 8 + 15 + 10)
;
; Key Concepts / 关键概念:
;   - MMA: fused multiply-add of packed 16-bit elements
;   - r1 = {hi: a1=3, lo: a0=2} — two 16-bit values in one reg
;   - r2 = {hi: b1=5, lo: b0=4}
;   - r4 = a0*b0 + a1*b1 + r3 (accumulator)
;   - This is a simplified tensor core operation
;   - MMA 是张量核心的基本运算单元
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   LD r1, [0]      ← r1 = packed [a1=3, a0=2]
;   LD r2, [1]      ← r2 = packed [b1=5, b0=4]
;   MOV r3, 10      ← r3 = 10 (累加器初始值 / accumulator init)
;     |
;     v
;   MMA r4, r1, r2, r3  ← r4 = a0*b0 + a1*b1 + r3
;     |                    = 2*4 + 3*5 + 10
;     |                    = 8 + 15 + 10 = 33
;     v
;   ST r4, [10]     ← mem[10] = 33
;     |
;     v
;   HALT
;
; 数据布局 / Data layout:
;   mem[0] = 0x00030002  (a1=3 | a0=2)
;   mem[1] = 0x00050004  (b1=5 | b0=4)
; ============================================================

; 从内存加载打包的操作数 / Load packed operands from memory
; 注意: 需要预先在 mem[0] 和 mem[1] 中放置打包数据
; Note: packed data must be pre-loaded in mem[0] and mem[1]
LD r1, [0]       ; r1 = mem[0] = packed [a1=3, a0=2]
LD r2, [1]       ; r2 = mem[1] = packed [b1=5, b0=4]

; 加载累加器初始值 / Load accumulator initial value
MOV r3, 10       ; r3 = 10

; 执行 MMA 点积运算 / Perform MMA dot product
; r4 = a0*b0 + a1*b1 + r3
; 其中 a0=2, a1=3, b0=4, b1=5
; where a0=2, a1=3, b0=4, b1=5
MMA r4, r1, r2, r3  ; r4 = 2*4 + 3*5 + 10 = 8 + 15 + 10 = 33

; 存储结果 / Store result
ST r4, [10]      ; mem[10] = 33

; 程序终止 / Terminate program
HALT
