; ============================================================
; Phase 0: Scalar Core — Memory Load/Store
; 阶段 0：标量核心 — 内存加载与存储
;
; Purpose / 目的:
;   Demonstrate global memory load (LD) and store (ST)
;   instructions, including read-after-write across
;   different addresses.
;   演示全局内存加载 (LD) 和存储 (ST) 指令，包括
;   跨不同地址的写后读操作。
;
; Expected Result / 预期结果:
;   mem[10] = 100, mem[20] = 200, mem[30] = 300
;   r3 = 100, r4 = 200, r5 = 300
;
; Key Concepts / 关键概念:
;   - LD (load): read value from global memory into register
;   - ST (store): write register value to global memory
;   - Multiple memory addresses can be used independently
;   - ALU can operate on loaded values
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 100     ← 准备数据
;   ST r1, [10]     ← mem[10] = 100  (写入地址 10)
;     |
;     v
;   MOV r2, 200     ← 准备数据
;   ST r2, [20]     ← mem[20] = 200  (写入地址 20)
;     |
;     v
;   LD r3, [10]     ← r3 = mem[10] = 100  (读回)
;   LD r4, [20]     ← r4 = mem[20] = 200  (读回)
;     |
;     v
;   ADD r5, r3, r4  ← r5 = 100 + 200 = 300
;     |
;     v
;   ST r5, [30]     ← mem[30] = 300
;     |
;     v
;   HALT
; ============================================================

; === Phase 1: Write data to memory / 阶段 1：写入数据到内存 ===

; 向地址 10 写入 100 / Write 100 to address 10
MOV r1, 100      ; r1 = 100  (待存储的值 / value to store)
ST r1, [10]      ; mem[10] = 100

; 向地址 20 写入 200 / Write 200 to address 20
MOV r2, 200      ; r2 = 200
ST r2, [20]      ; mem[20] = 200

; === Phase 2: Read data back from memory / 阶段 2：从内存读回数据 ===

; 从地址 10 加载 / Load from address 10
LD r3, [10]      ; r3 = mem[10] = 100

; 从地址 20 加载 / Load from address 20
LD r4, [20]      ; r4 = mem[20] = 200

; === Phase 3: Operate on loaded values / 阶段 3：对加载的值做运算 ===

; 将两个加载的值相加 / Add the two loaded values
ADD r5, r3, r4   ; r5 = 100 + 200 = 300

; 将总和写回新地址 / Store sum to new address
ST r5, [30]      ; mem[30] = 300

; 程序终止 / Terminate program
HALT
