; ============================================================
; Phase 12: Warp Communication — SHFL (Warp Shuffle)
; 阶段 12：Warp 通信 — Warp 洗牌指令
;
; Purpose / 目的:
;   Demonstrate warp-level shuffle: each thread reads a register
;   from ANOTHER thread in the same warp. SHFL enables fast
;   warp-wide data exchange without shared memory.
;   演示 warp 内 shuffle：每个线程读取同 warp 内另一个线程的
;   寄存器值，无需通过 shared memory 即可实现 warp 内数据交换。
;
; SHFL Modes / 模式:
;   SHFL rd, rs1, lane, 0  — IDX:  读 lane 号线程的 rs1
;   SHFL rd, rs1, delta, 1 — UP:   读 (tid-delta) 号线程的 rs1
;   SHFL rd, rs1, delta, 2 — DOWN: 读 (tid+delta) 号线程的 rs1
;   SHFL rd, rs1, mask, 3  — XOR:  读 (tid^mask) 号线程的 rs1
;
; Scenario / 场景 (warp_size=8):
;   Step 1: Each thread computes tid * 10 (unique per thread)
;   Step 2: SHFL IDX reads thread 0's value → all get 0
;   Step 3: SHFL DOWN(1) reads neighbor → butterfly pattern
;   Step 4: SHFL XOR(1) reads XOR partner → pairwise exchange
;
; Expected Results / 预期结果:
;   mem[0+tid]  = tid * 10  (每个线程的原始值)
;   mem[8+tid]  = 0 (所有线程读到 tid=0 的值)
;   mem[16+tid] = (tid+1) * 10 (SHFL DOWN 1: 读到下一个线程的值)
;   mem[24+tid] = (tid^1) * 10 (SHFL XOR 1: 读到配对线程的值)
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1              ← r1 = tid
;   MOV r2, 10
;   MUL r3, r1, r2      ← r3 = tid * 10
;   ST r3, [0]          ← mem[tid] = tid*10 (原始值)
;     |
;     v
;   SHFL r4, r3, 0, 0   ← IDX: 读 thread 0 的 r3 → 所有线程得到 0
;   ST r4, [8]          ← mem[8+tid] = r4 (应该全是 0)
;     |
;     v
;   SHFL r5, r3, 1, 2   ← DOWN 1: 读 tid+1 的 r3
;   ST r5, [16]         ← mem[16+tid] = (tid+1)*10
;     |
;     v
;   SHFL r6, r3, 1, 3   ← XOR 1: 读 tid^1 的 r3 (交换对)
;   ST r6, [24]         ← mem[24+tid] = (tid^1)*10
;     |
;     v
;   HALT
; ============================================================

; Step 1: Compute per-thread unique value (计算每线程唯一值)
TID r1                   ; r1 = thread_id (0..7)
MOV r2, 10               ; r2 = 10
MUL r3, r1, r2           ; r3 = tid * 10

; Store original values (存储原始值)
ST r3, [0]               ; mem[0 + tid] = tid * 10

; Step 2: SHFL IDX — all threads read thread 0's r3
; IDX 模式: 所有线程读取线程 0 的 r3 (= 0)
SHFL r4, r3, 0, 0        ; r4 = r3 of thread 0 = 0
ST r4, [8]               ; mem[8 + tid] = 0 (所有线程相同)

; Step 3: SHFL DOWN 1 — each thread reads neighbor below
; DOWN 模式: 每个线程读取 tid+1 的 r3
SHFL r5, r3, 1, 2        ; r5 = r3 of thread (tid+1)%8
ST r5, [16]              ; mem[16 + tid] = (tid+1)%8 * 10

; Step 4: SHFL XOR 1 — pairwise exchange
; XOR 模式: 每个线程读取 tid^1 的 r3 (配对交换)
SHFL r6, r3, 1, 3        ; r6 = r3 of thread (tid^1)
ST r6, [24]              ; mem[24 + tid] = (tid^1) * 10

HALT
