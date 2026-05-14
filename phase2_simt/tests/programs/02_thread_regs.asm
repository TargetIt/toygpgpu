; ============================================================
; Phase 2: SIMT Core — Thread-Independent Registers
; 阶段 2：SIMT 核心 — 线程独立寄存器
;
; Purpose / 目的:
;   Demonstrate that each thread has its own private
;   register file. Each thread computes tid * 10 and
;   stores the result.
;   演示每个线程拥有自己独立的寄存器文件。
;   每个线程计算 tid * 10 并存储结果。
;
; Expected Result / 预期结果:
;   mem[200..207] = [0, 10, 20, 30, 40, 50, 60, 70]
;   (tid * 10 for tid = 0..7)
;
; Key Concepts / 关键概念:
;   - Per-thread register state (not shared between threads)
;   - TID + MOV + MUL combined for per-thread computation
;   - 线程私有寄存器：每个线程的 r1, r2, r3 互不干扰
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1          ← r1 = tid (每个线程独立)
;     |
;     v
;   MOV r2, 10      ← r2 = 10  (常数值, 每个线程相同)
;     |
;     v
;   MUL r3, r1, r2  ← r3 = tid * 10 (每个线程结果不同)
;     |
;     v
;   ST r3, [200]    ← mem[200 + tid] = tid * 10
;     |
;     v
;   HALT
; ============================================================

; 获取当前线程 ID / Get current thread ID
TID r1           ; r1 = tid (0 到 warp_size-1)

; 加载常数值 / Load constant
MOV r2, 10       ; r2 = 10  (乘数 / multiplier)

; 执行乘法 / Perform multiplication
; r3 = tid * 10 — 每个线程有不同的结果
; r3 = tid * 10 — each thread produces a different result
MUL r3, r1, r2   ; r3 = tid * 10

; 存储结果 / Store result
; 地址基址 200 + tid，确保每个线程写到不同位置
; Base address 200 + tid ensures each thread writes a unique slot
ST r3, [200]     ; mem[200 + tid] = tid * 10

; 程序终止 / Terminate program
HALT
