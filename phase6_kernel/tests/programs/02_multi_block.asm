; ============================================================
; Phase 6: Kernel Launch — Multi-Block Kernel
; 阶段 6：内核启动 — 多块内核
;
; Purpose / 目的:
;   Demonstrate multi-block kernel execution. Each thread
;   in each block computes C[tid] = tid * 2. Multiple blocks
;   execute the same kernel code independently.
;   演示多块内核执行。每个块中的每个线程计算
;   C[tid] = tid * 2。多个块独立执行相同的内核代码。
;
; Expected Result / 预期结果:
;   Each block writes to mem[50+tid] = tid * 2
;   (Different blocks overlap addresses here — in real GPU,
;    blocks would use blockId * blockDim + tid addressing)
;   (不同块在此地址重叠 — 实际 GPU 中会使用
;    blockId * blockDim + tid 寻址)
;
; Key Concepts / 关键概念:
;   - Multiple blocks execute the same kernel concurrently
;   - Each block has its own warp(s) and register state
;   - No inter-block synchronization (they execute independently)
;   - 多块并行执行是 GPU 可扩展性的基础
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1           ← r1 = tid (块内线程索引)
;   MOV r2, 2
;   MUL r3, r1, r2   ← r3 = tid * 2
;     |
;     v
;   ST r3, [50]      ← mem[50+tid] = tid * 2
;     |
;     v
;   HALT
; ============================================================

; 获取线程 ID / Get thread ID
TID r1           ; r1 = tid (当前块内的线程索引)

; 计算值 = tid * 2 / Compute value = tid * 2
MOV r2, 2        ; r2 = 2
MUL r3, r1, r2   ; r3 = tid * 2

; 存储到全局内存 / Store to global memory
; 每个块中的线程都执行此操作
; Threads in every block execute this same store
ST r3, [50]      ; mem[50 + tid] = tid * 2

; 程序终止 / Terminate program
HALT
