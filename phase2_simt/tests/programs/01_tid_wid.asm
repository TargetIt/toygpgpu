; ============================================================
; Phase 2: SIMT Core — Thread ID (TID) and Warp ID (WID)
; 阶段 2：SIMT 核心 — 线程 ID 与 Warp ID
;
; Purpose / 目的:
;   Demonstrate the TID (Thread ID) instruction. Each
;   thread in a warp gets a unique ID (0..warp_size-1).
;   Each thread stores its TID to mem[100 + tid].
;   演示 TID（线程 ID）指令。warp 中的每个线程获得
;   一个唯一的 ID（0..warp_size-1）。每个线程将其
;   TID 存入 mem[100 + tid]。
;
; Expected Result / 预期结果:
;   Warp 0 (tid 0..7): mem[100..107] = [0, 1, 2, 3, 4, 5, 6, 7]
;
; Key Concepts / 关键概念:
;   - TID (thread ID): returns the calling thread's index in the warp
;   - SIMT model: all threads execute the same code, but data differs
;   - The address [100] is base; actual address = base + tid
;   - 这是 SIMT（单指令多线程）编程模型的核心概念
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1          ← r1 获得当前线程的 ID (0, 1, 2, ..., 7)
;     |                (取决于哪个线程在执行)
;     v
;   ST r1, [100]    ← mem[100 + tid] = tid
;     |                (每个线程写到不同的地址!)
;     v
;   HALT
;
; Thread mapping / 线程映射:
;   Thread 0: mem[100] = 0
;   Thread 1: mem[101] = 1
;   Thread 2: mem[102] = 2
;   ...
;   Thread 7: mem[107] = 7
; ============================================================

; 获取当前线程的 ID / Get current thread's ID
; TID 将 0..(warp_size-1) 的值写入目标寄存器
; TID writes 0..(warp_size-1) into the destination register
TID r1           ; r1 = tid (线程索引)

; 将线程 ID 存储到全局内存 / Store TID to global memory
; 地址基址为 100，实际上每个线程写入不同的位置
; Base address is 100, each thread writes a different slot
ST r1, [100]     ; mem[100 + tid] = tid

; 程序终止 / Terminate program
HALT
