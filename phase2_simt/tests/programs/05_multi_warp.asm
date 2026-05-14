; ============================================================
; Phase 2: SIMT Core — Multi-Warp Execution
; 阶段 2：SIMT 核心 — 多 Warp 执行
;
; Purpose / 目的:
;   Demonstrate multiple warps coexisting and executing
;   independently. Each warp has its own WID (Warp ID)
;   and its threads have TIDs (0..warp_size-1).
;   演示多个 warp 共存并独立执行。每个 warp 有自己的
;   WID（Warp ID），其中的线程有各自的 TID。
;
; Each thread computes a unique memory address:
;   address = wid * 100 + tid
; Then stores that value at mem[tid].
;
; Expected Result / 预期结果:
;   (warp_size = 8)
;   Warp 0: mem[0..7]   = [0, 1, 2, 3, 4, 5, 6, 7]
;   Warp 1: mem[8..15]  = [100, 101, 102, 103, 104, 105, 106, 107]
;   Warp 2: mem[16..23] = [200, 201, 202, ...]
;   (每个线程存储的是自己的 wid*100 + tid)
;
; Key Concepts / 关键概念:
;   - WID (warp ID): uniquely identifies which warp this thread belongs to
;   - Different warps have separate register files
;   - Multiple warps can time-share the SIMD pipeline
;   - 多 warp 是 GPU 实现高吞吐量的关键
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   WID r1          ← r1 = wid (warp 编号, 0, 1, 2, ...)
;     |
;     v
;   MOV r2, 100     ← r2 = 100 (warp 地址跨度)
;   MUL r3, r1, r2  ← r3 = wid * 100 (warp 的基址偏移)
;     |
;     v
;   TID r4          ← r4 = tid (warp 内的线程索引)
;     |
;     v
;   ADD r5, r3, r4  ← r5 = wid*100 + tid (全局唯一 ID)
;     |
;     v
;   ST r5, [0]      ← mem[tid] = wid*100 + tid
;     |                 (实际地址: mem[0+tid])
;     v
;   HALT
;
; 示例 (warp_size=4, 2 warps):
;   Warp 0, Thread 0 → mem[0] = 0*100 + 0 = 0
;   Warp 0, Thread 3 → mem[3] = 0*100 + 3 = 3
;   Warp 1, Thread 0 → mem[4] = 1*100 + 0 = 100
;   Warp 1, Thread 3 → mem[7] = 1*100 + 3 = 103
; ============================================================

; 获取 warp ID / Get warp ID
WID r1           ; r1 = wid (warp 编号, 从 0 开始)

; 计算 warp 基址偏移 / Compute warp base offset
; 每个 warp 负责不同的地址区域
; Each warp owns a distinct address region
MOV r2, 100      ; r2 = 100  (warp 地址间隔)
MUL r3, r1, r2   ; r3 = wid * 100  (当前 warp 的基址偏移)

; 获取线程 ID / Get thread ID within warp
TID r4           ; r4 = tid (warp 内的线程索引)

; 计算全局唯一标识 / Compute globally unique identifier
; 不同 warp 的线程会产生不同的值
; Threads from different warps produce different values
ADD r5, r3, r4   ; r5 = wid * 100 + tid

; 存储到内存 / Store to memory
; 由于所有 warp 执行相同代码但 WID 不同，每个线程
; 获得唯一的 r5 值
ST r5, [0]       ; mem[tid] = wid*100 + tid

; 程序终止 / Terminate program
HALT
