; ============================================================
; Phase 5: Memory Subsystem — Multi-Warp Shared Memory
; 阶段 5：内存子系统 — 多 Warp 共享内存
;
; Purpose / 目的:
;   Demonstrate shared memory access across multiple warps.
;   All warps use the same shared memory address space.
;   Since shared memory is per-block, different warps in the
;   same block can communicate through shared memory.
;   演示多个 warp 对共享内存的访问。所有 warp 使用相同
;   的共享内存地址空间。
;
; Expected Result / 预期结果:
;   shared_mem[tid] = (last_wid * 10 + tid)
;   mem[200+tid] = same value
;
; Key Concepts / 关键概念:
;   - Shared memory is shared by all warps in a block
;   - Last warp to write wins (no synchronization shown here)
;   - Combined SHST + SHLD + ST demonstrates shared-to-global copy
;   - 共享内存可用于 warp 间通信
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   WID r1           ← r1 = wid (warp ID)
;   MOV r2, 10
;   MUL r3, r1, r2   ← r3 = wid * 10 (warp 偏移 / warp offset)
;     |
;     v
;   TID r4           ← r4 = tid
;   ADD r5, r3, r4   ← r5 = wid*10 + tid (每个线程唯一值)
;     |
;     v
;   SHST r5, [0]     ← shared_mem[tid] = wid*10+tid
;     |                (所有 warp 写入同一个共享内存区)
;     |                (All warps write to same shared memory area)
;     v
;   SHLD r6, [0]     ← r6 = shared_mem[tid] (从共享内存读回)
;     |
;     v
;   ST r6, [200]     ← mem[200+tid] = r6 (写到全局内存验证)
;     |
;     v
;   HALT
;
; Note / 注意:
;   多个 warp 同时写入同一个共享内存地址时, 最后写入的
;   warp 的值会覆盖之前的。实际使用中需要 BAR + SHST 组合
;   来确保正确的同步。
;   Without synchronization, the last warp to write wins.
; ============================================================

; 计算 warp 基址偏移 / Compute warp base offset
WID r1           ; r1 = wid (warp ID)
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = wid * 10

; 计算线程唯一值 / Compute unique thread value
TID r4           ; r4 = tid (warp 内线程索引 / thread within warp)
ADD r5, r3, r4   ; r5 = wid * 10 + tid

; 写入共享内存 / Write to shared memory
; 所有 warp 的线程都通过 TID 索引共享内存
; All warps' threads index shared memory by TID
SHST r5, [0]     ; shared_mem[tid] = wid * 10 + tid

; 从共享内存读回 / Read back from shared memory
SHLD r6, [0]     ; r6 = shared_mem[tid]

; 写入全局内存以验证 / Write to global memory for verification
ST r6, [200]     ; mem[200 + tid] = r6

; 程序终止 / Terminate program
HALT
