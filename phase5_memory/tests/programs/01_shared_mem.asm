; ============================================================
; Phase 5: Memory Subsystem — Shared Memory Read/Write
; 阶段 5：内存子系统 — 共享内存读写
;
; Purpose / 目的:
;   Demonstrate shared memory (SHST/SHLD) instructions.
;   Shared memory is fast on-chip memory accessible by all
;   threads in a block. Each thread writes tid*10 to
;   shared_mem[tid], then reads it back.
;   演示共享内存指令（SHST/SHLD）。共享内存是块内
;   所有线程可访问的快速片上存储。
;
; Expected Result / 预期结果:
;   shared_mem[tid] = tid * 10
;   mem[100+tid] = tid * 10   (从共享内存读回并写到全局内存)
;
; Key Concepts / 关键概念:
;   - SHST (shared store): write register to shared memory
;   - SHLD (shared load): read shared memory into register
;   - Shared memory is lower latency than global memory
;   - Per-thread addressing: [0] means [0+tid]
;   - 共享内存比全局内存延迟更低，类似 GPU 的 shared memory
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1           ← r1 = tid
;   MOV r2, 10
;   MUL r3, r1, r2   ← r3 = tid * 10
;     |
;     v
;   SHST r3, [0]     ← shared_mem[tid] = tid*10  (写入共享内存)
;     |
;     v
;   SHLD r4, [0]     ← r4 = shared_mem[tid]     (读回共享内存)
;     |
;     v
;   ST r4, [100]     ← mem[100+tid] = tid*10    (写入全局内存)
;     |
;     v
;   HALT
; ============================================================

; 计算每个线程要存储的值 / Compute per-thread value
TID r1           ; r1 = tid
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = tid * 10

; 将值写入共享内存 / Write to shared memory
; SHST 使用与全局 ST 相同的基址+偏移模式
; SHST uses the same base+offset addressing as global ST
SHST r3, [0]     ; shared_mem[tid] = tid * 10

; 从共享内存读回 / Read back from shared memory
; SHLD 从共享内存加载到寄存器
; SHLD loads from shared memory into a register
SHLD r4, [0]     ; r4 = shared_mem[tid] = tid * 10

; 将验证后的值写入全局内存 / Write verified value to global memory
; 这确认共享内存操作正确
; This confirms shared memory operates correctly
ST r4, [100]     ; mem[100 + tid] = tid * 10

; 程序终止 / Terminate program
HALT
