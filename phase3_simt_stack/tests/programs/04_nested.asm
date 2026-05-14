; ============================================================
; Phase 3: SIMT Stack Core — Nested Conditional Divergence
; 阶段 3：SIMT 栈核心 — 嵌套条件发散
;
; Purpose / 目的:
;   Demonstrate nested divergence: first branch on tid < 4,
;   then a secondary computation after reconvergence.
;   演示嵌套发散：先按 tid < 4 分支，汇聚后再进行计算。
;
; tid 0-3 (low):  mem[100+tid] = 1
; tid 4-7 (high): mem[100+tid] = 0
; All:            mem[200+tid] = tid
;
; Expected Result / 预期结果:
;   mem[100..103] = 1,  mem[104..107] = 0
;   mem[200..207] = [0, 1, 2, 3, 4, 5, 6, 7]
;
; Key Concepts / 关键概念:
;   - Nesting: one divergence inside another (or sequential divergence)
;   - SIMT stack depth grows with nesting level
;   - Each divergence pushes a new entry on the SIMT stack
;   - Reconvergence pops the stack
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1
;   MOV r2, 4
;   DIV r4, r1, r2    ← r4 = tid/4
;   MOV r5, 0
;     |
;     v
;   BEQ r4, r5, low_tid  ── r4==0 ── [tid<4] ──┐
;     |                                           │
;     | [tid>=4: high]                            │
;     v                                           v
;   MOV r6, 0                                    MOV r6, 1
;   ST r6, [100]                                 ST r6, [100]
;     |                                           |
;     +--- JMP after ────────────────+--- JMP after
;     |                               |
;     v                               v
;   after:  ←────── reconvergence point / 汇聚点
;     |
;     v
;   TID r7
;   ST r7, [200]      ← all threads
;     |
;     v
;   HALT
;
; Thread behavior / 线程行为:
;   tid=0:  tid/4=0 → low_tid → mem[100]=1, mem[200]=0
;   tid=4:  tid/4=1 → high    → mem[104]=0, mem[204]=4
; ============================================================

; 计算 tid < 4 的条件 / Compute condition tid < 4
TID r1           ; r1 = tid
MOV r2, 4        ; r2 = 4
DIV r4, r1, r2   ; r4 = tid / 4 (tid 0-3 → 0; tid 4-7 → 1)
MOV r5, 0        ; r5 = 0 (比较值 / comparison value)

; 分支: 如果 r4 == 0 (tid < 4), 跳转到 low_tid
; Branch: if r4 == 0 (tid < 4), jump to low_tid
BEQ r4, r5, low_tid

; === High path: tid >= 4 (tid 4 到 7) ===
; 高位线程执行此路径
MOV r6, 0        ; r6 = 0 (高位线程写入值)
ST r6, [100]     ; mem[100 + tid] = 0
JMP after        ; 跳转到汇聚点

; === Low path: tid < 4 (tid 0 到 3) ===
low_tid:
MOV r6, 1        ; r6 = 1 (低位线程写入值)
ST r6, [100]     ; mem[100 + tid] = 1
JMP after        ; 跳转到汇聚点

; === Reconvergence point / 汇聚点 ===
after:
TID r7           ; r7 = tid
ST r7, [200]     ; mem[200 + tid] = tid

; 程序终止 / Terminate program
HALT
