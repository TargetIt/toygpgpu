; ============================================================
; Phase 11: Console Demo — Branch Divergence
; 阶段 11：控制台演示 — 分支发散
;
; Purpose / 目的:
;   A SIMT divergence demo ideal for the interactive console.
;   Even threads (tid%2==0) take the even_path, odd threads
;   take the odd path, then reconverge. Console users can
;   step through and observe warp splitting/reconvergence.
;   为交互式控制台设计的 SIMT 发散演示。偶线程走 even_path
;   路径，奇线程走 odd 路径，然后汇聚。用户可在控制台中
;   单步执行并观察 warp 分裂与汇聚。
;
; Expected Result / 预期结果:
;   mem[100+tid] = 2 (even) or 1 (odd)
;   mem[200+tid] = tid * 2 (all threads)
;
; Key Concepts / 关键概念:
;   - Console-friendly: clear labels, minimal instructions
;   - TID-based divergence with visible register/memory changes
;   - Shows both taken and not-taken paths in single-step mode
;   - Demonstrates SIMT stack reconvergence
;   - 控制台友好：清晰的标签、少量指令、可见的寄存器变化
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1           ← r1 = tid (0..7)
;   MOV r2, 2
;   DIV r3, r1, r2
;   MUL r4, r3, r2   ← r4 = (tid/2)*2 (偶数 == tid)
;   MOV r5, 0
;     |
;     v
;   BEQ r4, r1, even_path ── [even: r4==r1] ──┐
;     |                                           |
;     | [odd: r4!=r1]                             |
;     v                                           v
;   MOV r6, 1                                   MOV r6, 2
;   ST r6, [100]                                ST r6, [100]
;   JMP done                                    JMP done
;     |                                           |
;     +--- done: ←────────────────────────────────+
;              |
;              v
;   TID r7
;   ADD r7, r7, r7   ← r7 = tid * 2
;   ST r7, [200]     ← mem[200+tid] = tid*2
;     |
;     v
;   HALT
;
; 控制台观察点 / Console observation points:
;   1. BEQ 前: 检查 r1, r4, r5 的值 / Check r1, r4, r5 values
;   2. BEQ 后: 观察哪些线程走了哪条路径 / Observe divergence
;   3. done: 观察线程汇聚 / Observe reconvergence
; ============================================================

; === Part 1: Determine even/odd / 确定奇偶性 ===
TID r1           ; r1 = tid
MOV r2, 2        ; r2 = 2
DIV r3, r1, r2   ; r3 = tid / 2
MUL r4, r3, r2   ; r4 = (tid / 2) * 2
MOV r5, 0        ; r5 = 0

; === Part 2: Conditional branch / 条件分支 ===
; BEQ 导致 warp 发散: 偶线程跳转, 奇线程顺序执行
; BEQ causes warp divergence: even threads jump, odd fall through
BEQ r4, r1, even_path

; === Odd path / 奇线程路径 ===
; 仅奇线程执行 / Only odd threads execute
MOV r6, 1        ; r6 = 1 (奇线程标记)
ST r6, [100]     ; mem[100 + tid] = 1
JMP done         ; 跳转到汇聚点

; === Even path / 偶线程路径 ===
even_path:
; 仅偶线程执行 / Only even threads execute
MOV r6, 2        ; r6 = 2 (偶线程标记)
ST r6, [100]     ; mem[100 + tid] = 2
JMP done         ; 跳转到汇聚点

; === Reconvergence / 汇聚点 ===
done:
TID r7           ; r7 = tid
ADD r7, r7, r7   ; r7 = tid * 2
ST r7, [200]     ; mem[200 + tid] = tid * 2

; 程序终止 / Terminate program
HALT
