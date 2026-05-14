; ============================================================
; Phase 3: SIMT Stack Core — Warp Divergence (if/else)
; 阶段 3：SIMT 栈核心 — Warp 发散（if/else）
;
; Purpose / 目的:
;   Demonstrate SIMT divergence handling via the stack.
;   Even threads (tid%2==0) take the "even" path,
;   odd threads (tid%2==1) take the "odd" path,
;   then both reconverge at the shared label.
;   演示通过 SIMT 栈处理 warp 发散。偶线程走 even
;   路径，奇线程走 odd 路径，然后在汇聚点汇合。
;
; Expected Result / 预期结果:
;   Even threads: mem[200+tid] = 2
;   Odd threads:  mem[200+tid] = 1
;   All threads:  mem[300+tid] = tid
;
; Key Concepts / 关键概念:
;   - Divergence: threads in same warp take different branches
;   - SIMT stack tracks: which threads are active, PC for each path
;   - Reconvergence: after JMP, divergent paths merge
;   - The stack-based reconvergence is an optimization over
;     simply serializing all paths
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1
;   MOV r2, 2
;   DIV r3, r1, r2
;   MUL r4, r3, r2    ← r4 = (tid/2)*2 (偶数则 == tid)
;     |
;     v
;   BEQ r4, r1, even_path
;     |                    |
;     | [odd path]         | [even path]
;     v                    v
;   MOV r6, 1            MOV r6, 2
;   ST r6, [200]         ST r6, [200]
;     |                    |
;     +--- JMP reconv ----+--- JMP reconv
;     |                    |
;     v                    v
;   reconv:  ←───── reconvergence point / 汇聚点
;     |
;     v
;   TID r7
;   ST r7, [300]         ← all threads execute this
;     |
;     v
;   HALT
;
; Thread behavior / 线程行为:
;   tid=0 (even): r4=0, r1=0 → equal → even path → mem[200]=2
;   tid=1 (odd):  r4=0, r1=1 → not equal → odd path → mem[201]=1
;   tid=2 (even): r4=2, r1=2 → equal → even path → mem[202]=2
; ============================================================

; 计算 tid 的奇偶性 / Determine if tid is even or odd
TID r1           ; r1 = tid
MOV r2, 2        ; r2 = 2
DIV r3, r1, r2   ; r3 = tid / 2
MUL r4, r3, r2   ; r4 = (tid / 2) * 2  (如果 tid 为偶, r4 == tid)

; 条件分支: 偶线程走 even_path, 奇线程走 odd_path
; Conditional branch: even threads → even_path, odd threads → odd_path
BEQ r4, r1, even_path  ; r4 == r1 → tid 是偶数

; === Odd path / 奇线程路径 (tid % 2 == 1) ===
; 只有奇线程执行这里
; Only odd threads execute this path
MOV r6, 1        ; r6 = 1 (奇线程的标记)
ST r6, [200]     ; mem[200 + tid] = 1
JMP reconv       ; 跳转到汇聚点 / Jump to reconvergence point

; === Even path / 偶线程路径 (tid % 2 == 0) ===
even_path:
; 只有偶线程执行这里
; Only even threads execute this path
MOV r6, 2        ; r6 = 2 (偶线程的标记)
ST r6, [200]     ; mem[200 + tid] = 2
JMP reconv       ; 跳转到汇聚点 / Jump to reconvergence point

; === Reconvergence point / 汇聚点 ===
; 所有线程在此汇合后继续执行
; All threads reconverge here and continue together
reconv:
TID r7           ; r7 = tid
ST r7, [300]     ; mem[300 + tid] = tid  (所有线程都执行 / all threads)

; 程序终止 / Terminate program
HALT
