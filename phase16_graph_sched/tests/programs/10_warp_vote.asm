; ============================================================
; Phase 12: Warp Communication — VOTE & BALLOT
; 阶段 12：Warp 通信 — 投票与选票指令
;
; Purpose / 目的:
;   Demonstrate VOTE (any/all) and BALLOT warp-level primitives.
;   VOTE.ANY checks if ANY thread has non-zero value.
;   VOTE.ALL checks if ALL threads have non-zero value.
;   BALLOT creates a bitmask of threads with non-zero value.
;   演示 VOTE(任意/全部) 和 BALLOT warp 级原语。
;
; Scenario / 场景 (warp_size=8):
;   Step 1: Even threads get value 1, odd threads get 0
;   Step 2: VOTE.ANY on all → 1 (some non-zero)
;   Step 3: VOTE.ALL on all → 0 (not all non-zero)
;   Step 4: BALLOT on all → 0b01010101 = 0x55 (even threads)
;
; Expected Results / 预期结果:
;   mem[0]  = 1  (VOTE.ANY: 有非零线程 → true)
;   mem[1]  = 0  (VOTE.ALL: 不全是非零 → false)
;   mem[2]  = 0x55 = 85 (BALLOT: 偶线程 bitmask)
;   mem[8]  = 1  (VOTE.ANY on only non-zero regs → true)
;   mem[9]  = 0  (VOTE.ALL on only non-zero → false; some are 0)
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1
;   MOV r2, 2
;   DIV r3, r1, r2
;   MUL r4, r3, r2
;   SUB r5, r1, r4       ← r5 = tid % 2 (0=even, 1=odd)
;     |
;     v
;   MOV r6, 0            ← 默认值 0
;   SETP.EQ r5, r0       ← 偶线程 pred=true
;     |
;     v
;   @p0 MOV r6, 1        ← 偶线程: r6=1, 奇线程: r6=0
;     |
;     +----+----+----+
;     |         |    |
;     v         v    v
;   VOTE.ANY  VOTE.ALL  BALLOT
;   r7, r6   r8, r6    r9, r6
;     |         |    |
;     v         v    v
;   ST r7,[0] ST r8,[1] ST r9,[2]
;     |
;     v
;   HALT
; ============================================================

; Step 1: Compute tid % 2 (even=0, odd=1)
TID r1                   ; r1 = tid
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2
SUB r5, r1, r4           ; r5 = tid % 2

; Step 2: Even threads set r6=1, odd threads keep r6=0
MOV r6, 0                ; all threads: r6 = 0
SETP.EQ r5, r0           ; even threads: pred=true (r5==r0==0)
@p0 MOV r6, 1            ; only even threads: r6 = 1

; Step 3: VOTE.ANY — any thread have non-zero r6?
VOTE.ANY r7, r6          ; r7 = 1 (even threads have r6=1)
ST r7, [0]               ; mem[0] = 1

; Step 4: VOTE.ALL — all threads have non-zero r6?
VOTE.ALL r8, r6          ; r8 = 0 (odd threads have r6=0)
ST r8, [1]               ; mem[1] = 0

; Step 5: BALLOT — bitmask of threads with non-zero r6
BALLOT r9, r6            ; r9 = 0b01010101 = 0x55 (warp_size=8)
ST r9, [2]               ; mem[2] = 0x55

; Step 6: Verify with all-threads-nonzero (r10=1 everywhere)
MOV r10, 1               ; all threads: r10 = 1
VOTE.ANY r11, r10        ; r11 = 1 (all non-zero → any is true)
ST r11, [8]              ; mem[8] = 1
VOTE.ALL r12, r10        ; r12 = 1 (all non-zero → all is true)
ST r12, [9]              ; mem[9] = 1

HALT
