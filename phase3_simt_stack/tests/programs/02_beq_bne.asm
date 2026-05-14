; ============================================================
; Phase 3: SIMT Stack Core — Conditional Branch (BEQ)
; 阶段 3：SIMT 栈核心 — 条件分支
;
; Purpose / 目的:
;   Demonstrate BEQ (Branch if Equal) instruction.
;   When r1 == r2, control transfers to the label.
;   When not equal, execution continues sequentially.
;   演示 BEQ（相等则分支）指令。当 r1 == r2 时
;   跳转到标签；否则顺序执行。
;
; Expected Result / 预期结果:
;   mem[1] = 42  (BEQ 条件为真, 跳转到 equal 标签)
;   mem[0] 不会被写入 (因为它跟在 BEQ 之后, 被跳过)
;
; Key Concepts / 关键概念:
;   - BEQ (branch if equal): BEQ rA, rB, label → if rA == rB, jump
;   - BNE (branch if not equal): complementary conditional (not shown here)
;   - Conditional branches can cause SIMT divergence (different threads
;     may take different paths)
;   - 条件分支可能引起 SIMT warp 发散
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 5
;   MOV r2, 5       ← r1 == r2 (条件为真 / condition true)
;   MOV r3, 10
;     |
;     v
;   BEQ r1, r2, equal  ──────────── 条件成立 ──┐
;     |  (r1 == r2 → taken)                    │
;     v                                         │
;   ST r3, [0]  ← 跳过 (不执行)                 │
;   HALT       ← 跳过 (不执行)                  │
;     |                                         │
;     v                                         │
;   equal:  ←───────────────────────────────────┘
;     |
;     v
;   MOV r4, 42
;   ST r4, [1]        ← mem[1] = 42
;     |
;     v
;   HALT
; ============================================================

; 设置条件 / Set up condition
MOV r1, 5        ; r1 = 5
MOV r2, 5        ; r2 = 5  (r1 == r2, 条件为真)
MOV r3, 10       ; r3 = 10 (用于未分支路径的值)

; 条件分支: 如果 r1 == r2, 跳转到 equal 标签
; Condition check: if r1 == r2, jump to "equal"
BEQ r1, r2, equal  ; r1(5) == r2(5) → 跳转 (taken)

; === 未分支路径 (条件为假时才执行) ===
; 由于条件为真, 这两条指令被跳过
; NOT executed because condition is true
ST r3, [0]       ; (跳过) mem[0] = 10
HALT             ; (跳过) 如果执行到这里程序就终止了

; === 分支路径目标 / Branch target ===
equal:
MOV r4, 42       ; r4 = 42
ST r4, [1]       ; mem[1] = 42

; 程序终止 / Terminate program
HALT
