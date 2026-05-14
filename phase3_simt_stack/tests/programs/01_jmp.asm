; ============================================================
; Phase 3: SIMT Stack Core — Unconditional Jump (JMP)
; 阶段 3：SIMT 栈核心 — 无条件跳转
;
; Purpose / 目的:
;   Demonstrate the JMP (unconditional jump) instruction.
;   JMP transfers control to a named label, skipping
;   any instructions in between.
;   演示 JMP（无条件跳转）指令。跳转到指定标签，
;   跳过中间的指令。
;
; Expected Result / 预期结果:
;   mem[0] = 10, mem[2] = 20
;   mem[1] remains uninitialized (0)
;   (The MOV r1, 99 and ST r1, [1] after JMP are skipped)
;
; Key Concepts / 关键概念:
;   - JMP: unconditional branch to label (always taken)
;   - Labels are symbolic names followed by ':'
;   - Instructions after JMP (before target) are NOT executed
;   - No SIMT divergence: ALL threads jump together
;   - 与后续的 BEQ/BNE 不同，JMP 不会引起 warp 发散
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10
;   ST r1, [0]        ← mem[0] = 10
;     |
;     v
;   JMP skip  ──────────────────────────────────┐
;     |                                          │
;   (skipped) MOV r1, 99   ← 跳过 / bypassed    │
;   (skipped) ST r1, [1]   ← 跳过 / bypassed    │
;     |                                          │
;     v                                          │
;   skip:  ←─────────────────────────────────────┘
;     |
;     v
;   MOV r1, 20
;   ST r1, [2]        ← mem[2] = 20
;     |
;     v
;   HALT
; ============================================================

; 跳转前的正常执行 / Normal execution before jump
MOV r1, 10       ; r1 = 10
ST r1, [0]       ; mem[0] = 10

; 无条件跳转到标签 "skip" / Unconditional jump to label "skip"
JMP skip

; === 以下指令被跳过 (NOT executed) / Instructions below are skipped ===
MOV r1, 99       ; 跳过了 — r1 保持为 10
ST r1, [1]       ; 跳过了 — mem[1] 不会被写入

; === 跳转目标标签 / Jump target label ===
skip:
MOV r1, 20       ; 从 JMP 直接跳到这里 / Execution resumes here after JMP
ST r1, [2]       ; mem[2] = 20

; 程序终止 / Terminate program
HALT
