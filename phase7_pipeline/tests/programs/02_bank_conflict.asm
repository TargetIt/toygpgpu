; ============================================================
; Phase 7: Pipeline — Register Bank Conflict Detection
; 阶段 7：流水线 — 寄存器组冲突检测
;
; Purpose / 目的:
;   Demonstrate register bank conflicts in the pipeline.
;   Registers are organized into banks. When two source
;   registers are in the same bank, an extra cycle is needed
;   (bank conflict). When they're in different banks, no
;   extra cycle.
;   演示流水线中的寄存器组冲突。寄存器被组织为多个
;   组。当两个源寄存器在同一组时，需要额外周期（组冲突）。
;
; Expected Result / 预期结果:
;   mem[0] = 30   (10 + 20, with bank conflict)
;   mem[1] = 8    (5 + 3, no bank conflict)
;
; Key Concepts / 关键概念:
;   - Registers mapped to banks by register number
;   - Same-bank source operands → bank conflict → +1 cycle
;   - Different-bank source operands → no conflict
;   - Bank conflicts affect performance, not correctness
;   - 组冲突影响性能但不影响正确性
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   第一部分: bank 冲突示例 / Bank conflict example
;   MOV r1, 10      ← r1 → bank 1
;   MOV r5, 20      ← r5 → bank 1 (same bank / 同组!)
;     |
;     v
;   ADD r2, r1, r5  ← r1(bank1) + r5(bank1) → 冲突 +1 周期!
;   ST r2, [0]      ← mem[0] = 30
;     |
;     v
;   第二部分: 无冲突示例 / No conflict example
;   MOV r2, 5       ← r2 → bank 2
;   MOV r3, 3       ← r3 → bank 3
;     |
;     v
;   ADD r4, r2, r3  ← r2(bank2) + r3(bank3) → 无冲突!
;   ST r4, [1]      ← mem[1] = 8
;     |
;     v
;   HALT
;
; Bank mapping assumption / 寄存器组映射假定:
;   Register rX → bank (X mod num_banks)
;   r1, r5 both in bank1; r2 in bank2; r3 in bank3
; ============================================================

; === Part 1: Bank conflict example / 组冲突示例 ===
; r1 和 r5 在同一组 → ADD 会遇到 bank 冲突
; r1 and r5 are in the same bank → ADD will have a bank conflict
MOV r1, 10       ; r1 = 10  (→ bank 1)
MOV r5, 20       ; r5 = 20  (→ bank 1, same bank as r1!)

; r1 和 r5 同组, 读取它们需要多 1 个周期
; r1 and r5 share a bank; reading both costs an extra cycle
ADD r2, r1, r5   ; r2 = 10 + 20 = 30  (冲突 / conflict → +1 cycle)

; 存储冲突路径的结果 / Store result (conflict path)
ST r2, [0]       ; mem[0] = 30

; === Part 2: No conflict example / 无冲突示例 ===
; r2 和 r3 在不同组 → ADD 无冲突
; r2 and r3 are in different banks → ADD has no conflict
MOV r2, 5        ; r2 = 5   (→ bank 2)
MOV r3, 3        ; r3 = 3   (→ bank 3, different bank from r2!)

; r2 和 r3 不同组, 可以同时读取, 无额外延迟
; r2 and r3 are in different banks; reads proceed in parallel
ADD r4, r2, r3   ; r4 = 5 + 3 = 8  (无冲突 / no conflict)

; 存储无冲突路径的结果 / Store result (no-conflict path)
ST r4, [1]       ; mem[1] = 8

; 程序终止 / Terminate program
HALT
