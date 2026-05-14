; ============================================================
; Phase 4: Scoreboard — No Hazard (Independent Registers)
; 阶段 4：记分板 — 无冒险（独立寄存器）
;
; Purpose / 目的:
;   Demonstrate a program with NO data hazards. All
;   registers are independent, so the scoreboard does
;   NOT stall the pipeline. This shows zero-overhead
;   operation of the scoreboard.
;   演示没有数据冒险的程序。所有寄存器独立，记分板
;   不会停顿流水线。展示记分板的零开销操作。
;
; Expected Result / 预期结果:
;   mem[0] = 60  (30 + 30)
;
; Key Concepts / 关键概念:
;   - No register dependencies → no stalls needed
;   - r1, r2, r3 are each written by distinct MOVs
;   - r4 depends on r1, r2 (independent of r3)
;   - r5 depends on r3, r4 — but r3 and r4 are both ready
;   - 记分板在无冒险时不会引入额外开销
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10      ← 写入 r1, 与其他写入无冲突
;   MOV r2, 20      ← 写入 r2, 独立
;   MOV r3, 30      ← 写入 r3, 独立
;     |
;     v
;   ADD r4, r1, r2  ← r1 和 r2 都已就绪, 无冒险
;     |                (both r1, r2 are ready)
;     v
;   ADD r5, r3, r4  ← r3 和 r4 都已就绪, 无冒险
;     |                (both r3, r4 are ready)
;     v
;   ST r5, [0]      ← mem[0] = 60
;     |
;     v
;   HALT
; ============================================================

; --- Phase 1: Three independent writes / 三个独立的写操作 ---
; 每个 MOV 写入不同的寄存器, 没有 WAW 冲突
; Each MOV writes a different register — no WAW hazard
MOV r1, 10       ; r1 = 10  (独立 / independent)
MOV r2, 20       ; r2 = 20  (独立 / independent)
MOV r3, 30       ; r3 = 30  (独立 / independent)

; --- Phase 2: Independent reads / 独立的读操作 ---
; r1 和 r2 都已经就绪, ADD 不需要停顿
; r1 and r2 are both ready — ADD proceeds without stall
ADD r4, r1, r2   ; r4 = 10 + 20 = 30

; --- Phase 3: Chain read on ready registers / 链式读取已就绪寄存器 ---
; r3 和 r4 都已经就绪, 无冒险
; r3 and r4 are both ready — no hazard
ADD r5, r3, r4   ; r5 = 30 + 30 = 60

; --- Store final result / 存储最终结果 ---
ST r5, [0]       ; mem[0] = 60

; 程序终止 / Terminate program
HALT
