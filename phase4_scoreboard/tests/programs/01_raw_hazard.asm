; ============================================================
; Phase 4: Scoreboard — RAW Hazard Detection
; 阶段 4：记分板 — RAW（写后读）冒险检测
;
; Purpose / 目的:
;   Demonstrate RAW (Read After Write) hazard detection.
;   Instruction ADD reads r1 and r2 which were just written
;   by previous MOVs. The scoreboard stalls ADD until the
;   MOV results are available.
;   演示 RAW（写后读）冒险检测。ADD 指令读取刚刚由
;   前一条 MOV 写入的 r1 和 r2。记分板会停顿 ADD
;   直到 MOV 的结果可用。
;
; Expected Result / 预期结果:
;   mem[0] = 15  (10 + 5)
;
; Key Concepts / 关键概念:
;   - RAW hazard: instruction reads a register before previous
;     write completes
;   - Scoreboard: hardware mechanism that detects hazards and
;     stalls the pipeline until safe
;   - Without scoreboard: r3 could get stale/incorrect value
;   - 记分板确保正确性：检测 RAW 冒险并插入停顿
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10      ← 写入 r1 (Write r1)
;   MOV r2, 5       ← 写入 r2 (Write r2)
;     |                 r1, r2 还在写回阶段
;     |                 (still in writeback stage)
;     v
;   ADD r3, r1, r2  ← RAW 冒险! 读取正在被写入的 r1, r2
;     |                (RAW hazard! Reads r1,r2 being written)
;     |                记分板检测到并插入停顿
;     |                (Scoreboard detects and stalls)
;     v
;   ST r3, [0]      ← mem[0] = 15 (结果正确 / correct result)
;     |
;     v
;   HALT
; ============================================================

; --- Write operands / 写入操作数 ---
MOV r1, 10       ; 写入 r1 = 10  (指令 1: 写 r1)
MOV r2, 5        ; 写入 r2 = 5   (指令 2: 写 r2)

; --- RAW hazard: r1 and r2 have pending writes! ---
; ADD 需要 r1 和 r2 的值, 但它们还在流水线中
; ADD needs r1 and r2, but those writes are still in the pipeline
; 记分板检测到 RAW 依赖: r1 和 r2 ← 之前的 MOV 尚未完成
; Scoreboard detects RAW: r1 and r2 ← previous MOV not yet retired
; 流水线停顿直到 MOV 的结果可用
; Pipeline stalls until MOV results are available
ADD r3, r1, r2   ; r3 = 10 + 5 = 15

; --- Store result / 存储结果 ---
ST r3, [0]       ; mem[0] = 15

; 程序终止 / Terminate program
HALT
