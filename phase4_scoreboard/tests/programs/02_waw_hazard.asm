; ============================================================
; Phase 4: Scoreboard — WAW Hazard Detection
; 阶段 4：记分板 — WAW（写后写）冒险检测
;
; Purpose / 目的:
;   Demonstrate WAW (Write After Write) hazard detection.
;   Two consecutive MOV instructions write to the same
;   register r1. The scoreboard ensures the second write
;   waits until the first write completes, so the final
;   value is correct (20, not 10).
;   演示 WAW（写后写）冒险检测。两条连续的 MOV 指令
;   写入同一个寄存器 r1。记分板确保第二次写入等待
;   第一次写入完成，因此最终值正确 (20 而不是 10)。
;
; Expected Result / 预期结果:
;   mem[0] = 50  (20 + 30)
;
; Key Concepts / 关键概念:
;   - WAW hazard: consecutive writes to same register
;   - Without scoreboard: r1 could end up with wrong value (10)
;   - Scoreboard serializes the writes to guarantee correctness
;   - 记分板正确处理寄存器重名的情况
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10      ← 第一次写入 r1 (first write to r1)
;     |
;     v
;   MOV r1, 20      ← WAW 冒险! 再次写入同一个 r1
;     |                (WAW hazard! Second write to r1)
;     |                记分板等待第一次写入完成
;     |                (Scoreboard waits for first write)
;     v
;   MOV r2, 30      ← 独立操作 (independent)
;     |
;     v
;   ADD r3, r1, r2  ← r3 = 20 + 30 = 50 (r1 取最新值)
;     |
;     v
;   ST r3, [0]      ← mem[0] = 50
;     |
;     v
;   HALT
; ============================================================

; --- First write to r1 / 第一次写入 r1 ---
MOV r1, 10       ; r1 = 10 (指令 1: 写入 r1)

; --- WAW hazard: second write to r1 / 第二次写入 r1 ---
; r1 在上一条指令中被写入但尚未完成
; r1 was written by previous instruction but not yet committed
; 记分板检测到 WAW: 必须等第一次写入完成
; Scoreboard detects WAW: must wait for first write to finish
MOV r1, 20       ; r1 = 20 (指令 2: 覆盖 r1)

; --- Independent write / 独立写入 ---
MOV r2, 30       ; r2 = 30 (与 r1 无关, 无冒险 / no hazard)

; --- Read r1 (gets the latest value 20) / 读取 r1 (得到最新值 20) ---
ADD r3, r1, r2   ; r3 = 20 + 30 = 50

; --- Store result / 存储结果 ---
ST r3, [0]       ; mem[0] = 50

; 程序终止 / Terminate program
HALT
