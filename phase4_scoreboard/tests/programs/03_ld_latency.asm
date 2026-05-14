; ============================================================
; Phase 4: Scoreboard — LD Pipeline Latency
; 阶段 4：记分板 — LD 流水线延迟
;
; Purpose / 目的:
;   Demonstrate LD instruction's multi-cycle latency (4 cycles).
;   A subsequent ADD that reads the loaded register triggers
;   a RAW hazard. The scoreboard stalls and retries until
;   the LD result is available.
;   演示 LD 指令的多周期延迟（4 个周期）。后续读取
;   加载寄存器的 ADD 会触发 RAW 冒险。记分板停顿并
;   重试直到 LD 结果可用。
;
; Expected Result / 预期结果:
;   mem[0] = 84  (42 + 42)
;   (LD r2, [50] loads 42; ADD doubles it)
;
; Key Concepts / 关键概念:
;   - LD takes 4 cycles to complete (memory access latency)
;   - RAW hazard on loaded register: ADD must stall
;   - Scoreboard retries: keeps checking if register is ready
;   - 这是内存延迟隐藏的关键测试
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 42
;   ST r1, [50]        ← mem[50] = 42
;     |
;     v
;   LD r2, [50]        ← r2 = mem[50] = 42 (4 周期延迟)
;     |                   (4 cycle latency — 开始加载)
;     |                  记分板标记 r2 为"忙" (pending)
;     v
;   ADD r3, r2, r2     ← RAW 冒险! r2 尚未就绪
;     |                  记分板停顿并重试 ADD
;     |                  直到 LD 完成, r2 可用
;     |                  (Scoreboard stalls until r2 ready)
;     v
;   ST r3, [0]         ← mem[0] = 84
;     |
;     v
;   HALT
; ============================================================

; --- Prepare data in memory / 准备内存中的数据 ---
MOV r1, 42       ; r1 = 42
ST r1, [50]      ; mem[50] = 42

; --- Load from memory (multi-cycle) / 从内存加载（多周期） ---
; LD 需要 4 个周期完成, 期间 r2 标记为"忙"
; LD takes 4 cycles to complete; r2 is marked "busy" during this time
LD r2, [50]      ; r2 = mem[50] = 42  (开始 4 周期加载 / start 4-cycle load)

; --- RAW hazard on loaded register / 加载寄存器上的 RAW 冒险 ---
; r2 还在加载中, 不能立即使用
; r2 is still being loaded, cannot be used immediately
; 记分板检测到 r2 忙 → 停顿 ADD 指令
; Scoreboard detects r2 is busy → stalls ADD instruction
; 流水线会重试 ADD, 直到 LD 完成写回
; Pipeline retries ADD until LD completes writeback
ADD r3, r2, r2   ; r3 = 42 + 42 = 84  (停顿后正确执行)

; --- Store final result / 存储最终结果 ---
ST r3, [0]       ; mem[0] = 84

; 程序终止 / Terminate program
HALT
