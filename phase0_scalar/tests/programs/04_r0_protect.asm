; ============================================================
; Phase 0: Scalar Core — r0 Write Protection
; 阶段 0：标量核心 — r0 写保护测试
;
; Purpose / 目的:
;   Verify that register r0 is hardwired to zero and
;   cannot be modified. In this toy GPU, r0 behaves
;   like $zero in MIPS RISC — writes are silently ignored.
;   验证寄存器 r0 被硬件固定为 0 且不可修改。
;   在本 toy GPU 中，r0 的行为类似 MIPS RISC 的 $zero
;   寄存器——写入操作被静默忽略。
;
; Expected Result / 预期结果:
;   mem[0] = 42
;   r1 = 42, r0 = 0 (write ignored), r2 = 42
;
; Key Concepts / 关键概念:
;   - r0 is hardwired to 0 (read-only / 只读)
;   - ADD r0, ... has no effect on r0's value
;   - Reading r0 always yields 0
;   - Important for ensuring architectural correctness
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 42      ← r1 = 42
;     |
;     v
;   ADD r0, r1, r1  ← 尝试写入 r0 (r0 = 42+42=84)
;     |                写入被忽略! r0 保持为 0
;     |                Write ignored! r0 stays 0
;     v
;   ADD r2, r0, r1  ← r2 = 0 + 42 = 42 (r0 始终读为 0)
;     |
;     v
;   ST r2, [0]      ← mem[0] = 42
;     |
;     v
;   HALT
; ============================================================

; 加载测试值 / Load test value
MOV r1, 42       ; r1 = 42

; 尝试向只读寄存器 r0 写入 / Attempt to write read-only r0
; 在正常 CPU 中这会使 r0 = 84
; 但在本架构中，r0 的写使能被硬件屏蔽
; In a normal CPU this would set r0 = 84,
; but in this architecture, r0's write-enable is masked by hardware
ADD r0, r1, r1   ; 尝试写 r0 = 84 → 无效 (ignored)
                   ; r0 保持为 0 (remains 0)

; 使用 r0 作为源操作数 / Use r0 as source operand
; r0 始终读回 0，所以 r2 = 0 + 42 = 42
; r0 always reads as 0, so r2 = 0 + 42 = 42
ADD r2, r0, r1   ; r2 = 0 + 42 = 42

; 验证结果 / Verify result
ST r2, [0]       ; mem[0] = 42

; 程序终止 / Terminate program
HALT
