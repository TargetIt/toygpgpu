; ============================================================
; Phase 7: Pipeline — Branch + I-Buffer Interaction
; 阶段 7：流水线 — 分支与指令缓冲的交互
;
; Purpose / 目的:
;   Demonstrate how conditional branches interact with the
;   I-Buffer. When a branch is not taken, the I-Buffer
;   continues sequential prefetch. When taken, the I-Buffer
;   must flush and fetch from the branch target.
;   演示条件分支如何与指令缓冲交互。分支不跳转时，
;   I-Buffer 继续顺序预取；跳转时，I-Buffer 必须刷新
;   并从目标地址重新取指。
;
; Condition: r1(10) != r2(5) → branch NOT taken, fall through.
; 条件: r1(10) != r2(5) → 不跳转, 顺序执行。
;
; Expected Result / 预期结果:
;   mem[0] = 1   (BEQ 未跳转, 顺序执行)
;   mem[10] = 42 (done 处的最终存储)
;   mem[1] 保持未初始化 (skip 路径未执行)
;
; Key Concepts / 关键概念:
;   - I-Buffer must handle branch target changes
;   - Not-taken branch: I-Buffer continues normally
;   - Taken branch: I-Buffer flushed, refilled from target
;   - Branch prediction could optimize this (future work)
;   - I-Buffer 必须正确处理分支目标变更
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10     ← r1 = 10
;   MOV r2, 5      ← r2 = 5
;     |
;     v
;   BEQ r1, r2, skip  ← 10 != 5 → NOT taken (不跳转)
;     |
;     v  (fall through / 顺序执行)
;   MOV r3, 1
;   ST r3, [0]        ← mem[0] = 1
;   JMP done
;     |
;     v
;   skip:             ← skipped (BEQ condition was false)
;   MOV r3, 2         ← NOT executed
;   ST r3, [1]        ← NOT executed
;     |
;     v
;   done:
;   MOV r4, 42
;   ST r4, [10]       ← mem[10] = 42
;     |
;     v
;   HALT
; ============================================================

; 设置条件 / Set up branch condition
MOV r1, 10       ; r1 = 10
MOV r2, 5        ; r2 = 5

; 条件分支: 如果 r1 == r2 则跳转到 skip
; I-Buffer 在此处可能需要刷新
; I-Buffer may need to flush here if branch is taken
; (10 != 5, 所以不跳转 / not taken)
BEQ r1, r2, skip ; 10 == 5? false → fall through

; === Fall-through path (taken when BEQ condition is false) ===
MOV r3, 1        ; r3 = 1
ST r3, [0]       ; mem[0] = 1
JMP done         ; 跳转到 done

; === Branch target (taken when BEQ condition is true) ===
skip:
MOV r3, 2        ; (被跳过 / skipped when 10!=5)
ST r3, [1]       ; (被跳过)

; === Reconvergence / 汇聚点 ===
done:
MOV r4, 42       ; r4 = 42
ST r4, [10]      ; mem[10] = 42

; 程序终止 / Terminate program
HALT
