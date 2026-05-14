; ============================================================
; Phase 7: Pipeline — I-Buffer Basic Operation
; 阶段 7：流水线 — 指令缓冲基础操作
;
; Purpose / 目的:
;   Demonstrate basic I-Buffer (instruction buffer) pipeline
;   operation. The I-Buffer prefetches instructions to hide
;   fetch latency. This simple scalar program runs through
;   the new pipeline stage.
;   演示指令缓冲 (I-Buffer) 流水线的基本操作。I-Buffer
;   预取指令以隐藏取指延迟。
;
; Formula / 公式:  r3 = 10 + 3 = 13, mem[0] = 13
;
; Expected Result / 预期结果:
;   mem[0] = 13
;
; Key Concepts / 关键概念:
;   - I-Buffer prefetches instructions ahead of execution
;   - Pipeline now has: Fetch → I-Buffer → Decode → Execute → Writeback
;   - I-Buffer helps hide fetch stalls
;   - 指令缓冲是现代处理器流水线的重要组成部分
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 10      ← I-Buffer 预取此指令
;   MOV r2, 3       ← I-Buffer 预取此指令
;     |
;     v
;   ADD r3, r1, r2  ← 流水线执行 / pipeline execution
;     |
;     v
;   ST r3, [0]      ← mem[0] = 13
;     |
;     v
;   HALT
; ============================================================

; 加载操作数 / Load operands
MOV r1, 10       ; r1 = 10
MOV r2, 3        ; r2 = 3

; 执行加法 / Perform addition
ADD r3, r1, r2   ; r3 = 10 + 3 = 13

; 存储结果 / Store result
ST r3, [0]       ; mem[0] = 13

; 程序终止 / Terminate program
HALT
