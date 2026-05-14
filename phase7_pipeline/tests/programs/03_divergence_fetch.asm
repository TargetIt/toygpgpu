; ============================================================
; Phase 7: Pipeline — Divergence + I-Buffer Flush
; 阶段 7：流水线 — 分支发散与指令缓冲刷新
;
; Purpose / 目的:
;   Demonstrate the interaction between SIMT divergence and
;   the I-Buffer pipeline. When threads diverge, the I-Buffer
;   must handle multiple instruction streams and flush when
;   switching between divergent paths.
;   演示 SIMT 发散与 I-Buffer 流水线的交互。当线程发散时，
;   I-Buffer 必须处理多个指令流并在切换发散路径时刷新。
;
; Same logic as Phase 3 divergence test:
; Even threads (tid%2==0): mem[100+tid] = 2
; Odd threads  (tid%2==1): mem[100+tid] = 1
; All:                      mem[200+tid] = tid
;
; Expected Result / 预期结果:
;   mem[100+tid] = 2 (even) / 1 (odd)
;   mem[200+tid] = tid
;
; Key Concepts / 关键概念:
;   - I-Buffer must flush when warp switches execution paths
;   - Divergent branches cause I-Buffer refill
;   - The pipeline must handle both taken and not-taken paths
;   - 发散分支导致 I-Buffer 刷新和重新填充
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1
;   MOV r2, 2
;   DIV r3, r1, r2
;   MUL r4, r3, r2       ← r4 = (tid/2)*2
;     |
;     v
;   BEQ r4, r1, even_path  ─── [even] ──┐
;     |                                    |
;     | [odd]                              |
;     v                                    v
;   MOV r6, 1                            MOV r6, 2
;   ST r6, [100]                         ST r6, [100]
;   JMP reconv                           JMP reconv
;     |                                    |
;     +----- reconv: ←─────────────────────+
;              |
;              v
;   TID r7
;   ST r7, [200]
;     |
;     v
;   HALT
; ============================================================

; 计算线程奇偶性 / Determine even/odd
TID r1           ; r1 = tid
MOV r2, 2        ; r2 = 2
DIV r3, r1, r2   ; r3 = tid / 2
MUL r4, r3, r2   ; r4 = (tid / 2) * 2

; 条件分支: 偶线程 → even_path
; I-Buffer 在发散处可能需要多次刷新
; I-Buffer may need flush at divergence point
BEQ r4, r1, even_path

; === Odd path / 奇线程路径 ===
MOV r6, 1        ; r6 = 1
ST r6, [100]     ; mem[100 + tid] = 1
JMP reconv       ; 跳转到汇聚点

; === Even path / 偶线程路径 ===
even_path:
MOV r6, 2        ; r6 = 2
ST r6, [100]     ; mem[100 + tid] = 2
JMP reconv       ; 跳转到汇聚点

; === Reconvergence / 汇聚点 ===
reconv:
TID r7           ; r7 = tid
ST r7, [200]     ; mem[200 + tid] = tid

; 程序终止 / Terminate program
HALT
