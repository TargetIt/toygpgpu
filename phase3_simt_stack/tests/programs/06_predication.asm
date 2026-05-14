; ============================================================
; Phase 3: SIMT Stack Core — Predicated Execution (PRED)
; 阶段 3：SIMT 栈核心 — 谓词执行
;
; Purpose / 目的:
;   Demonstrate predication as an alternative to divergent
;   branching. SETP sets a predicate flag; @p0 conditionally
;   enables/disables instruction execution WITHOUT splitting
;   the warp. This avoids SIMT stack overhead.
;   演示谓词执行作为分支发散的替代方案。SETP 设置谓词
;   标志；@p0 条件性地启用/禁用指令执行，而无需分裂
;   warp，避免了 SIMT 栈开销。
;
; Even threads (tid%2==0) write to mem[100+tid],
; odd threads skip. All threads remain lockstep.
; 偶线程写入 mem[100+tid]，奇线程跳过。所有线程保持同步。
;
; Expected Result (warp_size=8) / 预期结果:
;   Even threads: mem[100+0]=100, mem[100+2]=100, ...
;   Odd threads:  mem[101], [103], ... remain uninitialized (0)
;   All:          mem[200+tid] = tid + 10
;
; Key Concepts / 关键概念:
;   - SETP (set predicate): compare registers, set pred flag
;   - @p0: predicated instruction prefix (gate)
;   - Predication keeps warp intact (no divergence)
;   - Advantage: no SIMT stack push/pop, no serialization
;   - Disadvantage: both paths are fetched; disabled lanes
;     still consume issue cycles but their writes are masked
;   - 谓词执行：warp 不分裂，但两个路径的指令都被取指
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
;   SUB r5, r1, r4       ← r5 = tid % 2 (奇偶性)
;     |
;     v
;   SETP.EQ r5, r0       ← 如果 r5==0, pred=true (偶线程)
;     |
;     v
;   @p0 MOV r6, 100      ← 仅偶线程执行
;   @p0 ST r6, [100]     ← 仅偶线程写入
;     |
;     v
;   TID r7
;   MOV r8, 10
;   ADD r7, r7, r8       ← r7 = tid + 10 (所有线程)
;   ST r7, [200]         ← mem[200+tid] = tid+10
;     |
;     v
;   HALT
;
; Compare with 03_divergence.asm — same logic but no stack!
; 与 03_divergence.asm 对比 — 相同逻辑但无需 SIMT 栈!
; ============================================================

; === Step 1: Compute tid % 2 (predicate condition / 计算谓词条件) ===
; 我们想区分奇偶线程: 如果 tid%2==0 则谓词为真
; We want to distinguish even/odd threads
TID r1           ; r1 = tid
MOV r2, 2        ; r2 = 2
DIV r3, r1, r2   ; r3 = tid / 2
MUL r4, r3, r2   ; r4 = (tid / 2) * 2
SUB r5, r1, r4   ; r5 = tid % 2 (0=偶/even, 1=奇/odd)

; === Step 2: Set predicate register / 设置谓词寄存器 ===
; SETP.EQ 比较 r5 和 r0 (恒为 0), 若相等则 pred=true
; SETP.EQ compares r5 and r0 (always 0), sets pred if equal
; pred 为真 → 偶线程 (tid%2==0)
; pred is true → even threads
SETP.EQ r5, r0   ; pred = (r5 == r0) = (tid % 2 == 0)

; === Step 3: Predicated execution / 谓词化执行 ===
; @p0 前缀: 仅当 pred 为真时执行该指令
; @p0 prefix: execute only when pred is true
; warp 在此不分裂, 所有线程保持同步!
; Warp does NOT split here; all threads stay in lockstep!
@p0 MOV r6, 100  ; 偶线程: r6 = 100; 奇线程: 被屏蔽 (masked)
@p0 ST r6, [100] ; 偶线程: mem[100+tid] = 100; 奇线程: 不写入

; === Step 4: All threads continue together / 所有线程共同执行 ===
; 所有线程在这之后都执行, 无分歧
; All threads execute from here, no divergence
TID r7           ; r7 = tid
MOV r8, 10       ; r8 = 10
ADD r7, r7, r8   ; r7 = tid + 10
ST r7, [200]     ; mem[200 + tid] = tid + 10

; 程序终止 / Terminate program
HALT
