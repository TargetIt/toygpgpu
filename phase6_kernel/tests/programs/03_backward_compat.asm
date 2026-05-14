; ============================================================
; Phase 6: Kernel Launch — Full Backward Compatibility
; 阶段 6：内核启动 — 完整的向后兼容性测试
;
; Purpose / 目的:
;   Comprehensive backward compatibility test covering all
;   features from Phases 0-5: scalar ops, TID/WID, branching,
;   scoreboard, shared memory.
;   综合性向后兼容测试，覆盖 Phase 0-5 的所有特性。
;
; Expected Result (per thread) / 预期结果:
;   mem[0] = tid * 10
;   tid even: mem[100+tid] = 2
;   tid odd:  mem[100+tid] = 1
;   mem[200+tid] = shared_mem[tid] (read back)
;
; Key Concepts / 关键概念:
;   - Kernel launch must preserve all prior functionality
;   - BEQ/JMP divergence still works across blocks
;   - Shared memory (SHLD) works inside kernel
;   - 保证整个架构的一致性
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1
;   MOV r2, 10
;   MUL r3, r1, r2  ← r3 = tid * 10
;   ST r3, [0]      ← mem[0+tid] = tid*10
;     |
;     v
;   奇偶判定 (tid % 2 == 0?)
;   DIV r4, r1, r2  → MUL r4, r4, r2
;   BEQ r4, r1, even
;     |                    |
;     | [odd]              | [even]
;     v                    v
;   MOV r5, 1            MOV r5, 2
;   ST r5, [100]         ST r5, [100]
;     |                    |
;     +--- JMP done -------+
;     |
;     v
;   done:
;   SHLD r6, [0]     ← r6 = shared_mem[tid]
;   ST r6, [200]     ← mem[200+tid] = r6
;     |
;     v
;   HALT
; ============================================================

; --- Part 1: Scalar arithmetic / 标量算术 ---
TID r1           ; r1 = tid
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = tid * 10
ST r3, [0]       ; mem[0 + tid] = tid * 10

; --- Part 2: Even/odd divergence / 奇偶分支发散 ---
MOV r2, 2        ; r2 = 2
DIV r4, r1, r2   ; r4 = tid / 2
MUL r4, r4, r2   ; r4 = (tid/2) * 2

; 条件分支: 偶线程 → even; 奇线程 → 顺序执行
; Conditional: even → even; odd → fall through
BEQ r4, r1, even

; Odd path / 奇线程路径
MOV r5, 1        ; r5 = 1
ST r5, [100]     ; mem[100+tid] = 1
JMP done         ; 跳转到汇聚点

; Even path / 偶线程路径
even:
MOV r5, 2        ; r5 = 2
ST r5, [100]     ; mem[100+tid] = 2
JMP done         ; 跳转到汇聚点

; --- Part 3: Shared memory read / 共享内存读取 ---
done:
SHLD r6, [0]     ; r6 = shared_mem[tid]
ST r6, [200]     ; mem[200+tid] = r6

; 程序终止 / Terminate program
HALT
