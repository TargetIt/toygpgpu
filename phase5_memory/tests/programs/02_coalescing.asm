; ============================================================
; Phase 5: Memory Subsystem — Memory Coalescing
; 阶段 5：内存子系统 — 内存合并访问
;
; Purpose / 目的:
;   Demonstrate coalesced memory access pattern. All threads
;   access consecutive global memory addresses, allowing
;   the memory controller to combine accesses into fewer
;   transactions.
;   演示合并内存访问模式。所有线程访问连续的全局内存
;   地址，使内存控制器能将多个访问合并为更少的事务。
;
; Expected Result / 预期结果:
;   After storing:  mem[50..57] = [0, 10, 20, 30, 40, 50, 60, 70]
;   After loading:  mem[200..207] = same as above (read back)
;
; Key Concepts / 关键概念:
;   - Coalescing: consecutive thread accesses consecutive addresses
;   - Coalesced access = efficient memory bandwidth utilization
;   - Non-coalesced (strided) access = wasted bandwidth
;   - Thread i accesses address [base + i] = perfect coalescing
;   - 合并访存是 GPU 高性能的关键之一
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   TID r1           ← r1 = tid
;   MOV r2, 10
;     |
;     v
;   MUL r3, r1, r2   ← r3 = tid * 10
;     |
;     v
;   ST r3, [50]      ← mem[50+tid] = tid*10 (连续访问!)
;     |                 (Coalesced store / 合并写入)
;     v
;   LD r4, [50]      ← r4 = mem[50+tid]    (连续读取!)
;     |                 (Coalesced load / 合并读取)
;     v
;   ST r4, [200]     ← mem[200+tid] = r4   (验证 / verify)
;     |
;     v
;   HALT
;
; Access pattern / 访问模式:
;   Thread 0 → addr 50
;   Thread 1 → addr 51
;   Thread 2 → addr 52
;   ... (all consecutive / 全部连续)
; ============================================================

; 计算每个线程的值 / Compute per-thread value
TID r1           ; r1 = tid
MOV r2, 10       ; r2 = 10
MUL r3, r1, r2   ; r3 = tid * 10

; 合并存储: 所有线程访问连续的地址
; Coalesced store: all threads access consecutive addresses
; 内存控制器可将这些合并为一次宽事务
; Memory controller can merge these into one wide transaction
ST r3, [50]      ; mem[50 + tid] = tid * 10

; 合并加载: 从同样的连续地址读回
; Coalesced load: read back from the same consecutive addresses
LD r4, [50]      ; r4 = mem[50 + tid]

; 将读回的值存储到验证区 / Store read-back to verification area
ST r4, [200]     ; mem[200 + tid] = r4  (验证数据完整性)

; 程序终止 / Terminate program
HALT
