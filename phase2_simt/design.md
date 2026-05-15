# Phase 2: SIMT Core — Design Document / 设计文档

> **对应 GPGPU-Sim**: shader_core_ctx (SM 顶层) + shd_warp_t (warp 状态) + scheduler_unit (warp 调度器)
> **参考**: GPGPU-Sim shader.h (shd_warp_t), shader.cc (scheduler_unit), CUDA 线程模型

## 1. Introduction / 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMT Core (simt_core.py)                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             Warp Scheduler (scheduler.py)                 │   │
│  │       Round-Robin: warp_0 → warp_1 → ... → warp_N        │   │
│  │       (GPGPU-Sim: scheduler_unit::cycle())               │   │
│  └───────────────────────────┬─────────────────────────────┘   │
│                              │ selected warp                    │
│                              ▼                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
│  │ Warp 0  │  │ Warp 1  │  │ Warp 2  │  ...                   │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │                        │
│  │ │Th 0 │ │  │ │Th 0 │ │  │ │Th 0 │ │                        │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │                        │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │                        │
│  │ │Th 1 │ │  │ │Th 1 │ │  │ │Th 1 │ │                        │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │                        │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │                        │
│  │ │ ... │ │  │ │ ... │ │  │ │ ... │ │                        │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │                        │
│  │ │Th 7 │ │  │ │Th 7 │ │  │ │Th 7 │ │                        │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │                        │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │                        │
│  │ 共享PC   │  │ 共享PC   │  │ 共享PC   │                        │
│  │ actmask │  │ actmask │  │ actmask │                        │
│  │ barrier │  │ barrier │  │ barrier │                        │
│  └─────────┘  └─────────┘  └─────────┘                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │     Global Memory (all warps shared, Memory class)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ learning_console │  │--trace mode       │  │ Vec4 ALU    │  │
│  │ .py (w cmd)      │  │(per-warp trace)   │  │(SWAR ops)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

Phase 2 introduces the SIMT (Single Instruction, Multiple Threads) execution model -- the core abstraction of modern GPUs. Unlike Phase 1's SIMD model (vector registers with lanes), SIMT models **multiple threads grouped into warps**, where each thread has its own scalar register file but shares a program counter with other threads in the same warp.

Phase 2 引入了 SIMT（单指令多线程）执行模型——现代 GPU 的核心抽象。与 Phase 1 的 SIMD 模型（带 lane 的向量寄存器）不同，SIMT 将**多个线程分组为 warp**，每个线程拥有自己的标量寄存器堆，但与同一 warp 中的其他线程共享程序计数器。

This is a fundamental paradigm shift: instead of "vector of data", we now have "collection of threads" -- which is exactly how CUDA and GPU programming models work.

## 2. Motivation / 设计动机

Phase 1 showed SIMD data parallelism, but real GPUs don't expose vector registers to programmers. Instead, CUDA/OpenCL expose a thread-based programming model where thousands of lightweight threads execute the same kernel. The hardware groups these threads into warps (32 on NVIDIA GPUs) and executes them in SIMD fashion.

Phase 1 展示了 SIMD 数据并行，但真实 GPU 不向程序员暴露向量寄存器。相反，CUDA/OpenCL 暴露了基于线程的编程模型，数千个轻量级线程执行相同的内核。硬件将这些线程分组为 warp（NVIDIA GPU 上为 32 个）并以 SIMD 方式执行。

**What Phase 2 enables:**
- Multiple warps with round-robin scheduling
- Per-thread scalar register files (independent per thread)
- Thread identification (TID/WID instructions)
- Barrier synchronization within a warp
- Thread-diverse memory addressing (LD/ST using base_addr + thread_id)
- Multiple warp interleaving (hiding latency)

**GPGPU-Sim context**: GPGPU-Sim's `shd_warp_t` manages a warp's state: thread registers, PC, active mask, SIMT stack, and barrier state. The `scheduler_unit` selects which warp to issue each cycle. The `shader_core_ctx` is the SM (Streaming Multiprocessor) that contains all warps and the scheduler. Phase 2 implements simplified versions of all three.

**Key difference from Phase 1**: Phase 1's vector registers (v0-v7, each VLEN=8 lanes) are replaced by per-thread scalar registers (each thread has r0-r15). This models real GPU architecture where each thread has its own set of registers, not a shared vector register file.

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 SIMT Execution Model / SIMT 执行模型

```
SIMTCore::step():
  1. scheduler.select_warp() → warp    # Choose next warp to execute
  2. Fetch & Decode instruction at warp.pc
  3. For each active thread in warp:
       Execute instruction using thread's own register file
  4. warp.pc += 1
  5. If HALT: mark warp as done
```

The key insight: **one PC, N threads**. The same instruction is broadcast to all active threads, but each thread reads/writes its own registers. This is the SIMT abstraction.

### 3.2 Round-Robin Warp Scheduling / 轮询 Warp 调度

The scheduler cycles through warps sequentially. Each call to `select_warp()` returns the next non-done, non-barrier-waiting warp:

```python
def select_warp():
    for _ in range(num_warps):
        warp = warps[current_idx]
        current_idx = (current_idx + 1) % num_warps
        if warp is active:      # not done, not at barrier
            return warp
    return None                 # all warps done
```

This matches GPGPU-Sim's LRR (Loose Round Robin) policy. Real GPUs use more sophisticated policies like GTO (Greedy Then Oldest) to better hide memory latency.

### 3.3 Thread-Diverse Memory Addressing / 线程差异化内存寻址

In Phase 2, LD and ST instructions use `base_addr + thread_id` as the effective address:

```
LD rd, [imm]:      # effective address = imm + thread_id
  for each active thread t:
    addr = (imm + t.thread_id) & mask
    t.write(rd, memory.read(addr))

ST rs1, [imm]:     # effective address = imm + thread_id
  for each active thread t:
    addr = (imm + t.thread_id) & mask
    memory.write(addr, t.read(rs1))
```

This mimics the GPU programming pattern where thread i accesses element i of an array. The base address is the same for all threads, but each thread accesses a different element based on its ID.

### 3.4 Barrier Synchronization / 屏障同步

The BAR instruction acts as a synchronization point within a warp. Since all threads in a warp share a PC and execute in lockstep (in Phase 2, before divergence), BAR is effectively a no-op. It resets the barrier state for correct behavior when barriers are used in loops or conditional code.

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Lines | Purpose |
|------|-------|---------|
| `isa.py` | 145 | Adds TID(0x21), WID(0x22), BAR(0x23); drops vector opcodes |
| `register_file.py` | 72 | Scalar 16x32-bit (same as Phase 0, used per-thread) |
| `alu.py` | 62 | Scalar ALU (same as Phase 0) |
| `memory.py` | 75 | Global memory shared across all warps |
| **`warp.py`** | 103 | **NEW**: Thread and Warp classes |
| **`scheduler.py`** | 65 | **NEW**: WarpScheduler (round-robin) |
| **`simt_core.py`** | 333 | **NEW**: SIMTCore replaces CPU class |
| `assembler.py` | 231 | Extended with TID/WID/BAR parsing |
| `vec4_alu.py` | 66 | Vec4 SWAR ALU (carried from Phase 1) |
| **`learning_console.py`** | 415 | **NEW**: SIMT-aware console with `w` command |

**Total: ~1450 lines**

Key change from Phase 1: **No vector_register_file.py or vector_alu.py**. The SIMD vector approach is replaced by per-thread scalar registers with SIMT execution. Vector operations are emulated by having each thread handle one element.

### 4.2 Key Data Structures / 关键数据结构

**Thread** (warp.py):
```
thread_id: int           # Thread index within warp (0..warp_size-1)
reg_file: RegisterFile   # Per-thread 16x32-bit register file

read_reg(reg_id) -> int
write_reg(reg_id, value)
```

**Warp** (warp.py):
```
warp_id: int             # Warp index
threads: list[Thread]    # All threads in this warp (warp_size = 8)
pc: int                  # Shared program counter
active_mask: int         # Bitmask: bit i = 1 → thread i is active
at_barrier: bool         # True if warp is waiting at a barrier
barrier_count: int       # Number of threads that reached barrier
done: bool               # True if warp has executed HALT

active_threads() -> list[Thread]   # Returns threads with active_mask bit set
is_active(thread_id) -> bool
```

**SIMTCore** (simt_core.py):
```
scheduler: WarpScheduler
warps: list[Warp]
memory: Memory
warp_size: int
num_warps: int
program: list[int]
instr_count: int

load_program(program)    # Load program into all warps
step() -> bool           # Execute one warp instruction
run()                    # Execute until all warps done
_execute_warp(warp, instr)  # SIMD broadcast execution
```

**WarpScheduler** (scheduler.py):
```
warps: list[Warp]
current_idx: int         # Current warp index for round-robin
policy: str              # "rr" (round-robin)

select_warp() -> Warp | None   # Select next active warp
has_active_warps() -> bool
```

### 4.3 ISA Encoding / 指令编码

**New SIMT Opcodes (Phase 2):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x21 | TID | R | rd = thread_id (per-thread) |
| 0x22 | WID | R | rd = warp_id (per-thread) |
| 0x23 | BAR | - | Barrier synchronization (warp-level) |

**Removed opcodes (dropped from Phase 1):**
- Vector opcodes 0x11-0x17 (VADD, VSUB, VMUL, VDIV, VLD, VST, VMOV) — not carried to Phase 2
- Reason: SIMT uses per-thread scalar registers, not vector registers

**Retained opcodes:**
- Vec4 opcodes 0x25-0x28 (V4PACK, V4ADD, V4MUL, V4UNPACK) — operate on per-thread scalar registers

### 4.4 Module Interfaces / 模块接口

```python
class SIMTCore:
    def __init__(self, warp_size=8, num_warps=2, memory_size=1024)
    def load_program(self, program: list[int])
    def step(self) -> bool          # Execute one warp instruction
    def run(self, trace: bool = False)
    def dump_state(self) -> str

class WarpScheduler:
    def __init__(self, warps: List[Warp], policy: str = "rr")
    def select_warp(self) -> Optional[Warp]
    def has_active_warps(self) -> bool

class Thread:
    def __init__(self, thread_id: int, num_regs: int = 16)
    def read_reg(self, reg_id: int) -> int
    def write_reg(self, reg_id: int, value: int)

class Warp:
    def __init__(self, warp_id: int, warp_size: int = 8)
    def active_threads(self) -> List[Thread]
    def is_active(self, thread_id: int) -> bool
    def reset_barrier(self)
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Multi-Warp Execution Timeline / 多 Warp 执行时间线

Running `05_multi_warp.asm` with 2 warps x 8 threads (config: `--num-warps 2 --warp-size 8`):

```asm
; Each thread stores its TID to mem[tid], then its WID to mem[8+tid]
TID r1           ; r1 = thread_id
WID r2           ; r2 = warp_id
ST r1, [0]       ; mem[tid] = tid
ST r2, [8]       ; mem[8+tid] = warp_id
HALT
```

Trace output (round-robin interleaving):

```
[Cycle 0] W0 PC=0: TID rd=r1 rs1=r0 rs2=r0 imm=0 | active=8 threads | [T0:r1:0->0, T1:r1:0->1]
[Cycle 1] W1 PC=0: TID rd=r1 rs1=r0 rs2=r0 imm=0 | active=8 threads | [T0:r1:0->0, T1:r1:0->1]
[Cycle 2] W0 PC=1: WID rd=r2 rs1=r0 rs2=r0 imm=0 | active=8 threads | [T0:r2:0->0, T1:r2:0->0]
[Cycle 3] W1 PC=1: WID rd=r2 rs1=r0 rs2=r0 imm=0 | active=8 threads | [T0:r2:0->1, T1:r2:0->1]
[Cycle 4] W0 PC=2: ST rd=r0 rs1=r1 rs2=r0 imm=0 | active=8 threads | mem: mem[0]=0, mem[1]=1, ...
[Cycle 5] W1 PC=2: ST rd=r0 rs1=r1 rs2=r0 imm=0 | active=8 threads | mem: mem[0]=0, mem[1]=1, ...
[Cycle 6] W0 PC=3: ST rd=r0 rs1=r2 rs2=r0 imm=8 | active=8 threads | mem: mem[8]=0, mem[9]=0, ...
[Cycle 7] W1 PC=3: ST rd=r0 rs1=r2 rs2=r0 imm=8 | active=8 threads | mem: mem[8]=1, mem[9]=1, ...
[Cycle 8] W0 PC=4: HALT rd=r0 rs1=r0 rs2=r0 imm=0 | ...
[Cycle 9] W1 PC=4: HALT rd=r0 rs1=r0 rs2=r0 imm=0 | ...
[Summary] 10 cycles, 10 instructions
```

Notice the round-robin interleaving: W0 executes one instruction, then W1 executes one instruction, alternating. This is the basic form of latency hiding — while one warp stalls (e.g., on a memory load), another warp can execute.

### 5.2 TID/WID Demo / TID/WID 演示

Running `01_tid_wid.asm`:

```asm
TID r1           ; r1 = thread_id (0, 1, 2, ..., 7)
WID r2           ; r2 = warp_id (0)
MOV r3, 100      ; r3 = 100
ADD r4, r1, r3   ; r4 = thread_id + 100
ST r4, [0]       ; mem[0+tid] = tid + 100
HALT
```

Each thread computes different values because `TID` returns different `thread_id` values. With 8 threads, mem[0..7] = 100, 101, ..., 107.

### 5.3 Learning Console Session / 学习控制台会话

```
> python learning_console.py 01_tid_wid.asm --num-warps 2
+----------------------------------------------------------+
|     toygpgpu Learning Console -- Phase 2 (SIMT)           |
+----------------------------------------------------------+
|  Program: 6 instructions                                   |
|  Config:  2 warp(s) x 8 threads/warp                      |
|  Commands: Enter=step, r=run, i=info, w<id>, q=quit       |
+----------------------------------------------------------+

[0] > s
Cycle 0: PC=0 TID rd=r1 rs1=r0 rs2=r0 imm=0 | active=8 threads | [T0:r1:0->0, T1:r1:0->1 ...]
[1] > w 0
Warp 0: PC=1, mask=0b11111111, active=8/8, ACTIVE
  Threads:
    [A] T0: r1=0
    [A] T1: r1=1
    ...
[1] > r
...
  [OK] All warps completed at cycle 12
--- Final State ---
Cycles executed: 12
Total instructions: 12
Memory (non-zero):
  mem[  0]=100
  mem[  1]=101
  ...
```

## 6. Comparison with Phase 1 / 与 Phase 1 的对比

| Aspect | Phase 1 (SIMD) | Phase 2 (SIMT) | Change |
|--------|----------------|-----------------|--------|
| Files | 10 source files | 10 source files | Replaced: cpu→simt_core, +warp, +scheduler, -vector_reg, -vector_alu |
| Opcodes | 19 (scalar+vector+vec4) | 14 (scalar+SIMT+vec4) | +TID/WID/BAR(0x21-0x23), -vector ops(0x11-0x17) |
| Execution Unit | CPU (+ cpu.py) | SIMTCore (+ simt_core.py) | Complete replacement |
| Parallelism | 8-lane vector SIMD | N warps x M threads SIMT | More flexible threading |
| Register Model | Scalar + Vector registers | Per-thread scalar registers | Fundamental change |
| Thread ID | Lane index (implicit) | TID instruction (explicit) | Programmer-controlled |
| Scheduling | Single PC | Round-robin warp scheduling | Multiple PCs |
| Memory Model | Flat | Flat + TID-based addressing | Thread-diverse access |
| Vec4 Support | Yes (cpu.py) | Yes (simt_core.py) | Preserved |
| Console | vreg command | w <id> command | Warp-aware debugging |

**What was REMOVED from Phase 1:**
- `vector_register_file.py` — replaced by per-thread scalar registers in `warp.py`
- `vector_alu.py` — vector operations dropped (SIMT threads handle one element each)
- Vector opcodes 0x11-0x17 (VADD, VSUB, VMUL, VDIV, VLD, VST, VMOV)

**What was ADDED in Phase 2:**
- `warp.py` — Thread and Warp classes (replaces vector register file concept)
- `scheduler.py` — WarpScheduler for round-robin warp selection
- `simt_core.py` — SIMTCore (replaces cpu.py as top-level module)
- TID(0x21), WID(0x22), BAR(0x23) instructions
- Thread-diverse LD/ST addressing

**What was PRESERVED from Phase 1:**
- Scalar ISA (ADD, SUB, MUL, DIV, LD, ST, MOV, HALT)
- Vec4 ALU and all Vec4 opcodes (0x25-0x28)
- Memory and ALU implementations
- learning_console.py concept (rewritten for SIMT)

## 7. Known Issues and Future Work / 遗留问题与后续工作

**Known Limitations / 已知限制:**
- No branch divergence handling: BEQ/BNE/JMP not yet implemented (all code must be sequential)
- Barrier synchronization is a no-op (all threads in warp always in lockstep)
- Fixed warp size of 8 (vs. 32 in real GPUs)
- LD/ST only support thread-ID-based addressing, not register-offset
- No SIMT stack: all threads must follow the same execution path
- Round-robin scheduling is simple — no priority or GTO policy
- Warp scheduler doesn't model instruction fetch bandwidth limits

**What Phase 3 will add:**
- JMP (0x08), BEQ (0x09), BNE (0x0A) branch instructions
- SIMT Stack for branch divergence and reconvergence
- Predication: SETP (0x24), @p0 prefix, per-thread pred bit
- Warp-level uniform registers: WREAD (0x2A), WWRITE (0x2B)
- Two-pass assembler with label resolution
- JMP merging optimization

**Open Questions / 开放问题:**
- Should vector operations (VADD etc.) be reintroduced in future phases alongside SIMT?
  - Currently not planned — future phases focus on pipeline, memory hierarchy, PTX
- The `barrier_count` field in Warp is declared but not fully utilized. Should it track thread arrivals?
  - Phase 2's model (all threads always in lockstep) makes barriers trivially satisfied
  - In Phase 3+ with divergence, barrier semantics become more complex

**TODOs:**
- [ ] Add register-offset addressing mode (LD/ST with register base + offset)
- [ ] Implement proper barrier counting across divergent paths
- [ ] Consider adding GTO (Greedy Then Oldest) scheduling policy
- [ ] Add unit test for multi-warp memory bank conflicts
