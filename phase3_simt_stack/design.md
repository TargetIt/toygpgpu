# Phase 3: SIMT Stack — Design Document / 设计文档

> **对应 GPGPU-Sim**: simt_stack (stack.cc) + shd_warp_t 的 SIMT 栈管理 + shader_core_ctx 的分支处理
> **参考**: GPGPU-Sim stack.cc (simt_stack 实现), IPDOM (Immediate Post-Dominator) 理论, CUDA 分支发散模型

## 1. Introduction / 架构概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SIMT Core (simt_core.py)                          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Warp 0                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                    SIMT Stack (simt_stack.py)                 │  │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │  │   │
│  │  │  │ Entry 0 (top)│ │ Entry 1      │ │ Entry 2      │ ...     │  │   │
│  │  │  │ reconv_pc    │ │ reconv_pc    │ │ reconv_pc    │         │  │   │
│  │  │  │ orig_mask    │ │ orig_mask    │ │ orig_mask    │         │  │   │
│  │  │  │ taken_mask   │ │ taken_mask   │ │ taken_mask   │         │  │   │
│  │  │  │ fallthrough  │ │ fallthrough  │ │ fallthrough  │         │  │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                   │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│   │
│  │  │T0   │ │T1   │ │T2   │ │T3   │ │T4   │ │T5   │ │T6   │ │T7   ││   │
│  │  │RF   │ │RF   │ │RF   │ │RF   │ │RF   │ │RF   │ │RF   │ │RF   ││   │
│  │  │pred │ │pred │ │pred │ │pred │ │pred │ │pred │ │pred │ │pred ││   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│   │
│  │    shared PC | active_mask | warp_regs {wid, ntid, ...}          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Two-pass     │  │ Predication  │  │ Warp Regs    │                  │
│  │ Assembler    │  │ SETP / @p0   │  │ WREAD/WWRITE │                  │
│  │ (label supp) │  │ (PRED_FLAG)  │  │ (warp_regs)  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ learning_console│  │ --trace mode   │  │ Vec4 ALU      │            │
│  │ (stack/wreg cmd)│  │ (divergence,   │  │ (SWAR ops)    │            │
│  │                 │  │  reconverge)   │  │               │            │
│  └────────────────┘  └────────────────┘  └────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

Phase 3 is the crown jewel of the toygpgpu SIMT abstraction: **SIMT Stack for branch divergence and reconvergence**. This is perhaps the single most important concept that differentiates GPU execution from CPU execution. When threads in the same warp take different branches (divergence), the SIMT stack serializes the execution of each path and automatically reconverges threads when they reach a common program point.

Phase 3 是 toygpgpu SIMT 抽象的核心：**用于分支发散和重汇聚的 SIMT 栈**。这可能是区分 GPU 执行与 CPU 执行的最重要概念。当同一 warp 中的线程走不同分支时（发散），SIMT 栈将每个路径的执行串行化，并在线程到达共同程序点时自动重汇聚。

Additionally, Phase 3 introduces **predication** (conditional execution via SETP/@p0) as an alternative to branching, and **warp-level uniform registers** (WREAD/WWRITE) for sharing values across all threads in a warp.

## 2. Motivation / 设计动机

In Phase 2, all threads in a warp always followed the same execution path (no branches). Real GPU programs have conditionals (if/else, loops). When threads in the same warp take different branches, the GPU must serialize the divergent paths while tracking which threads belong to which path — this is the job of the SIMT stack.

在 Phase 2 中，warp 中的所有线程始终遵循相同的执行路径（无分支）。真实 GPU 程序包含条件语句。当同一 warp 中的线程走不同分支时，GPU 必须串行化发散路径，同时跟踪哪些线程属于哪个路径——这就是 SIMT 栈的工作。

**What Phase 3 enables:**
- Full if/else branch divergence with automatic reconvergence
- Nested divergence (branches within branches)
- Predicated execution as a divergence-free alternative
- Warp-level uniform registers for sharing values across threads
- Label-based two-pass assembler
- JMP merging optimization for efficient reconvergence

**GPGPU-Sim context**: GPGPU-Sim's `simt_stack` (stack.cc) uses compile-time IPDOM (Immediate Post-Dominator) analysis to determine reconvergence points. Phase 3 simplifies this: the reconvergence point is always `PC_after_branch + 1` (the instruction immediately following the branch). This works correctly for the most common patterns (if/else, if/endif) and is a well-known simplification used in many GPU educational simulators.

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 SIMT Stack Algorithm / SIMT 栈算法

The SIMT stack is a per-warp data structure that manages divergent execution paths.

**Push on divergence (branch where some threads go one way, others another):**

```python
when divergence detected at branch instruction:
    stack.push(Entry(
        reconv_pc = old_pc + 1,       # Reconvergence point = next instruction
        orig_mask = warp.active_mask,   # Save original active mask
        taken_mask = taken_mask,        # Threads that took the branch
        fallthrough_pc = old_pc + 1     # Non-taken path start
    ))
    warp.active_mask = taken_mask       # Execute taken path first
    warp.pc = target_pc                 # Jump to branch target
```

**Pop on reconvergence (PC matches top of stack's reconv_pc):**

```python
when warp.pc == stack.top().reconv_pc:
    entry = stack.pop()
    remaining = entry.orig_mask & ~entry.taken_mask   # Threads not yet executed
    if remaining:
        # Switch to the fallthrough path
        stack.push(Entry(
            reconv_pc = entry.reconv_pc,      # Same reconvergence point
            orig_mask = remaining,             # The remaining (untaken) threads
            taken_mask = remaining,            # All remaining threads will execute
            fallthrough_pc = entry.fallthrough_pc
        ))
        warp.active_mask = remaining
        warp.pc = entry.fallthrough_pc         # Execute fallthrough path
    else:
        # All paths done: restore original mask, continue at reconvergence
        warp.active_mask = entry.orig_mask
        warp.pc = entry.reconv_pc              # Continue from reconv point
```

### 3.2 IPDOM Reconvergence / IPDOM 重汇聚

IPDOM (Immediate Post-Dominator) is the compile-time analysis that identifies the earliest point where all divergent paths reconverge. In GPGPU-Sim, the compiler marks this point in the binary. Phase 3 uses a simplified approach: the reconvergence point is always `PC+1` of the branch instruction.

For the common if/else pattern:

```
    PC 5: BEQ r4, r1, even_path    # Branch: some threads go to even_path
    PC 6: (fallthrough)             # Reconvergence point (odd path starts here)
    ...
    PC N: even_path:                # Even path target
    ...
    PC M: JMP reconv                # Both paths jump here
    PC M+1: reconv:                 # Actual reconvergence
```

The stack ensures:
1. Even threads execute `even_path` first (PC jumps to even_path)
2. When even threads JMP to `reconv`, PC reaches `reconv_pc` (PC 6)
3. Stack pops, odd threads get scheduled to execute from PC 6 (fallthrough)
4. When odd threads reach `reconv` via JMP, POP again restores full mask

This serializes the two paths while ensuring both converge before continuing.

### 3.3 JMP Merging Optimization / JMP 合并优化

When a JMP instruction is encountered within a divergent path and all active threads take the JMP, the reconvergence point is updated to the JMP target:

```python
is_merge_jmp = (op == OP_JMP and not stack.empty and
                taken_mask == warp.active_mask)
if is_merge_jmp:
    top = stack.top()
    # Update reconv_pc to JMP target (more efficient reconvergence)
    top.reconv_pc = target_pc
```

This optimization moves the reconvergence point closer to the actual merge location rather than sticking with the original `PC+1` heuristic.

### 3.4 Predication Algorithm / 谓词执行算法

Predication avoids divergence entirely by conditionally disabling instruction execution per-thread:

```python
# SETP.EQ sets pred = (rs1 == rs2) for each thread
for t in active_threads():
    t.pred = (t.read_reg(rs1) == t.read_reg(rs2))

# @p0 prefix: only execute if thread's pred is True
exec_threads = [t for t in active_threads() if t.pred]
```

The PRED_FLAG (bit 31 of instruction word, mapped to bit 11 of the raw encoding for storage) marks an instruction as predicated. The assembler sets this bit when `@p0` prefix is used. The decoder strips it from the opcode but preserves it in `instr.raw`.

### 3.5 Warp-Level Uniform Registers / Warp 级统一寄存器

These are registers shared by all threads in a warp, not per-thread. They provide a way to store and broadcast warp-wide values:

```
WREAD rd, wid:     rd = warp.warp_regs[wid_idx]    (broadcast to all active threads)
WWRITE rs1, wid:   warp.warp_regs[wid_idx] = rs1   (written by first active thread)
```

Built-in warp registers: `wid` (warp_id, index 0), `ntid` (num_threads, index 1).

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Lines | Purpose |
|------|-------|---------|
| `isa.py` | 171 | +JMP(0x08), BEQ(0x09), BNE(0x0A), SETP(0x24), WREAD(0x2A), WWRITE(0x2B), PRED_FLAG |
| `register_file.py` | 72 | Scalar 16x32-bit (unchanged) |
| `alu.py` | 62 | Scalar ALU (unchanged) |
| `memory.py` | 75 | Global memory (unchanged) |
| `warp.py` | 121 | +pred flag per Thread, +warp_regs dict, +simt_stack, read/write_warp_reg |
| `scheduler.py` | 65 | WarpScheduler (unchanged from Phase 2) |
| **`simt_stack.py`** | 76 | **NEW**: SIMTStack + SIMTStackEntry classes |
| `simt_core.py` | 487 | +SIMT Stack reconvergence, branch execution, predication, WREAD/WWRITE |
| `assembler.py` | 185 | **Two-pass** assembler with label resolution, @p0 prefix |
| `vec4_alu.py` | 36 | Vec4 SWAR ALU (carried forward) |
| `learning_console.py` | 542 | +stack, wreg commands; divergence/reconvergence display |

**Total: ~1800 lines**

### 4.2 Key Data Structures / 关键数据结构

**SIMTStackEntry** (simt_stack.py):
```
reconv_pc: int          # Reconvergence PC: the PC where paths should reconverge
orig_mask: int          # Original active_mask before divergence
taken_mask: int         # Bitmask of threads that have executed the taken path
fallthrough_pc: int     # Starting PC for fallthrough (non-taken) path
```

**SIMTStack** (simt_stack.py):
```
entries: list[SIMTStackEntry]   # Stack of entries (supports nesting)

push(entry)                     # Push new divergence entry
pop() -> SIMTStackEntry | None  # Pop on reconvergence
top() -> SIMTStackEntry | None  # Peek at top
at_reconvergence(pc) -> bool    # Check if PC matches top's reconv_pc
empty: bool                     # True if stack is empty
```

**Thread** (warp.py, Phase 3 additions):
```
# Existing: thread_id, reg_file
pred: bool                      # NEW: predicate register for @p0 execution
```

**Warp** (warp.py, Phase 3 additions):
```
# Existing: warp_id, threads, pc, active_mask, at_barrier, barrier_count, done
simt_stack: SIMTStack           # NEW: per-warp SIMT stack
warp_regs: dict                 # NEW: warp-level uniform registers {0: wid, 1: ntid}

read_warp_reg(reg_idx) -> int   # NEW
write_warp_reg(reg_idx, value)  # NEW
```

### 4.3 ISA Encoding / 指令编码

**New Branch Opcodes (Phase 3):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x08 | JMP | - | Unconditional jump: PC = PC + 1 + imm |
| 0x09 | BEQ | R | Branch if Equal: PC = PC+1+imm if rs1 == rs2 |
| 0x0A | BNE | R | Branch if Not Equal: PC = PC+1+imm if rs1 != rs2 |

Branch target is computed as `PC + 1 + imm`, where PC is the instruction's own address and imm is a 12-bit signed offset. The `+1` accounts for the pre-execution PC increment in the core execution loop.

**New Predication Opcode (Phase 3):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x24 | SETP | R | Set predicate: imm[0]=0 → pred=(rs1==rs2), imm[0]=1 → pred=(rs1!=rs2) |

**Predication flag** (PRED_FLAG = 0x80000000, bit 31 in raw instruction word):
- Set by assembler when `@p0` prefix is used
- Stripped from opcode during decode (opcode & 0x7F)
- Checked during execution: `if instr.raw & PRED_FLAG: filter_threads_by_pred()`
- Only 1 predicate bit per thread (p0) in current implementation

**New Warp Register Opcodes (Phase 3):**

| Opcode | Name | Type | Operation |
|--------|------|------|-----------|
| 0x2A | WREAD | R | rd = warp_regs[imm & 0xF] (broadcast to threads) |
| 0x2B | WWRITE | R | warp_regs[imm & 0xF] = rs1 (from first active thread) |

**Warp register indices:**
| Index | Name | Description |
|-------|------|-------------|
| 0 | wid | warp_id |
| 1 | ntid | num_threads per warp (warp_size) |

### 4.4 Module Interfaces / 模块接口

```python
class SIMTStack:
    def __init__(self)
    def push(self, entry: SIMTStackEntry)
    def pop(self) -> Optional[SIMTStackEntry]
    def top(self) -> Optional[SIMTStackEntry]
    def at_reconvergence(self, pc: int) -> bool
    @property def empty(self) -> bool

class SIMTCore:  # Extended from Phase 2
    def _handle_reconvergence(self, warp: Warp)   # NEW: pop + path switch logic
    def _execute_branch(self, warp, instr, old_pc) # NEW: divergence logic
    def _exec_threads(self, warp, instr) -> list   # NEW: filter by @p0 predicate

class Warp:  # Extended from Phase 2
    def read_warp_reg(self, reg_idx: int) -> int   # NEW
    def write_warp_reg(self, reg_idx: int, value)  # NEW
    # new field: self.simt_stack = SIMTStack()
    # new field: self.warp_regs = {0: warp_id, 1: warp_size}
    # new field per Thread: self.pred = False

def assemble(source: str) -> list[int]  # Two-pass with labels
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Divergence Execution Timeline / 发散执行时间线

Running `03_divergence.asm` (8 threads, warp_size=8):

```asm
; tid:    0   1   2   3   4   5   6   7
; even:  Y       Y       Y       Y
; odd:      N       N       N       N

TID r1           ; r1 = tid
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2   ; r4 = (tid/2)*2  (== tid if even, != tid if odd)
BEQ r4, r1, even_path  ; even threads take branch, odd fall through

; === Odd path ===
MOV r6, 1
ST r6, [200]
JMP reconv

even_path:
; === Even path ===
MOV r6, 2
ST r6, [200]
JMP reconv

reconv:
TID r7
ST r7, [300]
HALT
```

Execution timeline:

```
Cycle  W0  PC  active_mask   Operation                        Stack (depth, content)
────── ── ── ───────────── ───────────────────────────────── ──────────────────────────────
  0    0   0  11111111      TID r1                           empty
  1    0   1  11111111      MOV r2, 2                        empty
  2    0   2  11111111      DIV r3, r1, r2                   empty
  3    0   3  11111111      MUL r4, r3, r2                   empty
  4    0   4  11111111      MOV r5=0 (assembled as MOV r5,0) empty
  5    0   5  11111111      BEQ r4,r1 → even_path            [depth=1: reconv=PC6, orig=FF, taken=55, fall=PC6]
                             active_mask → 01010101 (even)
                             PC → even_path (PC 9)
  6    0   9  01010101      MOV r6, 2                        [depth=1]
  7    0  10  01010101      ST r6, [200]                     [depth=1]
  8    0  11  01010101      JMP reconv → PC 13               [depth=1]
                             (JMP merging: reconv_pc updated to PC13)
  9    0  13  →RECONVERGE←                                    POP → push odd threads
                             active_mask → 10101010 (odd)     [depth=1: reconv=PC13, orig=FF, taken=AA, fall=PC6]
                             PC → 6 (fallthrough, odd path)
 10    0   6  10101010      MOV r6, 1                        [depth=1]
 11    0   7  10101010      ST r6, [200]                     [depth=1]
 12    0   8  10101010      JMP reconv → PC 13               [depth=1]
 13    0  13  →RECONVERGE←                                    POP → stack empty
                             active_mask → 11111111 (all)    empty
                             PC → 13 (reconv point)
 14    0  13  11111111      TID r7                           empty
 15    0  14  11111111      ST r7, [300]                     empty
 16    0  15  11111111      HALT                             empty
[Summary] 17 cycles, 17 instructions
```

**Memory result:**
```
mem[200] = 2 (even threads), mem[201] = 1 (odd threads),
mem[202] = 2, mem[203] = 1, ..., mem[207] = 1

mem[300] = 0, mem[301] = 1, ..., mem[307] = 7 (all threads)
```

### 5.2 Predication Demo / 谓词执行演示

Running `06_predication.asm` — same logic as divergence but without SIMT stack:

```asm
TID r1
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2
SUB r5, r1, r4        ; r5 = tid % 2
SETP.EQ r5, r0        ; pred = (r5 == 0) — true for even threads
@p0 MOV r6, 100       ; Even threads only: r6 = 100
@p0 ST r6, [100]      ; Even threads only: mem[100+tid] = 100
TID r7                ; All threads (no divergence!)
ADD r7, r7, 10
ST r7, [200]          ; All threads: mem[200+tid] = tid + 10
HALT
```

The key difference: no stack push/pop, no serialization. All threads stay together. However, both paths are still fetched (just disabled lanes don't write). This is more efficient when the condition is simple and the divergent code is short.

### 5.3 Warp Registers Demo / Warp 寄存器演示

Running `08_warp_regs.asm`:

```asm
WREAD r0, wid      ; r0 = warp_id (broadcast to all threads)
WREAD r1, ntid     ; r1 = warp_size (broadcast to all threads)
ST r0, [0]         ; mem[tid] = warp_id
ST r1, [1]         ; mem[1] = warp_size
MOV r3, 99
WWRITE r3, wid     ; warp_regs[wid] = 99
WREAD r2, wid      ; r2 = 99 (modified)
ST r2, [2]         ; mem[2] = 99
HALT
```

### 5.4 Trace Output Example / 追踪输出示例

From `--trace` mode with divergence:

```
[Cycle 0] W0 PC=0: TID rd=r1 rs1=r0 rs2=r0 imm=0 | active=0b11111111 | reg: T0:r1=0->0, T1:r1=0->1, ...
[Cycle 5] W0 PC=5: BEQ r4,r1 -> PC9 | active=0b11111111 | reg: ... | DIVERGE: taken=0b01010101 not_taken=0b10101010
[Cycle 6] W0 PC=9: MOV rd=r6 rs1=r0 rs2=r0 imm=2 | active=0b01010101 | reg: T0:r6=0->2, T2:r6=0->2, ...
[Cycle 9] W0 RECONVERGE: mask=0b01010101 path=PC13: ST rd=r0 rs1=r7 rs2=r0 imm=300
```

### 5.5 Learning Console Session / 学习控制台会话

```
> python learning_console.py 03_divergence.asm --warp-size 8
==============================================
  toygpgpu Learning Console (Phase 3)
  SIMT Stack / Branch Divergence Debugger
==============================================
  Program: 14 instructions
  Config:  1 warp(s) x 8 threads/warp
==============================================

[0] > s
[0] Warp 0: TID rd=r1 rs1=r0 rs2=r0 imm=0  | PC=0 active=0b11111111
         Reg W0 T0: r1 0 -> 0
         Reg W0 T1: r1 0 -> 1
...
[5] > s
[5] Warp 0: BEQ r4,r1 -> PC9  | PC=5 active=0b11111111
         SIMT: PUSH reconv=PC6 orig=0b11111111 taken=0b01010101
[6] > s
[6] Warp 0: MOV rd=r6 rs1=r0 rs2=r0 imm=2  | PC=9 active=0b01010101
[8] > s
[8] Warp 0: JMP -> PC13  | PC=11 active=0b01010101
[9] > stack
  Warp 0 (depth=1):
    [0] reconv=PC13 orig=0b11111111 taken=0b01010101 fallthrough=PC6
[9] > s
[9] Warp 0: MOV rd=r6 rs1=r0 rs2=r0 imm=1  | PC=6 active=0b10101010
         SIMT: POP (reconvergence)
         SIMT: now top entry mask=0b11111111
...
[DONE] All warps completed at cycle 17
```

## 6. Comparison with Phase 2 / 与 Phase 2 的对比

| Aspect | Phase 2 (SIMT) | Phase 3 (SIMT Stack) | Change |
|--------|----------------|----------------------|--------|
| Files | 10 source files | 11 source files | +1: simt_stack.py |
| Opcodes | 14 (0x00-0x07, 0x21-0x23, 0x25-0x28) | 21 (0x00-0x0A, 0x21-0x24, 0x25-0x28, 0x2A-0x2B) | +7: JMP/BEQ/BNE/SETP/WREAD/WWRITE |
| Key Feature | Sequential SIMT execution | Branch divergence + reconvergence | Full SIMT model |
| Branch Support | None (no JMP/BEQ/BNE) | Full: JMP, BEQ, BNE (0x08-0x0A) | New capability |
| Divergence Model | All threads lockstep | SIMT stack serialization | Fundamental addition |
| Predication | None | SETP + @p0 prefix + per-thread pred | New capability |
| Warp Registers | None | WREAD/WWRITE + warp_regs dict | New capability |
| Assembler | Single-pass | Two-pass with label resolution | Label support |
| Thread State | reg_file only | reg_file + pred flag | Extended |
| Memory Access | base+tid addressing | base+tid addressing | Unchanged |
| Vec4 Support | Yes | Yes | Preserved |
| Console | w <id> command | +stack, +wreg commands | Extended debugging |
| Trace | Per-warp trace | +divergence, +reconvergence markers | Enhanced |

**What was ADDED in Phase 3:**
- `simt_stack.py` — SIMTStack and SIMTStackEntry classes
- JMP (0x08), BEQ (0x09), BNE (0x0A) branch instructions
- SETP (0x24) predicate set instruction
- @p0 predicated instruction prefix (PRED_FLAG = 0x80000000)
- WREAD (0x2A), WWRITE (0x2B) warp register instructions
- Per-thread `pred` boolean flag in Thread class
- `warp_regs` dictionary in Warp class (pre-initialized with wid, ntid)
- `simt_stack` attribute in Warp class
- Two-pass assembler with label resolution (`_assemble_line` now takes labels dict)
- `_handle_reconvergence`, `_execute_branch`, `_exec_threads` methods in SIMTCore
- JMP merging optimization in `_execute_branch`

**What was MODIFIED in Phase 3:**
- `isa.py`: extended opcode list, PRED_FLAG constant, WREG_NAMES mapping
- `simt_core.py`: `_execute_warp` now calls `_exec_threads` for predication, handles SETP/WREAD/WWRITE, delegates branches to `_execute_branch`; `step()` now checks `at_reconvergence` before each instruction
- `warp.py`: Thread gets `pred` field, Warp gets `simt_stack` and `warp_regs`
- `assembler.py`: completely rewritten as two-pass assembler with `@p0`, label resolution, WREAD/WWRITE parsing
- `learning_console.py`: added `stack` and `wreg` commands, divergence/reconvergence display

**What was PRESERVED from Phase 2:**
- All scalar ISA (ADD, SUB, MUL, DIV, LD, ST, MOV, HALT)
- All SIMT ISA (TID, WID, BAR)
- All Vec4 ISA (V4PACK, V4ADD, V4MUL, V4UNPACK)
- WarpScheduler, Memory, ALU implementations
- Round-robin scheduling

## 7. Known Issues and Future Work / 遗留问题与后续工作

**Known Limitations / 已知限制:**

1. **Simplified reconvergence**: The `PC+1` heuristic works for if/else but may be incorrect for complex control flow. GPGPU-Sim uses compile-time IPDOM analysis for accurate reconvergence points.

2. **No IBuffer.peek()**: The reconvergence logic could skip redundant cycles if we could peek at the next instruction before executing. Phase 7 (Pipeline) adds IBuffer which fixes this.

3. **Nested divergence depth**: Limited only by Python recursion/list depth, but deep nesting can cause the VM state to grow large.

4. **Single predicate bit**: Only `@p0` is supported. Real GPUs have multiple predicate registers.

5. **Barrier with divergence**: The BAR instruction does not properly handle barriers across divergent paths. When threads are on different paths, a barrier should only synchronize threads that are on the same path.

6. **Warp register WWRITE**: Writes only from the first active thread. Real GPU warp registers are typically uniform (all threads see the same value).

7. **No branch prediction**: Every branch causes at least one cycle of fetch overhead.

8. **Memory addressing**: LD/ST still only support `base_addr + thread_id`. No register-offset or index addressing.

**What Phase 4 will add:**
- Scoreboard for hazard detection (RAW/WAR/WAW)
- Dynamic pipeline scheduling with scoreboard
- Memory latency modeling

**Open Questions / 开放问题:**
- Should JMP merging be optional? The optimization changes the reconv_pc, which could interact poorly with nested divergence.
- Should predication be extended to support `@p1`, `@!p0`, etc.?
- Warp register count: should it be configurable per warp?

**TODOs:**
- [ ] Add test for nested divergence exceeding 2 levels
- [ ] Add `@!p0` (inverse predication) support
- [ ] Implement proper barrier across divergent paths
- [ ] Consider adding PTX-style `@` predicate syntax for all instructions, not just selected ones
- [ ] Document the JMP merging optimization corner cases
- [ ] Add unit test for WREAD/WWRITE across multiple warps
