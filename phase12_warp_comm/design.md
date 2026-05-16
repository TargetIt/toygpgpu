# Phase 12: Warp Communication — Design Document / 设计文档

> **对应 GPGPU-Sim**: `gpgpu-sim/../gpgpusim_entry_point.cc` 中的 `shfl_inst` 处理逻辑,
> `gpgpu-sim/../ptx_parser.cc` 中的 vote/ballot PTX 指令解析
> **参考**: CUDA `__shfl_sync()`, `__any_sync()`, `__all_sync()`, `__ballot_sync()` 编程指南

## 1. Introduction / 架构概览

```
                     ┌──────────────────────────────────────────────┐
                     │       Warp Communication Architecture        │
                     │                                              │
                     │           Warp (8 threads)                    │
                     │     ┌──────┬──────┬──────┬──────┐            │
                     │     │ T0   │ T1   │ T2   │ T3   │  ...      │
                     │     │ r1=0 │ r1=10│ r1=20│ r1=30│           │
                     │     └──┬───┴──┬───┴──┬───┴──┬───┘            │
                     │        │      │      │      │                 │
                     │        │  SHFL: cross-thread register read    │
                     │        └──────┼──────┼──────┘                 │
                     │               │      │                        │
                     │              ┌▼──────▼┐                      │
                     │              │  ALL   │ r4 = r3 of thread 0   │
                     │              │  IDX   │ → (all get 0)        │
                     │              └────────┘                      │
                     │                                              │
                     │        ┌──────────────────┐                  │
                     │        │  VOTE: ANY/ALL   │                  │
                     │        │  across warp     │                  │
                     │        │  → bool (0 or 1) │                  │
                     │        └──────────────────┘                  │
                     │                                              │
                     │        ┌──────────────────┐                  │
                     │        │  BALLOT: bitmask │                  │
                     │        │  of threads with │                  │
                     │        │  non-zero rs1    │                  │
                     │        └──────────────────┘                  │
                     └──────────────────────────────────────────────┘
```

Phase 12 向 toygpgpu 添加了**三个 Warp 级通信原语**：SHFL（Warp Shuffle）、VOTE（Warp 投票）和 BALLOT（Warp 选票）。这些指令允许同一个 warp 内的线程在没有 Shared Memory 或 Global Memory 的情况下直接交换数据，是 GPGPU 编程中实现高效 warp 内协作的关键原语。

Phase 12 adds **three warp-level communication primitives** to toygpgpu: SHFL (Warp Shuffle), VOTE (Warp Vote), and BALLOT (Warp Ballot). These instructions allow threads within the same warp to exchange data directly without going through Shared Memory or Global Memory, forming key primitives for efficient intra-warp collaboration in GPGPU programming.

### 新增指令概览 / New Instructions Overview

| 指令 | Opcode | 功能 | CUDA 对应 |
|------|--------|------|-----------|
| SHFL | 0x30 | 读取指定线程的寄存器值 | `__shfl_sync()` |
| VOTE | 0x33 | Warp 级 ANY/ALL 规约 | `__any_sync()` / `__all_sync()` |
| BALLOT | 0x34 | 生成非零线程的 bitmask | `__ballot_sync()` |

## 2. Motivation / 设计动机

### 2.1 为什么需要 Warp 通信？/ Why Warp Communication?

在 GPGPU 编程中，同一 warp 内的线程经常需要协同工作。在引入 Warp 通信原语之前，线程之间的数据交换只能通过以下方式完成：

In GPGPU programming, threads within the same warp often need to collaborate. Before warp communication primitives, data exchange between threads could only be done through:

1. **Shared Memory**: 写入 shared memory → barrier 同步 → 读取。需要 3 条指令，有 bank conflict 风险。
2. **Global Memory**: 写入 global memory → __threadfence() → 读取。延迟高（数百周期）。
3. **寄存器直接交换**: 不存在。线程无法直接访问其他线程的寄存器。

Warp 通信原语(SHFL)允许线程**直接读取其他线程的寄存器值**，无需经过内存层次结构。这使得 warp 内的数据交换从 O(延迟数百周期) 降低到 O(1 周期)。

### 2.2 VOTE 和 BALLOT 的用途 / Purpose of VOTE and BALLOT

- **VOTE.ANY**: 检查 warp 中**是否有任何**线程满足条件。用于早期退出、收敛判断。
- **VOTE.ALL**: 检查 warp 中**是否所有**线程都满足条件。用于屏障条件、全局状态判断。
- **BALLOT**: 生成一个 bitmask，每位代表一个线程是否满足条件。是所有 warp 级规约操作的基础（如 `__popc(__ballot_sync())` 计数）。

### 2.3 与 CUDA 的对应关系 / CUDA Correspondence

```
CUDA PTX          │ toygpgpu      │ 说明
───────────────────┼───────────────┼────────────────────
shfl.sync.idx     │ SHFL rd,rs1,  │ 读取指定 lane 的值
                  │   lane,0      │
shfl.sync.up      │ SHFL rd,rs1,  │ 读取 (tid-delta) 的值
                  │   delta,1     │
shfl.sync.down    │ SHFL rd,rs1,  │ 读取 (tid+delta) 的值
                  │   delta,2     │
shfl.sync.bfly    │ SHFL rd,rs1,  │ 读取 (tid^mask) 的值
                  │   mask,3      │
vote.any          │ VOTE.ANY      │ 任意线程非零 → true
vote.all          │ VOTE.ALL      │ 全部线程非零 → true
vote.ballot       │ BALLOT        │ 非零线程的 bitmask
```

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 SHFL (Warp Shuffle) — 四种模式 / Four Modes

SHFL 的核心思想是：每条线程通过 `target_lane` 公式计算出要读取的源线程 ID，然后直接从该线程的寄存器中取值。所有读取操作在同一个周期内完成，不需要任何内存访问。

The core idea of SHFL: each thread computes a `target_lane` using a mode-specific formula, then reads the register value directly from that source thread. All reads complete in the same cycle without any memory access.

```
伪代码 / Pseudocode:

SHFL rd, rs1, src, mode:
    for each active thread t in warp:
        tid = t.thread_id
        
        if mode == 0:  // IDX — 读取指定 lane
            target = src
        elif mode == 1:  // UP — 读取 (tid - delta)
            target = (tid - src) % WARP_SIZE
        elif mode == 2:  // DOWN — 读取 (tid + delta)
            target = (tid + src) % WARP_SIZE
        elif mode == 3:  // XOR — 读取 (tid ^ mask)
            target = tid ^ src
        else:
            target = src
        
        t.rd = warp.threads[target].rs1
```

#### IDX 模式 (mode=0)

```
所有线程读取同一个 lane 的值：
  T0 → reads T3.rs1    T1 → reads T3.rs1
  T2 → reads T3.rs1    T3 → reads T3.rs1
```
用途：广播一个线程的值到整个 warp（如归约中的部分和广播）。

#### UP 模式 (mode=1)

```
每个线程读取 (tid-delta) 的值：
  delta=1:  T0→T7  T1→T0  T2→T1  T3→T2  (循环上移)
```
用途：移位寄存器、前缀和、相邻数据访问。

#### DOWN 模式 (mode=2)

```
每个线程读取 (tid+delta) 的值：
  delta=1:  T0→T1  T1→T2  T2→T3  T3→T4  (循环下移)
```
用途：与 UP 对称，用于不同方向的移位操作。

#### XOR 模式 (mode=3)

```
每个线程读取 (tid^mask) 的值：
  mask=1:   T0↔T1  T2↔T3  T4↔T5  T6↔T7  (配对交换)
  mask=2:   T0↔T2  T1↔T3  T4↔T6  T5↔T7  (更远配对)
```
用途：蝶形归约、FFT、并行归约中的配对交换。

### 3.2 VOTE (Warp Vote) — ANY 与 ALL / ANY vs ALL

VOTE 对 warp 内所有活跃线程的谓词条件进行跨线程的**逻辑 OR** 或**逻辑 AND** 规约。

VOTE performs a cross-warp **logical OR** or **logical AND** reduction on a predicate condition across all active threads.

```
伪代码 / Pseudocode:

VOTE.ANY rd, rs1:        // imm[0]=0
    result = False
    for each active thread t in warp:
        if t.rs1 != 0:
            result = True
            break        // early exit on first match
    for each active thread t in warp:
        t.rd = 1 if result else 0

VOTE.ALL rd, rs1:        // imm[0]=1
    result = True
    for each active thread t in warp:
        if t.rs1 == 0:
            result = False
            break        // early exit on first mismatch
    for each active thread t in warp:
        t.rd = 1 if result else 0
```

VOTE.ANY 和 VOTE.ALL 都在发现第一个匹配/不匹配线程时**提前退出**，避免不必要的遍历。

Both VOTE.ANY and VOTE.ALL **early-exit** upon finding the first match/mismatch to avoid unnecessary traversal.

### 3.3 BALLOT (Warp Ballot) — Bitmask 生成 / Bitmask Generation

BALLOT 创建一个 warp 宽的 bitmask，每位表示对应线程的 rs1 是否非零。这是所有 warp 级规约统计的基础操作。

BALLOT creates a warp-wide bitmask where each bit indicates whether the corresponding thread's rs1 is non-zero. This is the fundamental operation for all warp-level reduction statistics.

```
伪代码 / Pseudocode:

BALLOT rd, rs1:
    mask = 0
    for each active thread t in warp:
        if t.rs1 != 0:
            mask |= (1 << t.thread_id)
    for each active thread t in warp:
        t.rd = mask

示例 / Example (warp_size=8):
    T0 rs1=1  → bit 0 set
    T1 rs1=0  → bit 1 clear
    T2 rs1=1  → bit 2 set
    T3 rs1=0  → bit 3 clear
    ...
    Result: mask = 0b01010101 = 0x55
    All threads get the same mask value.
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Purpose / 用途 | Change from Phase 11 |
|------|---------------|---------------------|
| `isa.py` | Opcode definitions: OP_SHFL(0x30), OP_VOTE(0x33), OP_BALLOT(0x34) + OPCODE_NAMES | ADDED: 3 new opcodes + names |
| `assembler.py` | Two-pass assembler: SHFL/VOTE.ANY/VOTE.ALL/BALLOT mnemonics | ADDED: 4 new assembly mnemonics |
| `simt_core.py` | SIMT pipeline: _execute_warp() SHFL/VOTE/BALLOT handling | ADDED: 3 new execution blocks |
| `learning_console.py` | Interactive console (unchanged, backward compatible) | Same as Phase 11 |
| `console_display.py` | Rendering module (unchanged) | Same |
| `warp.py` | Warp/Thread with PRED, warp_regs, IBuffer, Scoreboard, SIMT stack | Same |
| `ibuffer.py` | IBuffer with peek()+reconv fix | Same |
| `alu.py`, `vec4_alu.py` | ALU operations | Same |
| `memory.py` | Global memory | Same |
| `cache.py` | L1 cache | Same |
| `register_file.py` | Per-thread register file | Same |
| `scoreboard.py` | Scoreboard with pipeline latency | Same |
| `scheduler.py` | Warp scheduler | Same |
| `simt_stack.py` | SIMT divergence management | Same |
| `operand_collector.py` | Multi-bank register file | Same |
| `thread_block.py` | Thread block, shared memory | Same |
| `shared_memory.py` | Shared memory | Same |
| `gpu_sim.py` | Top-level GPU simulator | Same |
| `run.sh` | Wrapper script | Same |
| `tests/programs/09_warp_shfl.asm` | SHFL demo: IDX, DOWN, XOR modes | NEW demo program |
| `tests/programs/10_warp_vote.asm` | VOTE/BALLOT demo: ANY, ALL, bitmask | NEW demo program |
| `tests/test_phase12.py` | Phase 12 test suite: ISA, assembler, SHFL, VOTE, BALLOT, backward compat | NEW test suite |

### 4.2 ISA Encoding / 指令编码

#### SHFL (0x30) — Warp Shuffle

```
[31:24] opcode=0x30  [23:20] rd  [19:16] rs1  [15:12] src_lane  [11:10] mode  [9:0] unused

  mode 字段 (imm[1:0]):
    00 = IDX  (读取指定 lane)
    01 = UP   (tid - delta)
    10 = DOWN (tid + delta)
    11 = XOR  (tid ^ mask)

  汇编语法 / Assembly:
    SHFL rd, rs1, src_lane, mode
    示例: SHFL r4, r3, 0, 0    ; IDX: 读 T0.r3
          SHFL r5, r3, 1, 2    ; DOWN 1: 读 T(tid+1).r3
          SHFL r6, r3, 1, 3    ; XOR 1: 读 T(tid^1).r3
```

#### VOTE (0x33) — Warp Vote

```
[31:24] opcode=0x33  [23:20] rd  [19:16] rs1  [15:12]=0  [11:1]=0  [0] mode

  imm[0]:
    0 = ANY (逻辑 OR)
    1 = ALL (逻辑 AND)

  汇编语法 / Assembly:
    VOTE.ANY rd, rs1    ; imm[0]=0 → ANY
    VOTE.ALL rd, rs1    ; imm[0]=1 → ALL
```

#### BALLOT (0x34) — Warp Ballot

```
[31:24] opcode=0x34  [23:20] rd  [19:16] rs1  [15:0]=0

  汇编语法 / Assembly:
    BALLOT rd, rs1    ; rd = bitmask of threads where rs1 != 0
```

### 4.3 Key Implementation / 关键实现

#### simt_core.py — SHFL 实现

```python
# Warp Communication (Phase 12)
if op == OP_SHFL:
    src_lane = instr.rs2 % warp.warp_size
    mode = instr.imm & 3
    for t in warp.active_threads():
        tid = t.thread_id
        if mode == 0:       # IDX
            target = src_lane
        elif mode == 1:     # UP
            target = (tid - src_lane) % warp.warp_size
        elif mode == 2:     # DOWN
            target = (tid + src_lane) % warp.warp_size
        elif mode == 3:     # XOR
            target = tid ^ src_lane
        else:
            target = src_lane
        src_thread = warp.threads[target]
        t.write_reg(instr.rd, src_thread.read_reg(instr.rs1))
```

#### simt_core.py — VOTE 实现

```python
if op == OP_VOTE:
    is_all = (instr.imm & 1) == 1  # 0=ANY, 1=ALL
    result = True if is_all else False
    for t in warp.active_threads():
        val = t.read_reg(instr.rs1)
        if is_all:          # ALL: early exit on first zero
            if val == 0:
                result = False
                break
        else:               # ANY: early exit on first non-zero
            if val != 0:
                result = True
                break
    for t in warp.active_threads():
        t.write_reg(instr.rd, 1 if result else 0)
```

#### simt_core.py — BALLOT 实现

```python
if op == OP_BALLOT:
    mask = 0
    for t in warp.active_threads():
        val = t.read_reg(instr.rs1)
        if val != 0:
            mask |= (1 << t.thread_id)
    for t in warp.active_threads():
        t.write_reg(instr.rd, mask)
```

### 4.4 Module Interfaces / 模块接口

```python
# isa.py
OP_SHFL   = 0x30  # rd = thread[src_lane].rs1
OP_VOTE   = 0x33  # rd = ANY/ALL of rs1 across warp
OP_BALLOT = 0x34  # rd = bitmask of threads where rs1 != 0

# assembler.py — New assembly mnemonics:
#   SHFL  rd, rs1, lane, mode
#   VOTE.ANY rd, rs1
#   VOTE.ALL rd, rs1
#   BALLOT rd, rs1

# simt_core.py — _execute_warp() handles OP_SHFL, OP_VOTE, OP_BALLOT
# No changes to other modules (backward compatible).
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 SHFL Execution — `09_warp_shfl.asm`

场景: warp_size=8，每个线程计算 `tid * 10`，然后通过 SHFL 的三种模式进行数据交换。

```
程序执行流程 / Execution Timeline:

Cycle 0-3: 标量计算阶段
  TID r1       → 每个线程: r1 = tid (0,1,2,3,4,5,6,7)
  MOV r2, 10   → 所有线程: r2 = 10
  MUL r3,r1,r2 → 每个线程: r3 = tid * 10 (0,10,20,30,40,50,60,70)
  ST r3,[0]    → mem[tid] = tid * 10

Cycle 4-5: SHFL IDX 模式 (mode=0, src_lane=0)
  SHFL r4,r3,0,0  → 所有线程从 T0 读取 r3
  T0: r4 = T0.r3 = 0
  T1: r4 = T0.r3 = 0
  T2: r4 = T0.r3 = 0
  ... (全部得到 0)
  ST r4,[8]    → mem[8+tid] = 0 ✓

Cycle 6-7: SHFL DOWN 模式 (mode=2, delta=1)
  SHFL r5,r3,1,2  → 每个线程从 tid+1 读取 r3
  T0: target = 0+1 = 1 → r5 = T1.r3 = 10
  T1: target = 1+1 = 2 → r5 = T2.r3 = 20
  T2: target = 2+1 = 3 → r5 = T3.r3 = 30
  T3: target = 3+1 = 4 → r5 = T4.r3 = 40
  T4: target = 4+1 = 5 → r5 = T5.r3 = 50
  T5: target = 5+1 = 6 → r5 = T6.r3 = 60
  T6: target = 6+1 = 7 → r5 = T7.r3 = 70
  T7: target = 7+1 = 0 → r5 = T0.r3 = 0
  ST r5,[16]   → mem[16+tid] = (tid+1)%8 * 10 ✓

Cycle 8-9: SHFL XOR 模式 (mode=3, mask=1)
  SHFL r6,r3,1,3  → 每个线程从 tid^1 读取 r3
  T0: target = 0^1 = 1 → r6 = T1.r3 = 10
  T1: target = 1^1 = 0 → r6 = T0.r3 = 0
  T2: target = 2^1 = 3 → r6 = T3.r3 = 30
  T3: target = 3^1 = 2 → r6 = T2.r3 = 20
  T4: target = 4^1 = 5 → r6 = T5.r3 = 50
  T5: target = 5^1 = 4 → r6 = T4.r3 = 40
  T6: target = 6^1 = 7 → r6 = T7.r3 = 70
  T7: target = 7^1 = 6 → r6 = T6.r3 = 60
  ST r6,[24]   → mem[24+tid] = (tid^1)*10 ✓

Cycle 10: HALT
```

### 5.2 VOTE/BALLOT Execution — `10_warp_vote.asm`

场景: warp_size=8，偶数线程 r6=1，奇数线程 r6=0。验证 VOTE.ANY、VOTE.ALL 和 BALLOT。

```
Cycle 0-4: 条件计算阶段
  TID r1         → r1 = tid
  MOV r2, 2
  DIV r3,r1,r2   → r3 = tid / 2
  MUL r4,r3,r2   → r4 = (tid/2) * 2
  SUB r5,r1,r4   → r5 = tid % 2 (偶=0, 奇=1)

Cycle 5-6: 谓词设置
  MOV r6, 0       → 所有线程: r6 = 0
  SETP.EQ r5, r0  → 偶线程(r5==0): pred=true
  @p0 MOV r6, 1   → 仅偶线程: r6 = 1

  T0 r6=1  T1 r6=0  T2 r6=1  T3 r6=0
  T4 r6=1  T5 r6=0  T6 r6=1  T7 r6=0

Cycle 7: VOTE.ANY r7, r6
  → 遍历活跃线程: T0.r6=1 ≠ 0 → result=True
  → 所有线程: r7 = 1 (至少有一个非零)
  ST r7, [0] → mem[0] = 1 ✓

Cycle 8: VOTE.ALL r8, r6
  → 遍历活跃线程: T0.r6=1 ≠ 0 继续; T1.r6=0 == 0 → result=False, break
  → 所有线程: r8 = 0 (不全是非零)
  ST r8, [1] → mem[1] = 0 ✓

Cycle 9: BALLOT r9, r6
  → 构建 bitmask:
    T0 r6=1 → bit0=1  T1 r6=0 → bit1=0
    T2 r6=1 → bit2=1  T3 r6=0 → bit3=0
    T4 r6=1 → bit4=1  T5 r6=0 → bit5=0
    T6 r6=1 → bit6=1  T7 r6=0 → bit7=0
  → mask = 0b01010101 = 0x55
  → 所有线程: r9 = 0x55
  ST r9, [2] → mem[2] = 0x55 ✓

Cycle 10-11: 全非零验证
  MOV r10, 1       → 所有线程: r10 = 1
  VOTE.ANY r11,r10 → ALL are non-zero → r11 = 1
  VOTE.ALL r12,r10 → ALL are non-zero → r12 = 1
  ST r11, [8] → mem[8] = 1 ✓
  ST r12, [9] → mem[9] = 1 ✓

Cycle 12: HALT
```

### 5.3 关键设计决策 / Key Design Decisions

1. **Not masked by active_mask for target selection**: SHFL 的目标线程选择不限于 active_mask —— 即使目标线程当前不活跃，仍然可以读取其寄存器值。这与 CUDA 的行为一致。

2. **VOTE 作用于活跃线程**: VOTE.ANY/ALL 只考虑 warp 中活跃线程 (active_mask) 的值，非活跃线程不参与规约。

3. **BALLOT 写入全体活跃线程**: BALLOT 生成的 bitmask 会广播写入所有活跃线程的 rd 寄存器，所有活跃线程获得相同结果。

4. **没有新的流水线阶段**: SHFL/VOTE/BALLOT 的延迟与标量 ALU 相同（1 周期），直接在 EXEC 阶段完成，无需新增流水线阶段或 scoreboard 修改。

5. **完全向后兼容**: 所有 Phase 0-11 的功能和测试保持不变。现有程序在 Phase 12 模拟器上运行结果相同。

## 6. Comparison with Phase 11 / 与 Phase 11 的对比

| Aspect / 方面 | Phase 11 | Phase 12 | Change / 变化 |
|---------------|----------|----------|---------------|
| **Focus** | Interactive learning console (capstone) | Warp communication primitives | NEW direction |
| **New Opcodes** | None (Phase 11 reused Phase 10 ISA) | OP_SHFL(0x30), OP_VOTE(0x33), OP_BALLOT(0x34) | ADDED: 3 opcodes |
| **New Assembly Mnemonics** | None | SHFL, VOTE.ANY, VOTE.ALL, BALLOT | ADDED: 4 mnemonics |
| **ISA File** | 87 lines | 103 lines (+16 lines) | Extended |
| **Assembler File** | ~200 lines | ~220 lines (+20 lines) | Extended |
| **simt_core.py** | ~440 lines (Phase 11 console) | ~445 lines (+5 lines for _execute_warp) | Extended |
| **Demo Programs** | demo_basic.asm, demo_divergence.asm | + 09_warp_shfl.asm, 10_warp_vote.asm | ADDED: 2 demos |
| **Test Suite** | test_phase11.py (console tests) | test_phase12.py (5 test cases) | NEW: focused on warp comm |
| **Test Cases** | Phase 11 tests | 5 tests: ISA, assembler, SHFL, VOTE/BALLOT, backward compat | ADDED |
| **Backward Compatibility** | N/A (Phase 11 is capstone) | All Phase 0-11 programs run unchanged | Maintained |
| **Learning Console** | Full interactive debugger | Console unchanged; new demos run in console | Same |
| **Pipeline Stages** | 5-stage (FETCH/DECODE/ISSUE/EXEC/WB) | Same pipeline + new opcodes in EXEC | Same |
| **Performance Impact** | N/A | SHFL/VOTE/BALLOT: 1-cycle latency (same as ALU) | Minimal |
| **CUDA Correspondence** | GDB stepi mode | __shfl_sync, __any_sync, __all_sync, __ballot_sync | Enhanced GPU model |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **No warp_id-based SHFL target**: 当前 SHFL 的目标线程基于 `thread_id` 而非 `lane_id`。多 warp 场景下，可通过 `warp_id * warp_size + thread_id` 扩展，当前未实现。

2. **VOTE 不区分 active mask 和 predicate**: 当 warp 由于分支发散（通过 SIMT Stack）而部分活跃时，VOTE 只在 active_mask 的线程上操作。更完整的实现应考虑 predicate 与 active mask 的交互。

3. **BALLOT 只支持 rs1 != 0**: CUDA 的 `__ballot_sync()` 可以接受任意谓词条件。当前实现只检查 `rs1 != 0`，但可以通过 SETP + BALLOT 组合来实现任意条件。

4. **No __syncwarp() equivalent**: 真正的 `__shfl_sync()` 需要一个同步掩码参数（mask），只有 mask 中指定的线程参与 shuffle。当前实现隐式使用 `active_mask`。

5. **No multi-warp SHFL**: 当前 SHFL 限制在单个 warp 内。未来的 GPU 架构（如 NVIDIA Ampere 及以后）支持跨 warp 的 `__shfl_sync()`，本项目暂未支持。

6. **No warp-level reduction builtins**: 缺少 `__reduce_add_sync()` 等 warp 级规约指令，这些可以通过 SHFL XOR + 循环组合实现，但当前没有提供直接指令。

7. **No PTX source-level mapping**: SHFL/VOTE/BALLOT 在汇编级别实现，没有从 PTX 到机器指令的源代码映射。

8. **Demo program coverage limited**: 当前 demo 覆盖了基本场景。额外场景（如 UP 模式、发散中的 VOTE、全零 BALLOT）可以扩展。

9. **Performance simulation not modeled**: SHFL 在真实硬件上可能具有与普通 ALU 操作不同的延迟和吞吐量特性，当前模拟器未对这些差异建模。
