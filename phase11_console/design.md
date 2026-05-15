# Phase 11: Interactive Learning Console / 交互式学习控制台 — Design Document / 设计文档

> **对应 GPGPU-Sim**: `gpgpu-sim/debug.h`, `gpgpu-sim/gpu-sim.cc` (debug mode), `gpgpu-sim/stat-tool`
> **参考**: GDB (GNU Debugger) stepi mode, NVIDIA Nsight Compute Interactive Profiler

## 1. Introduction / 架构概览

```
            ┌──────────────────────────────────────────────────────┐
            │           Interactive Console Architecture           │
            │                                                      │
            │   ┌──────────────┐     ┌───────────────────┐        │
            │   │  User Input  │────▶│ learning_console  │        │
            │   │  (keyboard)  │     │  .py              │        │
            │   └──────────────┘     │  (command loop)   │        │
            │                        └─────────┬─────────┘        │
            │                                  │                  │
            │                         ┌────────▼─────────┐        │
            │                         │ console_display  │        │
            │                         │  .py             │        │
            │                         │  (render functions)│       │
            │                         └────────┬─────────┘        │
            │                                  │                  │
            │                                  ▼                  │
            │   ┌──────────────────────────────────────────────┐  │
            │   │                SIMTCore                      │  │
            │   │  (Fetch/Decode/Issue/Exec/WB Pipeline)       │  │
            │   │  (Warp + Scoreboard + IBuffer + SIMT Stack)  │  │
            │   └──────────────────────────────────────────────┘  │
            │                                                      │
            │   Per-cycle rendering:                              │
            │   ┌────────────────────────────────────────────────┐ │
            │   │ ╔══ Cycle N ═══════════════════════════════╗  │ │
            │   │   W0:PC=N mask=0b1111 [ACT]                 │  │ │
            │   │   ┌─ Pipeline ──────────────────────────┐   │  │ │
            │   │   │ FETCH │ DECODE │ ISSUE │ EXEC │ WB │   │  │ │
            │   │   ├─ Register Changes ──────────────────┤   │  │ │
            │   │   ├─ Scoreboard ────────────────────────┤   │  │ │
            │   │   ├─ I-Buffer ──────────────────────────┤   │  │ │
            │   │   ├─ SIMT Stack ────────────────────────┤   │  │ │
            │   │   ├─ Memory Changes ────────────────────┤   │  │ │
            │   │   ╚═════════════════════════════════════╝   │  │ │
            │   └────────────────────────────────────────────────┘ │
            └──────────────────────────────────────────────────────┘
```

Phase 11 是 toygpgpu 的**集成学习控制台**（capstone 阶段）。它整合了 Phase 0-10 的所有功能，提供一个 GDB 风格的交互式调试器。用户可以通过 ANSI 着色渲染的流水线显示、断点、和每个模块的状态检查，逐周期观察 GPU 行为。

Phase 11 is the **integrated learning console** (capstone phase) of toygpgpu. It integrates all features from Phases 0-10, providing a GDB-style interactive debugger. Users can observe GPU behavior cycle-by-cycle through ANSI-colored pipeline displays, breakpoints, and per-module state inspection.

## 2. Motivation / 设计动机

Learning GPU architecture requires **observability**. GDB-style step-through debugging with GPU-specific state views is essential for understanding:

学习 GPU 架构需要**可观察性**。具有 GPU 特定状态视图的 GDB 风格单步调试对于理解以下内容至关重要：

- How the SIMT pipeline processes instructions
- How warp divergence and reconvergence work
- How scoreboards track register hazards
- How the operand collector avoids bank conflicts
- How memory coalescing works
- How tensor core MMA instructions pack/unpack data

Before Phase 11, the console was functional but basic. The capstone phase adds:
- **ANSI box-drawing** pipeline display with colored pipeline stages
- **Separated rendering module** (`console_display.py`) for clean architecture
- **Auto-step mode** with configurable interval
- **Dedicated demo programs** (`demo_basic.asm`, `demo_divergence.asm`)
- **run.sh** with `--trace` mode support

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Interactive Loop / 交互式主循环

```python
def run_console(simt, program_text, args):
    simt.load_program(assemble(program_text))
    breakpoints = set()
    cycle = 0

    while cycle < max_cycles:
        # 1. Wait for command or auto-step
        cmd = input(f"[{cycle}] > ").strip()

        # 2. Dispatch command
        if cmd == '' or cmd == 's':       # Single step
            _do_step(simt, cycle, ...)
        elif cmd == 'r':                  # Run to completion
            auto_step = True
        elif cmd == 'r N':                # Run N cycles
            for _ in range(N): _do_step(...)
        elif cmd == 'i':                  # Print state
            print_state(simt)
        elif cmd == 'b <pc>':             # Set breakpoint
            breakpoints.add(pc)
        elif cmd == 'q':                  # Quit
            break

        # 3. Execute one cycle
        simt.step()

        # 4. Render pipeline display
        render_cycle(cycle, simt, instr_info, old_regs, mem_changes, stage_info)

        # 5. Check breakpoints
        if warp.pc - 1 in breakpoints:
            auto_step = False
```

### 3.2 Pipeline Stage Collection / 流水线阶段收集

```python
def _do_step(simt, cycle, prev_regs, prev_mem, breakpoints):
    old_regs = snapshot_regs(simt)
    old_mem = snapshot_mem(simt)

    # Collect pipeline info before execution
    warp = None; instr = None
    stage_info = {'fetch': '', 'decode': '', 'issue': '', 'exec': '', 'wb': ''}
    for w in simt.warps:
        if not w.done and not w.scoreboard_stalled and not w.at_barrier:
            for e in w.ibuffer.entries:
                if e.valid and e.ready:
                    instr = decode(e.instruction_word)
                    warp = w
                    opname = OPCODE_NAMES.get(instr.opcode, '?')
                    stage_info['fetch'] = f"PC={e.pc} → IBuffer"
                    stage_info['decode'] = f"{opname} decoded"
                    stage_info['issue'] = f"Scoreboard check, bank check"
                    stage_info['exec'] = f"{opname} r{instr.rd}=r{instr.rs1} op r{instr.rs2}"
                    stage_info['wb'] = f"r{instr.rd} ← result (latency)"
                    break
            if instr: break

    # Execute one cycle
    simt.step()

    # Compute changes
    new_regs = snapshot_regs(simt)
    new_mem = snapshot_mem(simt)
    mem_changes = mem_diff(old_mem, new_mem)

    # Render
    render_cycle(cycle, simt, instr_info, old_regs, mem_changes, stage_info)
    return warp.pc - 1 in breakpoints  # Check breakpoints
```

### 3.3 ANSI Rendering Architecture / ANSI 渲染架构

The `console_display.py` module is a **pure rendering module** (no state). It provides stateless functions that take a SIMTCore snapshot and produce rendered strings:

```
render_cycle(cycle, simt, instr_info, prev_regs, mem_changes, stage_info) → str
    ├─ render warp status lines (colored: green for ACT, yellow for BAR, gray for DONE)
    ├─ render pipeline stages (FETCH/gray, DECODE/blue, ISSUE/yellow, EXEC/green, WB/magenta)
    ├─ render register changes (yellow highlight for changed values)
    ├─ render scoreboard (red if pending, green if clean)
    ├─ render I-Buffer (valid entries with opcode, PC, ready/wait indicator)
    ├─ render SIMT stack (if active: reconv PC, original mask, taken mask)
    ├─ render memory changes (address, old value → new value)
    └─ render OpCollector + L1Cache stats
```

ANSI color scheme:
| Color | Usage / 用途 |
|-------|-------------|
| Cyan | Box-drawing frame borders |
| Green | Active state, correct values, ready indicator |
| Red | Stall state, errors, waiting indicator |
| Yellow | Data changes, breakpoint hit |
| Gray | Inactive, empty, done state |
| Blue | Decode pipeline stage |
| Magenta | Writeback pipeline stage |

### 3.4 Command Dispatch / 命令分发

```
┌──────────────┬─────────────────────────────────────┐
│ Command      │ Action                              │
├──────────────┼─────────────────────────────────────┤
│ Enter / s    │ Step one cycle                      │
│ r            │ Run until all warps complete         │
│ r N          │ Run N cycles                        │
│ i            │ Print current state (PC, mask, etc.) │
│ m            │ Print non-zero memory locations      │
│ reg          │ Print all register values            │
│ wreg         │ Print warp-level uniform registers   │
│ sb           │ Print scoreboard state               │
│ ib           │ Print I-Buffer entries               │
│ stack        │ Print SIMT stack depth and entries   │
│ oc           │ Print operand collector bank status  │
│ mma          │ Print MMA/tensor register details    │
│ b <pc>       │ Set breakpoint at PC                │
│ b list       │ List all breakpoints                │
│ b clear      │ Clear all breakpoints               │
│ q            │ Quit console                        │
└──────────────┴─────────────────────────────────────┘
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Purpose / 用途 | Change from Phase 10 |
|------|---------------|---------------------|
| `learning_console.py` | Interactive loop, command dispatch, all print helpers | REFACTORED: cleaner loop, imports from console_display |
| `console_display.py` | Pure rendering functions: `render_cycle()`, `snapshot_regs()`, `snapshot_mem()`, `mem_diff()` | NEW module (extracted from Phase 10's learning_console.py) |
| `isa.py` | Opcode definitions (ALL phases: 0-10 + MMA) | Same as Phase 9 |
| `simt_core.py` | SIMT pipeline + ALL instructions | Same as Phase 9 |
| `assembler.py` | Two-pass assembler with ALL opcodes | Same as Phase 9 |
| `warp.py` | Warp/Thread with PRED, warp_regs, IBuffer, Scoreboard, SIMT stack | Same |
| `ibuffer.py` | IBuffer with peek()+reconv fix | Same (FIXED from Phase 7) |
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
| `run.sh` | Wrapper script with `--trace` mode | NEW |

### 4.2 Key Data Structures / 关键数据结构

```python
# console_display.py (stateless rendering module)
# No classes — pure functions:

def render_cycle(cycle, simt, instr_info, prev_regs, mem_changes, stage_info) -> str
def snapshot_regs(simt) -> dict              # {warp_id: {thread_id: [regs]}}
def snapshot_mem(simt) -> list               # list of memory words
def mem_diff(old_mem, new_mem) -> list       # [(addr, old, new)]

# Private helpers:
def _render_reg_changes(simt, prev_regs) -> list[str]
def _render_scoreboard(simt) -> list[str]
def _render_ibuffer(simt) -> list[str]
def _render_simt_stack(simt) -> list[str]
def _s32(val) -> int                          # unsigned → signed conversion

# learning_console.py (interactive loop)
# State: breakpoints: set       # program counter breakpoints
#         auto_step: bool       # run-until-bp mode
#         cycle: int            # current cycle count
#         prev_regs, prev_mem   # snapshot for diff computation
```

### 4.3 Module Interfaces / 模块接口

```python
# console_display.py
def render_cycle(cycle, simt, instr_info, prev_regs, mem_changes, stage_info) -> str

# learning_console.py
def run_console(simt, program_text, args)  # Main interactive loop
def print_state(simt)                       # Current state summary
def print_memory(simt)                      # Non-zero memory
def print_registers(simt)                   # All registers
def print_warp_regs(simt)                   # Warp-level registers
def print_scoreboard(simt)                  # Scoreboard status
def print_ibuffer(simt)                     # I-Buffer entries
def print_simt_stack(simt)                  # SIMT stack entries

# run.sh
# --trace: runs in auto mode with trace output
# Default: runs test suite and shows demo command
```

### 4.4 Rendering Layout / 渲染布局

```
╔══ Cycle N ══════════════════════════════════════════════════════╗
  W0:PC=N mask=0b1111 [ACT] │ W1:PC=N mask=0b0000 [DONE]
  ┌─ Pipeline ──────────────────────────────────────────────────┐
  │ FETCH  │ PC=N → IBuffer                                      │
  │ DECODE │ ADD decoded, scoreboard check                       │
  │ ISSUE  │ operand bank check                                  │
  │ EXEC   │ ADD r3=r1 op r2 executing on 4 threads              │
  │ WB     │ r3 ← result (latency)                               │
  └──────────────────────────────────────────────────────────────┘

  ┌─ Register Changes ──────────────────────────────────────────┐
  │ W0 T0 r1: 0x00000000 → 0x00000005 (5)                       │
  │ W0 T0 r2: 0x00000000 → 0x00000003 (3)                       │
  └──────────────────────────────────────────────────────────────┘

  ┌─ Scoreboard ────────────────────────────────────────────────┐
  │ W0: r3:2 (red) means r3 reserved for 2 more cycles          │
  └──────────────────────────────────────────────────────────────┘

  ┌─ I-Buffer ──────────────────────────────────────────────────┐
  │ W0: [ADD  PC= 2 ✓] [empty]                                  │
  └──────────────────────────────────────────────────────────────┘

  OpCollector: 4 banks, 12 reads, 2 conflicts (16.7%)  |  L1Cache: 4 hits, 2 misses
╚══════════════════════════════════════════════════════════════════╝
```

## 5. Functional Processing Flow / 功能处理流程

### 5.1 Stepping Through `demo_basic.asm`

```
User loads: python3 learning_console.py tests/programs/demo_basic.asm

Console startup:
  ╔══════════════════════════════════════════════════╗
  ║     toygpgpu Learning Console — 交互式调试器      ║
  ║  Program: 6 instructions                         ║
  ║  Config:  1 warp(s) × 4 threads/warp             ║
  ╚══════════════════════════════════════════════════╝

  ─── Program ───
    PC  0: MOV    rd=r1 rs1=r0 rs2=r0 imm=5
    PC  1: MOV    rd=r2 rs1=r0 rs2=r0 imm=3
    PC  2: ADD    rd=r3 rs1=r1 rs2=r2 imm=0
    PC  3: MUL    rd=r4 rs1=r3 rs2=r2 imm=0
    PC  4: ST     rd=r0 rs1=r4 rs2=r0 imm=0
    PC  5: HALT   rd=r0 rs1=r0 rs2=r0 imm=0

User presses Enter (step):
  ╔══ Cycle 0 ════════════════════════════════════════╗
    W0:PC= 1 mask=0b1111 [ACT]
    ┌─ Pipeline ─────────────────────────────────────┐
    │ FETCH  │ PC=0 → IBuffer                          │
    │ DECODE │ MOV decoded, scoreboard check           │
    │ ISSUE  │ operand bank check                      │
    │ EXEC   │ MOV r1=r0 op r0 executing on 4 threads  │
    │ WB     │ r1 ← result (latency)                   │
    ├─ Register Changes ──────────────────────────────┤
    │ W0 T0 r1: 0x00000000 → 0x00000005 (5)           │
    │ W0 T1 r1: 0x00000000 → 0x00000005 (5)           │
    │ W0 T2 r1: 0x00000000 → 0x00000005 (5)           │
    │ W0 T3 r1: 0x00000000 → 0x00000005 (5)           │
    ├─ Scoreboard ────────────────────────────────────┤
    │   (clean)                                        │
    ├─ I-Buffer ──────────────────────────────────────┤
    │ W0: [MOV  PC= 0 ✓] [empty]                      │
    ╚══════════════════════════════════════════════════╝

User presses Enter 5 more times:
  Cycle 5: HALT → warp done
  ✓ All warps completed at cycle 6

  ─── Final State ───
  Cycles executed: 6
  Memory (non-zero):
    mem[  0]=0x00000018(24)   ← result stored
```

### 5.2 Stepping Through `demo_divergence.asm`

```
Key observation points for divergence demo:

1. Cycle 0-4: Scalar setup (TID, DIV, MUL, MOV)
   → All threads active, mask=0b1111

2. Cycle 5: BEQ r4, r1, even_path
   → Divergence detected!
   → SIMT Stack: push {reconv=done, orig=0b1111, taken=0b0101 (even)}
     ↓
   → active_mask=0b0101 (even threads: T0, T2)
   → PC jumps to even_path

3. Cycle 6-7: Even path execution
   → MOV r6, 2; ST r6, [100]
   → mem[100]=2 (T0), mem[102]=2 (T2)

4. Cycle 8: JMP done
   → Merging jump: SIMT stack updates reconv_pc

5. Cycle 9: SIMT stack pop
   → Remaining mask 0b1010 (odd threads: T1, T3)
   → active_mask=0b1010, PC=fallthrough (after BEQ)

6. Cycle 10-11: Odd path execution
   → MOV r6, 1; ST r6, [100]
   → mem[101]=1 (T1), mem[103]=1 (T3)

7. Cycle 12+: Reconvergence at "done" label
   → All threads active again (mask=0b1111)
   → TID → ADD → ST mem[200+tid] = tid*2
```

## 6. Comparison with Phase 10 / 与 Phase 10 的对比

| Aspect / 方面 | Phase 10 | Phase 11 | Change / 变化 |
|---------------|----------|----------|---------------|
| **Rendering Architecture** | Inline in learning_console.py | Separate `console_display.py` module | REFACTORED |
| **Pipeline Display** | Simple ASCII with `+---` borders | ANSI box-drawing: `╔══`, `║`, `╚══`, `┌─`, `│`, `└─` | Improved |
| **Color Scheme** | Basic (green/red/gray/yellow) | Full: cyan frames, color-coded stages, yellow changes | Extended |
| **Module Export** | N/A | `from console_display import render_cycle, snapshot_regs, ...` | Cleaner API |
| **Auto-step** | `--auto` flag only | `--auto` + `--auto-interval N` (configurable delay) | Enhanced |
| **Demo Programs** | None | `demo_basic.asm`, `demo_divergence.asm` | NEW |
| **run.sh** | None | `run.sh --trace` + test suite + demo command | NEW |
| **Command Set** | Basic (step, run, info, reg, m) | Full (all Phase 10 cmds + enhanced rendering) | Same scope, better viz |
| **Breakpoints** | Per-PC breakpoints | Same + colored hit notification | Same |
| **Pipeline Stage Info** | Generic text | Step-specific: shows actual PC, opcode, active thread count | Improved |
| **SIMT Stack Display** | Text | Formatted mask bits with reconv PC | Same |
| **Trace Mode** | `--trace` for batch | `--trace` for batch + run.sh integration | Same |
| **Shared Features** | IBuffer.peek()+reconv, PRED, warp_regs, vec4_alu | Same | Stable |
| **Test Suite** | test_phase10.py | test_phase11.py | Updated |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **Terminal width limitation**: The box-drawing display can exceed typical terminal widths (80 chars) for wide warp configurations. No automatic width detection.
2. **No scrollback**: Console output scrolls off-screen. No pager integration (less/more).
3. **Single-console only**: No support for multi-window or split-pane display of different pipeline views simultaneously.
4. **No register watchpoints**: Breakpoints are PC-based only. No data watchpoints (stop when register X changes).
5. **No reverse execution**: Cannot step backward. Once past a cycle, state is lost.
6. **No command history persistence**: Command history is lost on exit. No `.learning_console_history` file.
7. **No tab completion**: No readline-style tab completion for commands and program addresses.
8. **No TUI mode**: The console is line-oriented. A future curses/TUI mode could provide a persistent header, scrollable state panels, and keyboard shortcuts.
9. **No source-level debugging**: Even with PTX support (Phase 8), the console shows only disassembled machine code, not original PTX source line mappings.
