# Phase 10: Visualization and Toolchain / 可视化与工具链 — Design Document / 设计文档

> **对应 GPGPU-Sim**: `gpgpu-sim/visualizer.cc`, `gpgpu-sim/stat-tool`, `aerialvision/`
> **参考**: Chrome Tracing / Perfetto JSON Trace Format, GPGPU-Sim Statistical Report

## 1. Introduction / 架构概览

```
            ┌──────────────────────────────────────────────────────┐
            │              Visualization Pipeline                  │
            │                                                      │
            │   SIMTCore      TraceCollector      Visualizer       │
            │   ┌──────┐      ┌────────────┐     ┌───────────┐    │
            │   │ step │─────▶│ TraceEvent │────▶│ Timeline  │    │
            │   │      │      │ per cycle  │     │ (ASCII)   │    │
            │   │      │      │ per warp   │     │           │    │
            │   │      │      │            │     │ Stalls    │    │
            │   │      │      │            │     │ (bar      │    │
            │   │      │      │ mem_       │     │  chart)   │    │
            │   │      │      │ accesses   │     │           │    │
            │   │      │      │            │     │ Heatmap   │    │
            │   │      │      │            │     │ (ASCII)   │    │
            │   │      │      │            │     │           │    │
            │   │      │      │            │     │ JSON      │    │
            │   │      │      │            │     │ Export    │    │
            │   └──────┘      └────────────┘     └───────────┘    │
            │                                                      │
            │   trace_runner.py     visualizer.py                  │
            └──────────────────────────────────────────────────────┘
```

Phase 10 为 toygpgpu 增加了完整的可视化工具链。`TraceCollector` 记录每个周期、每个 warp 的执行事件。`Visualizer` 将这些事件渲染为 ASCII 时间线、阻塞分析柱状图、内存访问热力图，并支持 JSON 导出（兼容 Chrome Tracing / Perfetto）。

Phase 10 adds a complete visualization toolchain to toygpgpu. The `TraceCollector` records per-cycle, per-warp execution events. The `Visualizer` renders these events as ASCII timelines, stall analysis bar charts, memory access heatmaps, and supports JSON export (compatible with Chrome Tracing / Perfetto).

## 2. Motivation / 设计动机

Understanding GPU behavior requires **visualization**. GPU profiling tools like NVIDIA Nsight, ROCProfiler, and GPGPU-Sim's built-in statistics are essential for performance analysis.

理解 GPU 行为需要**可视化**。GPU 性能分析工具如 NVIDIA Nsight、ROCProfiler 以及 GPGPU-Sim 的内建统计对于性能分析至关重要。

GPGPU-Sim provides:
- **AerialVision**: Visualization of warp activities over time
- **stat-tool**: Statistical performance reports
- **Visual profiler**: Pipeline state visualization

Before Phase 10, the only way to understand execution was the interactive console's per-cycle display. This made it impossible to:
- See the big picture of execution over hundreds of cycles
- Identify performance bottlenecks statistically
- Export data for external analysis tools
- Visualize memory access patterns

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Trace Event Model / 追踪事件模型

```python
@dataclass
class TraceEvent:
    cycle: int           # global cycle number
    warp_id: int         # warp identifier
    pc: int              # program counter of executed instruction
    opcode: int          # opcode value
    opcode_name: str     # human-readable opcode name
    active_threads: int  # number of active threads
    stall_reason: str    # "" = no stall, "sb" = scoreboard, "barrier", "ibuffer"
    mem_addr: int        # memory address for LD/ST instructions (-1 if none)
```

### 3.2 Trace Collection Algorithm / 追踪收集算法

```python
def run_with_trace(simt, max_cycles=500):
    collector = TraceCollector()
    for cycle in range(max_cycles):
        # 1. Check for execution or stall
        warp = simt.scheduler.select_warp()
        if warp is None:
            # All warps stalled: record stall events
            for w in simt.warps:
                if not w.done:
                    reason = 'sb' if w.scoreboard_stalled else 'barrier'
                    collector.record_stall(cycle, w.warp_id, reason)
            # Advance pipeline without execution
            advance_pipeline(simt)
            continue

        # 2. Try to issue from I-Buffer
        entry = warp.ibuffer.consume()
        if entry is not None:
            instr = decode(entry.instruction_word)
            # Scoreboard check
            if has_hazard(warp.scoreboard, instr):
                warp.scoreboard_stalled = True
                warp.ibuffer.write(entry.instruction_word, entry.pc)
            else:
                # Execute and record
                simt._execute_warp(warp, instr, entry.pc)
                mem_addr = instr.imm if is_mem_op(instr.opcode) else -1
                collector.record_exec(cycle, warp.warp_id, entry.pc,
                    instr.opcode, opcode_name, active_count, mem_addr)

        # 3. Advance pipeline
        advance_scoreboards(simt)
        fetch_decode(simt)

    return collector
```

### 3.3 ASCII Warp Timeline / ASCII Warp 时间线

Each row = 1 cycle, each column = 1 warp, showing 4-char opcode abbreviation:

```
Warp Timeline (first 20 cycles)
 Cyc | W 0   W 1
------------------
   0 | [TID ] [    ]
   1 | [ LD ] [    ]
   2 | [ LD ] [    ]
   3 | [ADD ] [TID ]
   4 | [ ST ] [ LD ]
  ... | ...   ...
```

Algorithm: Group `TraceEvent` by cycle, then by warp, then format opcode name to 4 chars.

### 3.4 Stall Analysis / 阻塞分析

Aggregate stall reasons into a horizontal bar chart:

```python
def stall_analysis(events):
    reasons = {}            # stall_reason → count
    total_stalls = 0
    for e in events:
        if e.stall_reason:
            reasons[e.stall_reason] += 1
            total_stalls += 1

    max_bar = 40            # max bar width
    for reason, count in sorted(reasons.items(), key=-count):
        bar_len = count / total_stalls * max_bar
        bar = '█' * bar_len
        output += f"  {reason:<12} {count:>4d} |{bar}"
```

### 3.5 Memory Heatmap / 内存热力图

Density map showing address access frequency over 10 levels:

```python
density_chars = ' .:-=+*#%@'
# 10 density levels from 0 (space) to 9 (@)

for each address:
    density = count / max_count * 9
    char = density_chars[int(density)]

# Output:
#    0 | . . . + * @ % . . . . . . . . . . . . . . . . . . . . . . . .|
#   32 | . . . . . . . . + + * * @ @ % . . . . . . . . . . . . . . . .|
```

### 3.6 JSON Export / JSON 导出

Perfetto-compatible JSON trace format:

```python
def export_json(filepath):
    trace = []
    for e in events:
        trace.append({
            'cycle': e.cycle,
            'warp': e.warp_id,
            'pc': e.pc,
            'op': e.opcode_name,
            'active': e.active_threads,
            'stall': e.stall_reason or None,
            'mem_addr': e.mem_addr if e.mem_addr >= 0 else None,
        })
    json.dump({
        'events': trace,
        'total_cycles': collector.total_cycles,
        'stall_rate': stall_cycles / total_cycles,
    }, file, indent=2)
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| File | Purpose / 用途 | Change from Phase 9 |
|------|---------------|---------------------|
| `visualizer.py` | `TraceCollector`, `TraceEvent` dataclass, `warp_timeline()`, `stall_analysis()`, `memory_heatmap()`, `export_json()`, `full_report()` | NEW module |
| `trace_runner.py` | `run_with_trace()`: wrapper executing SIMTCore with event recording; `trace_and_report()`: full pipeline | NEW module |
| `learning_console.py` | Added `trace`, `timeline`, `report` commands | NEW commands |
| `isa.py` | Unchanged from Phase 9 | Same |
| `simt_core.py` | Unchanged from Phase 9 | Same |
| `assembler.py` | Unchanged from Phase 9 | Same |
| `ptx_parser.py` | Unchanged from Phase 8 | Same |
| `alu.py`, `vec4_alu.py` | ALU modules | Same |
| `warp.py` | Warp/Thread with PRED, warp_regs | Same |
| `ibuffer.py` | IBuffer with peek()+reconv fix | Same |
| `memory.py` | Global memory | Same |
| `cache.py` | L1 cache | Same |
| `scoreboard.py` | Scoreboard | Same |
| `scheduler.py` | Warp scheduler | Same |
| `simt_stack.py` | SIMT divergence | Same |
| `operand_collector.py` | Multi-bank register file | Same |
| `gpu_sim.py` | Top-level GPU simulator | Same |

### 4.2 Key Data Structures / 关键数据结构

```python
@dataclass
class TraceEvent:
    cycle: int
    warp_id: int
    pc: int
    opcode: int
    opcode_name: str
    active_threads: int
    stall_reason: str = ""
    mem_addr: int = -1

class TraceCollector:
    events: list[TraceEvent]         # all recorded events
    mem_accesses: list[tuple]        # (cycle, warp_id, addr, is_write)
    total_cycles: int
    stall_cycles: int

    def record_exec(self, cycle, warp_id, pc, opcode, opcode_name, active, mem_addr=-1)
    def record_stall(self, cycle, warp_id, reason)
    def export_json(self, filepath)
```

### 4.3 Module Interfaces / 模块接口

```python
# visualizer.py
def warp_timeline(events: list[TraceEvent], num_warps: int,
                  max_cycles: int = 80, width: int = 60) -> str
def stall_analysis(events: list[TraceEvent]) -> str
def memory_heatmap(mem_accesses: list[tuple],
                   mem_size: int = 256, width: int = 32) -> str
def full_report(collector: TraceCollector,
                num_warps: int, mem_size: int = 256) -> str

# trace_runner.py
def run_with_trace(simt, max_cycles=500) -> TraceCollector
def trace_and_report(simt, num_warps=1, mem_size=256,
                     json_path=None) -> tuple[str, TraceCollector]

# learning_console.py (new commands)
def print_trace_buffer(collector)          # Show trace event buffer
def print_timeline(collector, num_warps)   # Show warp ASCII timeline
def print_stall_analysis(collector)         # Show stall bar chart
def print_full_report(collector, num_warps, mem_size)  # Full report
```

## 5. Functional Processing Flow / 功能处理流程

```
=== trace_and_report("program.asm") ===

Step 1: Run with trace collection
  ┌─────────────────────────────────────────────┐
  │ Cycle 0: ISSUE TID r1  (warp=0, active=4)   │
  │          FETCH PC=1 → IBuffer                │
  │ Cycle 1: ISSUE LD r2, [0]  (warp=0, active=4)│
  │          FETCH PC=2 → IBuffer                │
  │ Cycle 2: STALL (scoreboard)                  │
  │ Cycle 3: ISSUE ADD r3, r1, r2 (warp=0)       │
  │ ...                                          │
  │ Total: 12 cycles, 8 events, 2 stalls         │
  └─────────────────────────────────────────────┘

Step 2: Generate visual report

  ╔══════════════════════════════════════════════════╗
  ║         toygpgpu Execution Report                ║
  ╚══════════════════════════════════════════════════╝

  Warp Timeline (first 12 cycles):
   Cyc |  W 0
  -----|---------
     0 | [TID ]
     1 | [ LD ]
     2 | [----]  ← stall (scoreboard)
     3 | [ LD ]
     4 | [ADD ]
    ...

  Stall Analysis:
  Total events: 10, Stall events: 2 (20.0%)
    sb            2 |██████

  Memory Access Heatmap (0-255, width=32):
     0 | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|
    32 | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|
    16 | @ + + + . . . . . . . . . . . . . . . . . . . . . . . . . . . .|
    ...

Step 3: Export JSON (for Perfetto Chrome Tracing)
  → trace_output.json
  Events: 10, total_cycles: 12, stall_rate: 0.167
```

## 6. Comparison with Phase 9 / 与 Phase 9 的对比

| Aspect / 方面 | Phase 9 | Phase 10 | Change / 变化 |
|---------------|---------|----------|---------------|
| **Execution View** | Per-cycle console display | Full execution trace + aggregate reports | NEW perspective |
| **Data Collection** | None (transient state) | `TraceCollector` records all events | NEW data pipeline |
| **Event Model** | N/A | `TraceEvent` dataclass (cycle, warp, pc, opcode, stall, mem_addr) | NEW data structure |
| **Timeline** | Per-cycle step | ASCII warp timeline over N cycles | NEW visualization |
| **Stall Analysis** | None | Stall reason bar chart with percentages | NEW analysis |
| **Memory Heatmap** | None | Address frequency density map (10 levels) | NEW visualization |
| **Export Format** | None | JSON (Chrome Tracing/Perfetto compatible) | NEW export |
| **Trace Runner** | None | `trace_runner.py`: `run_with_trace()`, `trace_and_report()` | NEW module |
| **Visualizer** | None | `visualizer.py`: 4 visualization functions + report | NEW module |
| **Console Commands** | `mma` | `trace`, `timeline`, `report` | NEW commands |
| **Shared Features** | IBuffer.peek()+reconv, PRED, warp_regs, vec4_alu, trace mode | Same + enhanced trace mode | Extended |
| **Performance** | N/A | Trace overhead: O(cycles × warps) events | Added collection cost |

## 7. Known Issues and Future Work / 遗留问题与后续工作

1. **ASCII-only output**: No HTML or graphical rendering. Timeline and heatmap are limited to terminal display.
2. **Trace memory**: For long runs (>10000 cycles × 32 warps), event list can grow large. No sampling or aggregation during collection.
3. **Stall attribution accuracy**: Stall reasons are simplified (sb, barrier, ibuffer). Real GPUs have more nuanced stall models (scoreboard RAW/WAW, structural hazards, memory latency, TLB misses, etc.).
4. **JSON format is custom**: Not fully compatible with Chrome Tracing's `trace_event` format (which uses `ph`, `ts`, `dur`, `pid`, `tid` fields). Would need a converter for full Perfetto support.
5. **No heatmap over time**: The memory heatmap aggregates access counts over all cycles. No temporal component (access frequency per time window).
6. **No IPC/occupancy report**: Real GPU profilers show occupancy, achieved IPC, memory bandwidth utilization. These require more detailed counters.
7. **No pipeline stage breakdown in trace**: Traces show issue events but don't break down per-pipeline-stage timing.
