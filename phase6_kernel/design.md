# Phase 6: Kernel Launch — Design Document / 设计文档

> **对应 GPGPU-Sim**: `gpgpu_sim` 顶层类 (`gpgpu-sim/gpu-sim.cc`), `shader_core_ctx` multi-SM 模型, GTO scheduler (`scheduler_unit`)
> **参考**: GPGPU-Sim kernel launch flow, CUDA grid/block/warp 层次, GTO scheduling

## 1. Introduction / 架构概览

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        GPUSim (Phase 6)                              │
  │                                                                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                           │
  │  │Block 0   │  │Block 1   │  │Block N-1 │                           │
  │  │SIMTCore  │  │SIMTCore  │  │SIMTCore  │                           │
  │  │┌────────┐│  │┌────────┐│  │┌────────┐│                           │
  │  ││Warp 0  ││  ││Warp 0  ││  ││Warp 0  ││                           │
  │  ││Warp 1  ││  ││Warp 1  ││  ││Warp 1  ││                           │
  │  ││...     ││  ││...     ││  ││...     ││                           │
  │  │└────────┘│  │└────────┘│  │└────────┘│                           │
  │  │L1/Shared│  │L1/Shared│  │L1/Shared│                           │
  │  └──────────┘  └──────────┘  └──────────┘                           │
  │                                                                      │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │  PerfCounters                                                 │   │
  │  │  total_cycles | total_instructions | IPC | stall_*            │   │
  │  └──────────────────────────────────────────────────────────────┘   │
  │                                                                      │
  │  Kernel Launch: grid_dim × block_dim → total_blocks × total_threads │
  │  GTO Scheduler: greedy-then-oldest, oldest-ready warp first         │
  └──────────────────────────────────────────────────────────────────────┘
```

Phase 6 整合 Phase 0-5 所有模块，提供 CUDA 风格的内核启动 (kernel launch) 模型。支持 grid/block 维度配置、GTO 调度策略和性能计数器 (IPC/stall 统计)。这是 toygpgpu 的"系统级"整合阶段。

## 2. Motivation / 设计动机

Phase 5 之前的版本仅支持单个 SIMTCore 实例手动执行。真实 GPU 以 CUDA kernel launch 为执行模型：

- **Kernel Launch**：主机端调用 `kernel<<<grid, block>>>()`，GPU 自动将线程组织为 grid→block→warp 层次。
- **多 SM 架构**：现代 GPU 有数十个 Streaming Multiprocessors (SM)，每个 SM 运行多个 Thread Block。
- **Warp Scheduling Policy**：Round-Robin (RR) 在 warp 间公平轮转；Greedy-Then-Oldest (GTO) 优先执行完一个 warp 再切换，通常具有更好的缓存局部性和更低的上下文切换开销。
- **Performance Counters**：IPC (Instructions Per Cycle) 是衡量 GPU 性能的核心指标。

GPGPU-Sim 的 `gpgpu_sim::launch()` 负责创建 grid 结构、分配 block 到 SM、初始化性能计数器。`scheduler_unit` 实现 GTO/RR 等多种调度策略。

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 Kernel Launch

```
launch_kernel(program, grid_dim, block_dim):
    total_blocks = product(grid_dim)     # 例如 (2,) → 2 blocks
    total_threads = product(block_dim)   # 例如 (8,) → 8 threads
    num_warps_per_block = ceil(total_threads / warp_size)
    
    for block_id in range(total_blocks):
        core = SIMTCore(warp_size, num_warps_per_block)
        core.load_program(program)
        cores.append(core)
    
    # 输出: blocks × warps × threads
    # 例如: 2 blocks × 1 warp × 8 threads/warp = 16 threads
```

### 3.2 GTO Scheduler

GTO (Greedy-Then-Oldest) 是 GPGPU-Sim 的默认调度策略。

```
_select_gto():
    candidates = [w for w in warps
                  if not w.done AND not w.at_barrier AND not w.scoreboard_stalled]
    if candidates is empty: return None
    
    # 选最久未执行的 warp (oldest-ready)
    warp = min(candidates, key=lambda w: _warp_last_issue[w.warp_id])
    _warp_last_issue[warp.warp_id] = cycle_count++
    return warp
```

与 RR 的对比：

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| RR (Round-Robin) | 轮转选择，每个 warp 轮流执行 1 指令 | 公平吞吐，无局部性优势 |
| GTO (Greedy-Then-Oldest) | 优先执行同一 warp 直到 stall，然后选最老的 | 缓存局部性好，减少上下文切换 |

### 3.3 PerfCounters

```
PerfCounters:
    total_cycles: int           # 总周期数
    total_instructions: int     # 总指令数
    stall_scoreboard: int       # scoreboard 停顿周期
    stall_barrier: int          # barrier 等待周期
    active_cycles: int          # 有指令发射的周期
    
    ipc = total_instructions / total_cycles   # 每周期指令数
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| 文件 | 作用 |
|------|------|
| `gpu_sim.py` | GPUSim 顶层类 + PerfCounters + launch_kernel + run + report |
| `simt_core.py` | SIMTCore (同 Phase 5，含 L1 Cache + Shared Memory) |
| `scheduler.py` | WarpScheduler 新增 GTO 策略 (RR 保留) |
| `learning_console.py` | 交互式控制台，新增 `perf` 命令，支持多 block 调试 |
| `cache.py` | L1 数据缓存 (同 Phase 5) |
| `shared_memory.py` | 共享内存 (同 Phase 5) |
| `thread_block.py` | Thread Block 容器 (同 Phase 5) |
| `scoreboard.py` | 记分板 (同 Phase 4) |
| `isa.py` | 指令集 (同 Phase 5) |

### 4.2 Key Data Structures / 关键数据结构

```python
class PerfCounters:
    total_cycles: int              # 总周期
    total_instructions: int        # 总指令数
    stall_scoreboard: int          # scoreboard 停顿
    stall_barrier: int             # barrier 等待
    stall_branch: int              # 分支停顿
    active_cycles: int             # 活跃周期
    
    @property ipc() -> float       # IPC 计算

class GPUSim:
    cores: list[SIMTCore]          # 每个 block 一个 core
    perf: PerfCounters             # 性能计数器
    
    def launch_kernel(program, grid_dim, block_dim)
    def run()                       # 顺序执行所有 core
    def report()                    # 打印性能报告

class WarpScheduler (Phase 6):
    policy: str = "rr" | "gto"     # 调度策略
    _warp_last_issue: list[int]    # 每个 warp 最后发射周期 (GTO 用)
    _select_gto()                   # GTO 策略
    _select_rr()                    # RR 策略
```

### 4.3 Module Interfaces / 模块接口

```
GPUSim::run():
    while any core active:
        total_cycles++
        for each core:
            for each warp:
                scoreboard.advance()
                if stalled: perf.stall_scoreboard++
                if at_barrier: perf.stall_barrier++
            warp = scheduler.select_warp()
            if warp is None: continue
            fetch → decode → execute
            perf.total_instructions++
            sb.reserve(rd, latency)
    perf.active_cycles = total_cycles - total_stalls
```

## 5. Functional Processing Flow / 功能处理流程

### 示例 1: GTO 调度 (`01_gto_schedule.asm`)

```
配置: warp_size=8, num_warps=2 (per block), grid_dim=1, block_dim=16
程序: WID r1; MOV r2,10; MUL r3,r1,r2; TID r4; ADD r5,r3,r4; ST r5,[100]; HALT

GTO 调度输出 (trace 模式):
──────────────────────────────────────────────────────────
Cycle 0: Warp 0: WID r1  | reg: W0 T0-7: r1=0
Cycle 1: Warp 0: MOV r2,10 | reg: W0 T0-7: r2=10
Cycle 2: Warp 0: MUL r3,r1,r2 | reg: W0 T0-7: r3=0
Cycle 3: Warp 0: TID r4 | reg: W0 T0:r4=0 T1:r4=1 ... T7:r4=7
Cycle 4: Warp 0: ADD r5,r3,r4 | reg: W0 T0:r5=0 T1:r5=1 ... T7:r5=7
Cycle 5: Warp 0: ST r5,[100] | mem[100..107]=[0,1,2,3,4,5,6,7]
Cycle 6: Warp 0: HALT
Cycle 7: Warp 1: WID r1  | reg: W1 T0-7: r1=1
Cycle 8: Warp 1: MOV r2,10 | reg: W1 T0-7: r2=10
...

对比 RR (交替):
Cycle 0: Warp 0: WID r1
Cycle 1: Warp 1: WID r1
Cycle 2: Warp 0: MOV r2,10
Cycle 3: Warp 1: MOV r2,10
...
GTO 让每个 warp 连续执行减少上下文切换，RR 在 warp 间公平交替。
```

### 示例 2: 多 block (`02_multi_block.asm`)

```
配置: grid_dim=2, block_dim=8, warp_size=8
程序: TID r1; MOV r2,2; MUL r3,r1,r2; ST r3,[50]; HALT

GPUSim 创建 2 个 SIMTCore (Block 0, Block 1):
  Block 0: warp 0, mem[50..57] = [0,2,4,6,8,10,12,14]
  Block 1: warp 0, mem[50..57] = [0,2,4,6,8,10,12,14] (覆盖写入)
  
总线程数: 2 blocks × 8 threads = 16 threads
```

## 6. Comparison with Phase 5 / 与前一版本的对比

| Aspect | Phase 5 (Memory) | Phase 6 (Kernel) | Change |
|--------|------------------|-------------------|--------|
| 顶层模型 | SIMTCore (单 core) | GPUSim (多 core) | 系统级封装 |
| 程序执行 | `core.load_program()` | `gpu.launch_kernel(prog, grid, block)` | kernel launch |
| Block 数量 | 1 (单 block) | 任意 (grid_dim) | 多 block |
| 调度策略 | RR 轮询 | RR + GTO (贪心最老) | GTO 策略 |
| 性能计数器 | 无 | PerfCounters (IPC, stalls) | 性能分析 |
| 学习控制台 | `cache`/`smem` 命令 | 新增 `perf` 命令 + 多 block 视图 | 调试增强 |
| warp 选择考量 | scoreboard + barrier | scoreboard + barrier + GTO age | GTO 优先级 |
| 新增文件 | — | `gpu_sim.py` | 1 个新文件 |

## 7. Known Issues and Future Work / 遗留问题与后续工作

- **顺序执行 block**：当前 block 按顺序执行，未模拟真实 GPU 的并行 SM 调度。后续可引入并行 SM 模型和 block 分配器。
- **无 block 间同步**：CUDA 不支持 block 间同步（除全局内存 fence），但真实 GPU 的 block 调度顺序影响结果。当前简化忽略此问题。
- **GTO 简化**：GTO 实现仅追踪 `_warp_last_issue`，未模拟 GPGPU-Sim 更细粒度的 warp 优先级和得分机制。
- **无指令缓存 (I-Cache)**：程序直接存储在列表中，未模拟指令缓存 miss 惩罚。
- **无功耗建模**：GPGPU-Sim 包含功耗/能量统计（`gpgpusim_power`），Phase 6 未涉及。
- **PerfCounters 粒度**：当前为全局计数器，不支持 per-warp 或 per-block 的细粒度性能档案。

