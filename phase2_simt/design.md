# Phase 2: SIMT 核心 — 设计文档

## 1. 架构总览

对标 GPGPU-Sim 的 `shader_core_ctx` + `shd_warp_t` + `scheduler_unit`：

```
┌───────────────────────────────────────────────────────┐
│                  SIMT Core (simt_core.py)              │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │          Warp Scheduler (scheduler.py)           │ │
│  │         Round-Robin: warp_0 → warp_1 → ...       │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │ selected warp                │
│                        ▼                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ Warp 0  │  │ Warp 1  │  │ Warp 2  │  ...         │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │              │
│  │ │Th 0 │ │  │ │Th 0 │ │  │ │Th 0 │ │              │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │              │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │              │
│  │ │Th 1 │ │  │ │Th 1 │ │  │ │Th 1 │ │              │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │              │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │              │
│  │ │ ... │ │  │ │ ... │ │  │ │ ... │ │              │
│  │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │              │
│  │ │Th 7 │ │  │ │Th 7 │ │  │ │Th 7 │ │              │
│  │ │ RF  │ │  │ │ RF  │ │  │ │ RF  │ │              │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │              │
│  │ 共享PC  │  │ 共享PC  │  │ 共享PC  │              │
│  │ actmask │  │ actmask │  │ actmask │              │
│  └─────────┘  └─────────┘  └─────────┘              │
│                                                       │
│  ┌──────────────────────────────────────────────────┐│
│  │     Global Memory (all warps shared)              ││
│  └──────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────┘
```

## 2. 与 GPGPU-Sim 的精确对照

| GPGPU-Sim | Phase 2 | 说明 |
|-----------|---------|------|
| `shader_core_ctx` | `SIMTCore` | SIMT 核心顶层 |
| `shd_warp_t` | `Warp` | warp 状态：线程组、PC、active_mask |
| `shd_warp_t::m_thread[]` | `Warp.threads[]` | 每线程独立寄存器 |
| `scheduler_unit` | `WarpScheduler` | warp 调度：选下一个 warp |
| `simt_stack` | 无（Phase 3） | 分支发散/重汇聚延迟到 Phase 3 |
| `scoreboard` | 无（Phase 4） | 记分板延迟到 Phase 4 |

## 3. 核心类设计

### 3.1 Thread (warp.py)

```python
@dataclass
class Thread:
    thread_id: int
    reg_file: RegisterFile  # 16×32bit
```

### 3.2 Warp (warp.py)

```python
class Warp:
    warp_id: int
    threads: list[Thread]
    pc: int              # 共享 PC（所有线程执行同一条指令）
    active_mask: int     # bit i = 1 → thread i 活跃
    at_barrier: bool     # 是否在 barrier 等待
    barrier_count: int   # 到达 barrier 的线程数
```

### 3.3 WarpScheduler (scheduler.py)

```python
class WarpScheduler:
    warps: list[Warp]
    current_warp_idx: int
    policy: str  # "rr" (round-robin)

    def select_warp() -> Warp | None
```

### 3.4 SIMTCore (simt_core.py)

```python
class SIMTCore:
    scheduler: WarpScheduler
    memory: Memory          # 全局共享内存
    warp_size: int
    num_warps: int

    def load_program(program: list[int])
    def step() -> bool      # 执行一个 warp 的一条指令
    def run()
```

## 4. 执行流程

```
SIMTCore::step():
  1. scheduler.select_warp() → warp
  2. Fetch & Decode instruction at warp.pc
  3. For each active thread in warp:
       Execute instruction using thread's own reg_file
  4. warp.pc += 1
  5. If HALT: remove warp from scheduling
```

## 5. 指令集扩展

Phase 2 新增（在 Phase 0/1 基础上）：

| 指令 | 编码 | 功能 |
|------|------|------|
| TID | 0x21 | rd = thread_id |
| WID | 0x22 | rd = warp_id |
| BAR | 0x23 | 同步屏障 |

执行时 `TID`/`WID` 对每个 active 线程分别写入其 thread_id/warp_id。

## 6. 内存访问模型

Phase 2 的内存模型：
- 所有 warp 共享同一个全局 Memory
- 访存按线程顺序串行执行（简化设计）
- 每个线程访问不同地址（通过 TID 计算偏移）

## 7. 典型程序：多线程向量加法

```asm
; 每个线程处理一对数据
; Thread i: C[i] = A[i] + B[i]

TID r1          ; r1 = thread_id (0..7)
MOV r2, 8
MUL r3, r1, r2  ; r3 = thread_id * 8 (字节偏移？不，字偏移)

; 加载 A[tid] 和 B[tid]
LD r4, [r3]     ; 用偏移地址（注意: LD 当前只支持立即数地址！）
; ...
```

注意：当前 LD/ST 只支持立即数地址。需要为 Phase 2 增加 LD/ST 的寄存器偏移模式，或使用变通方法（用 TID 动态计算地址然后 ST 到固定地址再 LD）。

最简单的方案：增加 `LDR rd, rs1, rs2` 和 `STR rs1, rs2, rs3`（base+offset 寻址）。

或者更简单：让每个线程的 LD/ST 使用 `base_addr + thread_id` 作为实际地址。即在 SIMTCore 层将立即数地址解释为基地址，实际地址 = imm + thread_id。

我选后者——更符合 GPGPU-Sim 的 GPU 编程模型（每个线程看到的是全局地址空间的不同部分）。直接在 `_execute` 中处理：对于 warp 内指令，LD 地址 = imm + thread_id。
