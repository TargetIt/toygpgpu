# Phase 5: Memory Subsystem — Design Document / 设计文档

> **对应 GPGPU-Sim**: L1 Cache (`gpgpu-sim/gpu-cache.cc`), Shared Memory (`shader_core_ctx`), Memory Coalescing (`memory_partition_unit`)
> **参考**: GPGPU-Sim L1 cache 实现, CUDA 共享内存/合并访问模型

## 1. Introduction / 架构概览

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                         SIMTCore (Phase 5)                          │
  │                                                                      │
  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐        │
  │  │    Warp 0     │    │    Warp 1     │    │    Warp N     │        │
  │  │ ┌─────┐ ┌───┐ │    │ ┌─────┐ ┌───┐ │    │ ┌─────┐ ┌───┐ │        │
  │  │ │Score│ │SIMT│ │    │ │Score│ │SIMT│ │    │ │Score│ │SIMT│ │        │
  │  │ │board│ │Stk │ │    │ │board│ │Stk │ │    │ │board│ │Stk │ │        │
  │  │ └─────┘ └───┘ │    │ └─────┘ └───┘ │    │ └─────┘ └───┘ │        │
  │  └───────────────┘    └───────────────┘    └───────────────┘        │
  │          \                  |                 /                      │
  │           ┌─────────────────┴─────────────────┐                      │
  │           │       WarpScheduler (RR)          │                      │
  │           └─────────────────┬─────────────────┘                      │
  │                             v                                        │
  │  ┌────────────────────────────────────────────────┐                  │
  │  │  ThreadBlock (CTA)                              │                 │
  │  │  ┌─────────────────────────────────────────┐   │                  │
  │  │  │  SharedMemory (256 words, per-CTA)      │   │                  │
  │  │  │  SHLD / SHST per-thread access           │   │                  │
  │  │  └─────────────────────────────────────────┘   │                  │
  │  └────────────────────────────────────────────────┘                  │
  │                                                                      │
  │  ┌────────────────────────────────────────────────┐                  │
  │  │  L1 Data Cache (Direct-mapped)                 │                  │
  │  │  16 lines x 4 words/line = 256 bytes           │                  │
  │  │  Write-through, No-write-allocate               │                  │
  │  │  hit/miss statistics                            │                  │
  │  └───────────────────────┬────────────────────────┘                  │
  │                          v                                           │
  │  ┌────────────────────────────────────────────────┐                  │
  │  │  Global Memory (DRAM, 1024 words)               │                  │
  │  └────────────────────────────────────────────────┘                  │
  │                                                                      │
  │  Coalescing: contiguous warp addresses → 1 transaction               │
  └──────────────────────────────────────────────────────────────────────┘
```

Phase 5 为 toygpgpu 引入完整的内存层次结构：L1 数据缓存、共享内存（per-block）和合并访问检测。这是从"平面内存"模型到层次化内存系统的关键演进。

## 2. Motivation / 设计动机

Phase 4 使用平面内存模型——所有 LD/ST 指令直接访问全局内存，延迟固定为 4 周期。真实的 GPU 内存系统是高度层次化的：

- **L1 Cache**：每个 SM 的片上缓存，减少全局内存访问延迟。大多数 GPU (NVIDIA Fermi+) 有统一的 L1/共享内存，或独立的 L1 数据缓存。
- **Shared Memory**：每个 Thread Block 的低延迟片上 SRAM，由程序员显式管理（`__shared__`），比全局内存快 20-30 倍。
- **Memory Coalescing**：当 warp 内所有线程访问连续地址时，硬件可将多个内存请求合并为一次宽事务，极大提升带宽利用率。

GPGPU-Sim 实现模型：
- `gpu-cache` 模块实现可配置的缓存层次（L1/L2）
- `shader_core_ctx` 管理 shared memory 和 coalescing 逻辑
- `memory_partition_unit` 处理 DRAM 访问和带宽建模

Phase 5 对标这些模块的简化版本。

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 L1 Cache (Direct-Mapped)

直接映射缓存：每个地址映射到唯一缓存行。

```
参数: num_lines=16, line_size=4 words

tag   = addr // (num_lines * line_size)
index = (addr // line_size) % num_lines
offset = addr % line_size

read(addr):
    idx, tag, offset = compute(addr)
    if lines[idx].valid AND lines[idx].tag == tag:
        hits++; return lines[idx].data[offset]    # HIT
    else:
        misses++; return None                     # MISS

write(addr, value):  # Write-through, no-write-allocate
    if lines[idx].valid AND lines[idx].tag == tag:
        hits++; lines[idx].data[offset] = value   # HIT: update cache
    else:
        misses++                                   # MISS: pass through

fill_line(addr, data):
    idx = index(addr)
    lines[idx] = {valid=True, tag=tag, data=data}
```

### 3.2 Memory Coalescing

```
_is_contiguous(addrs) → bool:
    if len(addrs) <= 1: return True
    for i in range(1, len(addrs)):
        if addrs[i] != addrs[i-1] + 1: return False
    return True

_mem_load(warp, instr):
    addrs = sorted((base + t.tid) for active threads)
    if _is_contiguous(addrs):
        coalesce_count++       # 合并为 1 次事务
        # 预取完整缓存行
        start = addrs[0] & ~3  # 对齐到 4-word 边界
        fill cache line from memory
        read from cache for each thread
    else:
        # 每个线程独立加载（多次事务）
        for each thread: read from cache or fallback to memory
```

### 3.3 Shared Memory

```
SharedMemory (per ThreadBlock):
    data: bytearray (size_words * 4 bytes)
    
    read_word(addr):
        addr %= size_words
        return little_endian_32bit(data[addr*4:addr*4+4])
    
    write_word(addr, value):
        addr %= size_words
        data[addr*4:addr*4+4] = little_endian_32bit(value)

指令集扩展:
    SHLD rd, [imm]  → rd = shared_mem[(imm + tid) % size]
    SHST rs1, [imm] → shared_mem[(imm + tid) % size] = rs1
    opcode: SHLD=0x31, SHST=0x32
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| 文件 | 作用 |
|------|------|
| `cache.py` | L1Cache 类：直接映射缓存，hit/miss 统计 |
| `shared_memory.py` | SharedMemory 类：per-CTA 片上 SRAM |
| `thread_block.py` | ThreadBlock 类：CTA 容器，持有 warps + shared_memory |
| `simt_core.py` | SIMTCore 集成 L1Cache, ThreadBlock, 合并检测逻辑 |
| `isa.py` | 新增 OP_SHLD (0x31), OP_SHST (0x32) |
| `memory.py` | 全局内存 (无变更) |
| `scoreboard.py` | 记分板 (无变更) |
| `learning_console.py` | 交互式控制台，新增 `cache`/`smem` 命令 |

### 4.2 Key Data Structures / 关键数据结构

```python
@dataclass
class CacheLine:
    valid: bool = False
    tag: int = 0               # 块地址
    dirty: bool = False
    data: list = [0]*4         # 4 个 32-bit word

class L1Cache:
    num_lines: int = 16
    line_size: int = 4
    lines: list[CacheLine]
    hits: int
    misses: int
    
    def read(addr) -> Optional[int]   # 读; miss 返回 None
    def write(addr, value)            # 写-through
    def fill_line(addr, data)         # 填充缓存行
    @property hit_rate() -> float

class SharedMemory:
    data: bytearray           # 字节级存储
    size_words: int = 256     # 256 words × 32-bit
    def read_word(addr) -> int
    def write_word(addr, value)

class ThreadBlock:
    block_id: int
    warps: list
    shared_memory: SharedMemory
```

### 4.3 ISA Encoding / 指令编码

| Opcode | Mnemonic | 格式 | 描述 |
|--------|----------|------|------|
| 0x31 | SHLD | `rd = shared_mem[imm+tid]` | 共享内存加载 |
| 0x32 | SHST | `shared_mem[imm+tid] = rs1` | 共享内存存储 |

译码字段: `[31:24] opcode | [23:20] rd | [19:16] rs1 | [15:12] rs2 | [11:0] imm`

### 4.4 Module Interfaces / 模块接口

```
SIMTCore 构造函数:
    self.l1_cache = L1Cache()            # 新增 L1 缓存
    self.thread_block = ThreadBlock(...)  # 新增 ThreadBlock
    self.coalesce_count = 0              # 合并事务计数
    self.total_mem_reqs = 0              # 总内存请求计数

_mem_load:  读 L1 → miss 则读 DRAM → fill cache line
_mem_store: 写 L1 (write-through) + 写 DRAM (no-write-allocate)
_is_contiguous: 检查地址是否连续 → 合并统计
```

## 5. Functional Processing Flow / 功能处理流程

### 示例 1: 共享内存操作 (`01_shared_mem.asm`)

```
程序: TID r1; MOV r2,10; MUL r3,r1,r2 → r3=tid*10
      SHST r3,[0]      → smem[tid] = tid*10
      SHLD r4,[0]      → r4 = smem[tid]
      ST r4,[100]       → mem[100+tid] = tid*10
      HALT

warp_size=4, 单 warp:

周期  PC  指令               操作                         寄存器变化
──────────────────────────────────────────────────────────────────────
0    0   TID r1             T0:r1=0 T1:r1=1 T2:r1=2 T3:r1=3
1    1   MOV r2,10          r2=10 (广播到所有线程)
2    2   MUL r3,r1,r2       T0:r3=0 T1:r3=10 T2:r3=20 T3:r3=30
3    3   SHST r3,[0]        smem[0]=0 smem[1]=10 smem[2]=20 smem[3]=30
4    4   SHLD r4,[0]        T0:r4=0 T1:r4=10 T2:r4=20 T3:r4=30
5    5   ST r4,[100]        mem[100]=0 mem[101]=10 mem[102]=20 mem[103]=30
6    6   HALT
```

### 示例 2: 合并访问 (`02_coalescing.asm`)

```
ST r3,[50] → addrs = [50,51,52,53]  → _is_contiguous → True (coalesced)
              coalesce_count=1, total_mem_reqs=1

LD r4,[50] → addrs = [50,51,52,53]  → _is_contiguous → True (coalesced)
              cache: miss addr 50 → fill_line([50..53])
              cache: hit 51,52,53 → 无额外 DRAM 访问
              coalesce_count=2, total_mem_reqs=2
```

## 6. Comparison with Phase 4 / 与前一版本的对比

| Aspect | Phase 4 (Scoreboard) | Phase 5 (Memory) | Change |
|--------|----------------------|-------------------|--------|
| 内存模型 | 平面全局内存 | L1 Cache + 全局内存 | 层次化 |
| 缓存 | 无 | 直接映射 L1 (16行×4字) | 新增 L1Cache |
| 共享内存 | 无 | Per-block SharedMemory (256 words) | 新增 SHLD/SHST |
| Thread Block | 无 | ThreadBlock 容器 | 新增概念 |
| 访存合并 | 每个线程独立访存 | 合并连续地址为单事务 | 合并检测 |
| 内存统计 | 无 | hit/miss + coalescing 率 | 性能分析 |
| ISA 指令 | 14 条 | + SHLD/SHST = 16 条 | 扩展 |
| LD/ST | 直接 mem[imm+tid] | L1 → miss → mem 三级 | 多级访问 |
| 新增文件 | — | `cache.py`, `shared_memory.py`, `thread_block.py` | 3 个新文件 |

## 7. Known Issues and Future Work / 遗留问题与后续工作

- **直接映射简单**：直接映射比组相联 (set-associative) 缓存冲突率高。真实 GPU L1 通常为 4-8 路组相联。
- **Write-through 简化**：write-through 性能低于 write-back。GPGPU-Sim 支持 write-back + write-allocate。
- **Coalescing 简化**：仅检测完全连续地址。真实 GPU 支持更复杂的合并模式（如 stride、对齐检测）。
- **共享内存无 bank**：未模拟 shared memory bank conflict（真实 GPU shared memory 分为 32 banks）。
- **Fixed latency**：缓存 hit 和 miss 都用 4 周期延迟。真实 GPU 的缓存命中延迟约为 1-2 周期。
- **无 L2 Cache**：仅有 L1，缺少 GPU 内存层次中的 L2 缓存和内存控制器。

