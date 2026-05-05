# toygpgpu 调研报告

> 调研 GPGPU-Sim 架构及类似 Python 教学项目，为 toygpgpu 设计提供参考

---

## 一、主要参考：GPGPU-Sim

### 1.1 项目概况

| 项目 | 详情 |
|------|------|
| 名称 | GPGPU-Sim |
| 仓库 | https://github.com/gpgpu-sim/gpgpu-sim_distribution |
| 语言 | C++ |
| 规模 | ~10万行 |
| Stars | ~1,598 |
| 最新更新 | 2025-02 |
| 许可 | BSD |

### 1.2 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                       gpgpu_sim (顶层)                        │
│                                                              │
│   ┌───────────────────┐  ┌───────────────────┐               │
│   │ simt_core_cluster │  │ simt_core_cluster │  ...          │
│   │   ┌─────────────┐ │  │                   │               │
│   │   │shader_core  │ │  │                   │               │
│   │   │  ctx (SM)   │ │  │                   │               │
│   │   └─────────────┘ │  │                   │               │
│   └───────────────────┘  └───────────────────┘               │
│              │                     │                          │
│     ┌────────┴─────────────────────┴──────────┐              │
│     │        Interconnection Network           │              │
│     │           (intersim2/BookSim)            │              │
│     └────────┬─────────────────────┬──────────┘              │
│   ┌──────────┴───────┐   ┌─────────┴───────────┐             │
│   │ memory_partition │   │ memory_partition    │  ...        │
│   │   ┌───────────┐  │   │                     │             │
│   │   │ L2 Cache  │  │   │                     │             │
│   │   │ DRAM I/F  │  │   │                     │             │
│   │   └───────────┘  │   │                     │             │
│   └──────────────────┘   └─────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 关键模块详解

#### (1) shader_core_ctx（SIMT 核心 / SM）

这是 GPGPU-Sim 最核心的类，对应 NVIDIA SM。包含 6 级流水线：

```
FETCH → DECODE → ISSUE → READ_OPERAND → EXECUTE → WRITEBACK
```

**子组件**：

| 子组件 | 类名 | 功能 |
|--------|------|------|
| Warp 状态 | `shd_warp_t` | 模拟核心中每个 warp |
| SIMT 栈 | `simt_stack` | 分支发散与重汇聚（IPDOM） |
| 调度器 | `scheduler_unit` | 从 warp 中选择指令发射 |
| 记分板 | `scoreboard` | WAW/RAW 冒险检测 |
| 操作数收集器 | `opndcoll_rfu_t` | 从寄存器堆收集操作数 |
| 执行单元 | `simd_function_unit` | SP/SFU/Tensor 等管道 |

**warp 调度策略**：
- LRR（Loose Round Robin）
- GTO（Greedy Then Oldest）— 最常用
- Two-Level Active
- RRR（Strict Round Robin）

#### (2) simt_core_cluster（核心集群）

聚合多个 `shader_core_ctx` 为一个集群。共享：
- 到互连网络的注入端口
- 响应处理 FIFO

对应 NVIDIA TPC（Texture Processing Cluster）。

#### (3) memory_partition_unit（内存分区）

对应 NVIDIA 的 Memory Partition。包含：
- L2 Cache（多 bank）
- DRAM 控制器
- FR-FCFS 调度器
- 地址解码器（addr → channel/bank/row/col）

#### (4) 前端（cuda-sim/）

- PTX 解析器（Lex/Yacc）
- 指令功能模拟（~100 种 PTX 指令）
- 内存空间模拟（global/shared/constant/local/texture）

#### (5) 四大时钟域

| 时钟域 | 频率可配 | 覆盖范围 |
|--------|---------|---------|
| Core | 可配 | SIMT 核心集群 |
| ICNT | 可配 | 互连网络 |
| L2 | 可配 | 内存分区（除 DRAM）|
| DRAM | 可配 | GDDR DRAM |

→ 对我们 Python 版本的启示：**先只用一个时钟域**，简化设计。

---

## 二、类似 Python 项目调研

### 2.1 TinyGPU（deaneeth）

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/deaneeth/tinygpu |
| 语言 | Python 100% |
| 规模 | ~500 行 |
| Stars | ~44 |

**架构**：
- 自定义 `.tgpu` assembly 格式
- 12 条指令（SET, ADD, MUL, LD, ST, JMP, BNE, BEQ, SYNC, CSWAP, SHLD, SHST）
- Per-thread 寄存器（16 个）+ PC
- 全局共享内存
- 分支发散处理
- 可视化（热力图 + GIF 导出）

**优点**：纯 Python、小巧、可视化好、可直接运行
**缺点**：无 warp 概念、无流水线、无 SIMT 栈

### 2.2 SoftGPU（mhmdsabry）

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/mhmdsabry/SoftGPU |
| 语言 | Python |
| 规模 | ~300 行 |

**架构**：
- GPU 类（num_sms, threads_per_warp, warps_per_sm）
- Warp 调度（跨 SM 分布）
- 全局内存 + 共享内存
- Kernel launch（grid_dim, block_dim）
- Memory profiling

**优点**：有 grid/block/warp 概念、kernel launch 接口接近 CUDA
**缺点**：功能简单、无 SIMT 栈

### 2.3 GPU-Puzzles（srush）

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/srush/GPU-Puzzles |
| 语言 | Python（通过 numba.cuda）|

**架构**：11 个交互式 puzzle，从简单 map 到矩阵乘
**优点**：极好的教学材料，渐进式学习
**缺点**：依赖真实 GPU 或 CUDA 模拟器，不是独立的模拟器

### 2.4 Numba CUDA Simulator

| 项目 | 详情 |
|------|------|
| 位置 | numba.cuda 内置 |
| 使用 | `NUMBA_ENABLE_CUDASIM=1` |

**功能**：模拟 CUDA 线程模型（threadIdx, blockIdx, blockDim）、共享内存、同步、原子操作
**优点**：CUDA 兼容、无需 GPU、可调试
**缺点**：不是独立项目、黑盒模拟

---

## 三、对比分析

| 维度 | GPGPU-Sim | TinyGPU | SoftGPU | toygpgpu（目标） |
|------|-----------|---------|---------|-----------------|
| 语言 | C++ | Python | Python | **Python** |
| 规模 | 10万行 | 500行 | 300行 | **2000-5000行** |
| 精度 | 周期精确 | 功能级 | 功能级 | **功能级（先）** |
| Warp 概念 | ✅ | ❌ | ✅ | **✅** |
| SIMT 栈 | ✅ | ❌ | ❌ | **✅** |
| 流水线 | 6级 | 1级 | 1级 | **3级（先简后全）** |
| Scoreboard | ✅ | ❌ | ❌ | **✅** |
| 内存层次 | L1/L2/DRAM | Global | Global+Shared | **RF+Shared+L1+L2** |
| ISA | PTX | 自定义 | Python | **自定义** |
| 可视化 | AerialVision | 热力图 | Profiling | **ASCII trace** |
| 教学友好 | ❌（太复杂）| ✅ | ✅ | **✅（目标）** |

---

## 四、toygpgpu 设计建议

### 4.1 模块映射（GPGPU-Sim → Python）

| GPGPU-Sim (C++) | toygpgpu (Python) | 职责 |
|-----------------|-------------------|------|
| `gpgpu_sim` | `gpu_sim.py` | 顶层协调 |
| `shader_core_ctx` | `simt_core.py` | SIMT 核心流水线 |
| `shd_warp_t` | `warp.py` | warp 状态管理 |
| `simt_stack` | `simt_stack.py` | 分支发散/重汇聚 |
| `scheduler_unit` | `scheduler.py` | warp 调度 |
| `scoreboard` | `scoreboard.py` | 寄存器冒险 |
| `opndcoll_rfu_t` | `operand_collector.py` | 操作数收集 |
| `simd_function_unit` | `func_unit.py` | 功能单元（SP/SFU） |
| `ldst_unit` | `ldst_unit.py` | 访存单元 |
| `memory_partition_unit` | `memory.py` | 内存子系统 |
| `icnt_wrapper` | `interconnect.py` | 互连网络 |
| `instructions.cc` | `isa/executor.py` | 指令执行 |
| `ptx.l/ptx.y` | `isa/decoder.py` | 指令译码 |

### 4.2 渐进式实现路线

```
Phase 0 (标量) → Phase 1 (SIMD) → Phase 2 (Warp)
    ↓
Phase 5 (内存) ← Phase 4 (记分板) ← Phase 3 (SIMT栈)
    ↓
Phase 6 (多Warp调度) → Phase 7 (完整内核启动)
```

### 4.3 关键设计原则

1. **功能优先于时序**：先让程序跑对，再加流水线细节
2. **模块与 GPGPU-Sim 对齐**：类名、职责划分保持一致，方便对照学习
3. **可观测性**：每一步状态变化可打印、可追踪
4. **渐进复杂度**：每个 Phase 增加一个概念，不贪多

---

## 五、参考资料索引

| 资源 | 链接 |
|------|------|
| GPGPU-Sim 仓库 | https://github.com/gpgpu-sim/gpgpu-sim_distribution |
| GPGPU-Sim 手册 | https://pages.cs.wisc.edu/~chen-han/doc/GPGPU-Sim_Manual.html |
| GPGPU-Sim Codebase Org | https://deepwiki.com/gpgpu-sim/gpgpu-sim_distribution/1.3-codebase-organization |
| GPGPU-Sim Core Components | https://deepwiki.com/gpgpu-sim/gpgpu-sim_distribution/2-core-simulation-components |
| TinyGPU | https://github.com/deaneeth/tinygpu |
| SoftGPU | https://github.com/mhmdsabry/SoftGPU |
| GPU-Puzzles | https://github.com/srush/GPU-Puzzles |

---

*文档版本: v0.1 | 创建日期: 2026-05-05*
