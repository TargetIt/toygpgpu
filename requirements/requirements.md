# toygpgpu 需求文档

> 用 Python 实现一个教学用 GPGPU 模拟器，学习 GPGPU-Sim 的架构设计

---

## 一、项目目标

**核心目标**：用 Python 从头实现一个 GPGPU 功能模拟器，通过"写一遍"来理解 GPGPU-Sim 的架构设计。

**学习目标**：
- 理解 GPU 的 SIMT 执行模型（Single Instruction, Multiple Threads）
- 理解 warp/warp-divergence/scoreboard/operand-collector 等核心概念
- 理解 GPU 内存层次（register file, shared memory, L1/L2, DRAM）
- 理解 GPGPU-Sim 的模块划分设计哲学

**非目标**（不在初期范围内）：
- 不做周期精确（cycle-accurate）时序建模——先做功能级（functional）
- 不接真实 CUDA/OpenCL——先用自定义 assembly-like ISA
- 不追求性能——Python 原生实现，追求可读性

---

## 二、学习路径规划（分阶段）

### Phase 0: 单线程标量处理器
从最简单开始——一个支持加/减/乘/除/访存的标量处理器。

**指令**：`ADD, SUB, MUL, DIV, LD, ST, MOV, HALT`
**组件**：PC, Register File (16×32bit), ALU, Memory (256 words)

### Phase 1: SIMD 向量处理器
将标量处理器扩展为 SIMD——一条指令同时操作多个数据通道。

**指令**：`VADD, VSUB, VMUL, VLD, VST`（向量版本）
**组件**：Vector Register File (8 lanes × 16 regs), Vector ALU, Vector Memory

### Phase 2: SIMT 核心——引入 Warp
引入 warp 概念——一个 warp 包含 N 个线程，共享同一个 PC。

**新增概念**：
- Warp = 一组线程（默认 32 threads/warp）
- 每个线程有独立寄存器（register file per thread）
- 线程 ID（`thread_id`, `warp_id`）

### Phase 3: SIMT 栈——分支发散与重汇聚
处理 `if/else` 分支——不同线程走不同分支的情况。

**新增概念**：
- Active mask（哪些线程在当前路径上活跃）
- SIMT Stack（后支配栈，用于分支重汇聚）
- 发散（divergence）/ 重汇聚（reconvergence）

### Phase 4: 记分板与操作数收集器
增加流水线冒险检测。

**新增概念**：
- Scoreboard（跟踪寄存器写后读/写后写冒险）
- Operand Collector（从寄存器堆收集操作数）

### Phase 5: 内存层次
从单一内存扩展到多层内存。

**新增概念**：
- Shared Memory（同一 warp/block 内共享）
- L1/L2 Cache（简化版）
- Memory coalescing（合并访问）
- Barrier（`BAR` 指令，warp 间同步）

### Phase 6: 多 Warp 调度
多个 warp 并发执行。

**新增概念**：
- Warp Scheduler（Round-Robin / GTO）
- Warp 状态机（Active / Waiting / Blocked）
- 线程块（Thread Block / CTA）

---

## 三、模块划分（参考 GPGPU-Sim）

```
toygpgpu/
├── sim/
│   ├── gpu_sim.py              ← 顶层模拟器 (对应 gpgpu_sim)
│   ├── simt_core.py            ← SIMT 核心 (对应 shader_core_ctx)
│   ├── warp.py                 ← Warp 管理 (对应 shd_warp_t)
│   ├── simt_stack.py           ← SIMT 栈 (对应 simt_stack)
│   ├── scheduler.py            ← Warp 调度器 (对应 scheduler_unit)
│   ├── scoreboard.py           ← 记分板 (对应 scoreboard)
│   ├── operand_collector.py    ← 操作数收集器 (对应 opndcoll_rfu_t)
│   ├── func_unit.py            ← 功能单元 (对应 simd_function_unit)
│   ├── ldst_unit.py            ← 访存单元 (对应 ldst_unit)
│   ├── memory.py               ← 内存层次 (对应 memory_partition_unit)
│   └── interconnect.py         ← 互连网络 (对应 icnt_wrapper)
├── isa/
│   ├── opcodes.py              ← 指令集定义
│   ├── decoder.py              ← 指令译码
│   └── executor.py             ← 指令执行
├── config/
│   └── gpu_config.py           ← GPU 配置参数
├── tests/
│   └── ...                     ← 各阶段测试
└── requirements/               ← 本文档
```

---

## 四、第一阶段最小原型（Phase 0）

### 指令集

| 指令 | 格式 | 说明 |
|------|------|------|
| ADD | `ADD rd, rs1, rs2` | rd = rs1 + rs2 |
| SUB | `SUB rd, rs1, rs2` | rd = rs1 - rs2 |
| MUL | `MUL rd, rs1, rs2` | rd = rs1 * rs2 |
| DIV | `DIV rd, rs1, rs2` | rd = rs1 / rs2 |
| LD  | `LD rd, addr` | rd = mem[addr] |
| ST  | `ST rs, addr` | mem[addr] = rs |
| MOV | `MOV rd, imm` | rd = imm |
| HALT| `HALT` | 停止执行 |

### 硬件参数

| 参数 | 值 |
|------|-----|
| 寄存器数 | 16 × 32-bit |
| 内存大小 | 256 × 32-bit words |
| PC 位宽 | 16-bit |

### 程序示例

```asm
MOV r1, 5        # r1 = 5
MOV r2, 3        # r2 = 3
ADD r3, r1, r2   # r3 = 8
ST r3, 100       # mem[100] = 8
HALT
```

---

## 五、成功标准

1. Phase 0 程序能正确执行，输出与预期一致
2. 每个 Phase 有对应的测试用例
3. 代码模块划分与 GPGPU-Sim 概念一致
4. 每个模块有清晰的 docstring 和注释
5. 最终原型能运行一个简单的向量加法 kernel（多 warp 并行）

---

## 六、参考项目

| 项目 | 语言 | 规模 | 用途 |
|------|------|------|------|
| [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) | C++ | ~10万行 | 主要参考，学习架构 |
| [TinyGPU](https://github.com/deaneeth/tinygpu) | Python | ~500行 | Python GPU 模拟器参考 |
| [SoftGPU](https://github.com/mhmdsabry/SoftGPU) | Python | ~300行 | SIMT 概念参考 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | Python | ~200行 | CUDA 思维训练 |

---

*文档版本: v0.1 | 创建日期: 2026-05-05*
