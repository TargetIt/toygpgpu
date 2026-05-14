# toygpgpu

> 用 Python 实现一个教学用 GPGPU 模拟器，学习 [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) 的架构设计

## 项目目标

通过"自己写一遍"来理解 GPU 的 SIMT 执行模型、warp 调度、SIMT 栈、记分板、内存层次、Tensor Core 等核心概念。

模块划分与 GPGPU-Sim 保持一致，方便对照学习 C++ 原始实现。

## 渐进式实现路线

```
Phase 0: 标量处理器  → Phase 1: SIMD 向量  → Phase 2: Warp
                                                    ↓
Phase 5: 内存层次    ← Phase 4: Scoreboard ← Phase 3: SIMT Stack
       ↓
Phase 6: Kernel Launch → Phase 7: 流水线解耦 → Phase 8: PTX Frontend
       ↓
Phase 9: Tensor Core  → Phase 10: 可视化    → Phase 11: 学习控制台
```

## 12 个 Phase 总览

| Phase | 名称 | 测试 | 对标 GPGPU-Sim | 新增功能 |
|-------|------|------|----------------|----------|
| 0 | [标量处理器](phase0_scalar/) | 45 | simd_function_unit (SP) | learning_console.py, trace |
| 1 | [SIMD 向量](phase1_simd/) | 49 | simd_function_unit (多 lane) | learning_console, vec4/float4 |
| 2 | [SIMT Warp/Thread](phase2_simt/) | 43 | shd_warp_t, scheduler_unit | learning_console, warp-regs |
| 3 | [SIMT Stack 分支发散](phase3_simt_stack/) | 32 | simt_stack | learning_console, PRED, warp-regs |
| 4 | [Scoreboard 冒险](phase4_scoreboard/) | 15 | scoreboard | learning_console, PRED, vec4, warp-regs |
| 5 | [内存层次](phase5_memory/) | 17 | gpu-cache, shared mem | learning_console, PRED, vec4, warp-regs |
| 6 | [Kernel Launch + GTO](phase6_kernel/) | 12 | gpgpu_sim 顶层 | learning_console, PRED, vec4, warp-regs |
| 7 | [流水线解耦](phase7_pipeline/) | 25 | fetch/decode/issue, opndcoll | learning_console, PRED, vec4, warp-regs |
| 8 | [PTX Frontend](phase8_ptx/) | 16 | ptx_parser | learning_console, PRED, vec4, warp-regs |
| 9 | [Tensor Core MMA](phase9_tensor/) | 7 | HMMA instruction | learning_console, PRED, vec4, warp-regs |
| 10 | [可视化工具链](phase10_viz/) | 9 | AerialVision, stat-tool | TraceCollector, 可视化, JSON Export |
| 11 | [学习控制台](phase11_console/) | 25 | GDB-style debugger | 断点, ANSI 显示, 五级流水线可视化 |

**总计: 251 项测试，~5500 行 Python，覆盖 GPGPU-Sim 15 个核心模块**

## 新增功能概览

### learning_console.py 交互式调试

每个 Phase 均附带了 `learning_console.py` 交互式学习控制台，提供 GDB 风格的 GPU 流水线单步调试能力：

- **统一接口**: `python src/learning_console.py <program.asm> [options]`
- **逐周期单步**: 每步执行一个周期，显示完整的内部状态变化
- **状态查看命令**: `reg` 查看寄存器、`m` 查看内存、`i` 查看完整状态
- **自动运行**: `r` 运行到 HALT，`r N` 运行 N 个周期
- **Trace 模式**: `--trace` 或 `--auto` 参数用于非交互式批处理执行
- **Per-Phase 增强**: 每个 Phase 的控制台会展示该阶段特有的状态（SIMT Stack、Scoreboard、I-Buffer 等）

各 Phase 的 Quick Start 可在对应目录的 README.md 中找到。

### PRED 谓词执行

**SETP** 指令和 **@p0** 谓词标志位支持条件执行，无需改变控制流：

- `SETP rd, rs1, rs2` — 比较 rs1 和 rs2，根据比较模式(EQ/NE)设置 rd 的谓词位
- `@p0` 指令前缀 — 指令字 bit 31 作为谓词标志，谓词为真时指令才执行
- 与 SIMT Stack 互补：谓词执行适用于简单条件，避免分支发散的开销
- Perc-thread 谓词寄存器：每个 `Thread` 对象包含 `pred` 属性
- 汇编器透明支持：`@p0 ADD r1, r2, r3` 语法

### vec4/float4 复合数据类型

对标 GPU shader 中的 `float4`/`uchar4` 类型，在标量 32-bit 寄存器内部实现 4×8-bit SIMD (SWAR)：

- `V4PACK rd, rs1, rs2` — 从两个源寄存器的低 16 位打包 4 个 8-bit 值
- `V4ADD rd, rs1, rs2` — 4×8-bit 独立加法（每个 byte lane 独立，结果截断到 8-bit）
- `V4MUL rd, rs1, rs2` — 4×8-bit 独立乘法
- `V4UNPACK rd, rs1, lane` — 提取指定 byte lane (0-3)，零扩展到 32-bit
- `Vec4ALU` 模块：独立的 ALU 实现，支持 pack/add/mul/unpack 操作
- 编写紧凑的像素/颜色处理代码，无需向量寄存器

### Warp-Level 寄存器

Warp 级别的统一寄存器，warp 内所有线程共享，用于线程束级别的元数据访问：

- `WREAD rd, imm` — 读取 warp 寄存器到标量寄存器
- `WWRITE imm, rs1` — 写入标量寄存器到 warp 寄存器
- 内置 warp 寄存器：`wid` (warp_id, idx=0), `ntid` (线程数, idx=1)
- 所有线程可见：warp 内任意线程均可访问统一值
- `wreg` 命令：learning_console.py 中查看 warp 寄存器状态
- 对标 GPGPU-Sim 中 warp 级别的 uniform 寄存器

### Trace/Debug 追踪功能

完整的执行追踪与可视化工具链（Phase 10-11）：

- **TraceCollector**: 逐周期事件收集器，记录每条指令的 cycle/warp/PC/opcode/active_mask/stall_reason
- **ASCII Warp Timeline**: 每个 warp 的 PC 随时间变化的时间线图
- **Stall Analysis**: Stall 原因分布柱状图（scoreboard/barrier/ibuffer）
- **Memory Heatmap**: 内存访问密度字符热力图
- **JSON Export**: Chrome Tracing 兼容格式，可用 `chrome://tracing` 加载
- **`full_report()`**: 一键生成完整分析报告
- 所有 Phase 的 learning_console.py 支持 `--trace` 标志进行批处理追踪输出

## 快速开始

```bash
# 运行任意 Phase 的测试
cd phase0_scalar && bash run.sh

# 启动交互式学习控制台（Phase 11 完整版）
cd phase11_console
python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

# Trace 模式（非交互式批处理追踪）
python3 src/learning_console.py tests/programs/demo_basic.asm --trace

# 自动播放模式（每 0.5 秒一个周期）
python3 src/learning_console.py tests/programs/demo_divergence.asm --auto-interval 0.5

# 运行指定周期数
python3 src/learning_console.py tests/programs/demo_basic.asm --max-cycles 100

# PTX 程序（Phase 8）
cd phase8_ptx
python3 src/learning_console.py tests/programs/01_vector_add.ptx --auto
```

## 文档

- [需求分析](requirements/requirements.md)
- [调研报告](research/research.md)
- 每个 Phase 目录下均有 `requirements.md` + `design.md` + `README.md`

## 参考项目

| 项目 | 语言 | 用途 |
|------|------|------|
| [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) | C++ | 主要参考，学习架构 |
| [TinyGPU](https://github.com/deaneeth/tinygpu) | Python | Python GPU 模拟器参考 |
| [SoftGPU](https://github.com/mhmdsabry/SoftGPU) | Python | SIMT 概念参考 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | Python | CUDA 思维训练 |
