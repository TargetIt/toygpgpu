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

## 11 个 Phase 总览

| Phase | 名称 | 测试 | 对标 GPGPU-Sim |
|-------|------|------|----------------|
| 0 | [标量处理器](phase0_scalar/) | 45 | simd_function_unit (SP) |
| 1 | [SIMD 向量](phase1_simd/) | 49 | simd_function_unit (多 lane) |
| 2 | [SIMT Warp/Thread](phase2_simt/) | 43 | shd_warp_t, scheduler_unit |
| 3 | [SIMT Stack 分支发散](phase3_simt_stack/) | 32 | simt_stack |
| 4 | [Scoreboard 冒险](phase4_scoreboard/) | 15 | scoreboard |
| 5 | [内存层次](phase5_memory/) | 17 | gpu-cache, shared mem |
| 6 | [Kernel Launch + GTO](phase6_kernel/) | 12 | gpgpu_sim 顶层 |
| 7 | [流水线解耦](phase7_pipeline/) | 25 | fetch/decode/issue, opndcoll |
| 8 | [PTX Frontend](phase8_ptx/) | 16 | ptx_parser |
| 9 | [Tensor Core MMA](phase9_tensor/) | 7 | HMMA instruction |
| 10 | [可视化工具链](phase10_viz/) | 9 | AerialVision, stat-tool |
| 11 | [学习控制台](phase11_console/) | 25 | GDB-style debugger |

**总计: 251 项测试，~5500 行 Python，覆盖 GPGPU-Sim 15 个核心模块**

## 快速开始

```bash
# 运行任意 Phase 的测试
cd phase0_scalar && bash run.sh

# 启动交互式学习控制台
cd phase11_console
python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4
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
