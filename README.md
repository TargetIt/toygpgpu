# toygpgpu

> 用 Python 实现一个教学用 GPGPU 模拟器，学习 [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) 的架构设计

## 项目目标

通过"自己写一遍"来理解 GPU 的 SIMT 执行模型、warp 调度、SIMT 栈、记分板、内存层次等核心概念。

模块划分与 GPGPU-Sim 保持一致，方便对照学习 C++ 原始实现。

## 渐进式实现路线

```
Phase 0: 标量处理器  → Phase 1: SIMD 向量  → Phase 2: Warp
                          ↓
Phase 5: 内存层次    ← Phase 4: 记分板     ← Phase 3: SIMT 栈
                          ↓
                     Phase 6: 多 Warp 调度
```

## 文档

- [需求分析](requirements/requirements.md)
- [调研报告](research/research.md)

## 参考项目

| 项目 | 语言 | 用途 |
|------|------|------|
| GPGPU-Sim | C++ | 主要参考，学习架构 |
| TinyGPU | Python | Python GPU 模拟器参考 |
| SoftGPU | Python | SIMT 概念参考 |
| GPU-Puzzles | Python | CUDA 思维训练 |
