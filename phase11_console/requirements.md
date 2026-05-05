# Phase 11: 学习控制台需求

## 1. 目标
为初学者提供逐周期、逐流水级的 GPU 内部状态观察工具。

## 2. 功能
- FR-01: 交互式单步执行 (Enter 键)
- FR-02: 五级流水线可视化
- FR-03: 寄存器变化追踪
- FR-04: Scoreboard/I-Buffer/SIMT Stack 实时显示
- FR-05: 断点支持
- FR-06: 自动运行模式

## 3. 验收标准
- AC-01: 单步模式每周期输出完整状态
- AC-02: 流水线、Scoreboard、I-Buffer、Stack 同时可见
- AC-03: 断点命中时暂停
- AC-04: 分支发散程序的 SIMT Stack push/pop 可见
