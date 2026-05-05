# Phase 3 实验报告

## 1. 实验概述

**目标**: 实现 SIMT Stack，正确处理 GPU warp 内线程的分支发散（Branch Divergence）与重汇聚（Reconvergence）。

**对标**: GPGPU-Sim 的 `simt_stack` 类（基于 IPDOM 的后支配栈）。

## 2. 新增模块

| 文件 | 行数 | 说明 |
|------|------|------|
| simt_stack.py | 65 | SIMTStack 类 (对标 simt_stack) |
| simt_core.py | 260 | 分支执行 + 重汇聚逻辑 (重写) |
| isa.py | +6 | JMP/BEQ/BNE 操作码 |
| assembler.py | 130 | Two-pass: label 解析 + 分支指令 |

## 3. 关键设计决策

### 3.1 SIMT Stack 结构

```python
SIMTStackEntry:
  reconv_pc       # 重汇聚 PC
  orig_mask       # 发散前的 active_mask (始终保持 full mask)
  taken_mask      # 已执行过的线程路径 (累积)
  fallthrough_pc  # 非跳转路径的起始 PC
```

### 3.2 发散处理流程

1. 条件分支时，计算 taken_mask（满足条件的线程）和 not_taken_mask
2. 如果两者都非零 → **发散**：Push entry，设 active_mask=taken
3. 路径内 JMP 到汇合标签 → 更新 reconv_pc
4. 到达重汇聚点 → Pop entry，切换到剩余路径
5. remaining mask 为空 → 恢复 full mask

### 3.3 累积 taken_mask

关键设计：second path push 时 `taken_mask` 累积为 `entry.taken_mask | remaining_mask`，确保最终正确检测"所有路径已执行"。

### 3.4 Two-Pass 汇编器

Pass 1: 扫描 label，建立 label→PC 映射
Pass 2: 使用相对偏移生成分支指令

## 4. 测试结果

32/32 全部通过：
- SIMT Stack: 6/6 ✅
- ISA: 6/6 ✅
- Assembler: 6/6 ✅
- 集成测试: 14/14 ✅
  - JMP 跳过正确
  - BEQ 条件分支正确
  - **偶数/奇数线程 if/else 发散 + 重汇聚** ✅
  - tid 高低分组发散 ✅
  - 向后兼容

## 5. 已知限制

1. reconv_pc 依赖 JMP merge 更新（无自动 IPDOM 检测）
2. 不支持嵌套 SIMT Stack（代码结构支持，但需更多测试）
3. 跨 warp 发散未覆盖
