# Phase 3 实验报告

## 2026-05-15: Feature Additions

### PRED vs DPC 分支比较
- 新增 predication 演示程序，对比两种分支处理策略在 GPU 上的表现：
  - **DPC（Diverse PC）**：使用 JMP/BEQ 分支指令 + SIMT Stack 管理，发散时线程走不同路径后重汇聚
  - **Predication（断言执行）**：不使用分支指令，通过 if-conversion 将条件判断转换为断言掩码，所有线程执行相同指令序列，不活跃线程的结果被屏蔽
- 对比结果：
  - DPC 优势：跳过不执行路径，减少总指令数；劣势：SIMT Stack 管理开销，发散严重时性能下降
  - Predication 优势：无分支开销，无 SIMT Stack 操作；劣势：所有线程始终执行全部指令，存在无效计算
  - 在分支高度发散（50% 线程走 then，50% 走 else）场景下 Predication 可能更快
  - 在分支稀疏（仅少数线程走 then）场景下 DPC 占优

### predication 演示程序
- 实现 if-conversion 模式的 demo：使用条件 MOV 指令（或掩码 ST）替代显式分支
- 验证 predication 下所有线程保持相同控制流，不存在发散/重汇聚
- trace 输出显示 predication 模式下的 active mask 始终保持全 1（所有线程活跃）
- 对比 DPC 模式 trace：发散时 active mask 分割为 taken_mask 和 not_taken_mask 交替执行

### learning_console.py 交互式调试体验
- Phase 3 调试器增强支持 SIMT Stack 状态查看：
  - `simt_stack`: 显示当前 warp 的 SIMT Stack 深度和每个 entry 的详细信息
  - `stack entry` 包含：reconv_pc（重汇聚点）、taken_mask（已执行线程掩码）、fallthrough_pc（非跳转路径 PC）
- 新增 predication 模式调试支持：
  - `pred_mode on/off`: 切换 predication 和 DPC 执行模式
  - `active_mask`: 显示当前线程活跃掩码
- 分支发散过程可逐指令追踪：观察 taken_mask 的切换和 SIMT Stack 的 push/pop

### warp 寄存器行为测试
- 分支发散场景下验证每线程寄存器独立性：then 路径和 else 路径中同一寄存器不会相互干扰
- 验证 JMP 指令后目标线程的正确性：仅 taken_mask 中的线程更新寄存器
- 验证重汇聚后所有线程寄存器状态完整，未因发散丢失数据

### trace 输出分析
- DPC 模式 trace 显示发散过程：
  - 分支指令处：active_mask 分割
  - then 路径：active_mask = taken_mask
  - else 路径：active_mask = not_taken_mask
  - 重汇聚点：active_mask 恢复 full mask
- Predication 模式 trace 显示：
  - active_mask 始终保持 full mask
  - 条件操作根据断言位决定写回与否
- trace 格式扩展包含 `SIMTStackDepth` 和 `ActiveMask` 字段

---

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
