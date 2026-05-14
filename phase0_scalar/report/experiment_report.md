# Phase 0 实验报告

## 2026-05-15: Feature Additions

### learning_console.py 交互式调试体验
- 新增 `learning_console.py` 为所有 phase 提供统一的交互式调试前端
- Phase 0 标量模式下支持以下调试命令：
  - `step` / `s`: 单步执行一条指令，每次 step 后显示当前指令和寄存器状态
  - `run` / `r`: 连续执行至程序结束或断点触发
  - `disasm` / `d`: 反汇编当前加载的程序，显示 PC 到指令的映射
  - `regs`: 打印所有标量寄存器 r0-r15 的当前值
  - `mem <addr>`: 读取指定内存地址的内容
  - `break <addr>` / `b <addr>`: 在指定 PC 设置断点
  - `trace on` / `trace off`: 启用/禁用每 cycle 详细执行日志
- 调试器自动检测当前 phase 模块，加载对应的 CPU、汇编器和内存模型
- 执行历史记录维护在 `self.history` 列表中，支持后续分析和可视化

### trace 输出分析
- trace 模式提供 cycle 级精度的执行追踪，每条 trace 记录包含：
  - `Cycle`: 当前执行周期数
  - `PC`: 程序计数器（指令地址）
  - `Insn`: 指令助记符和操作数
  - `ALU`: ALU 运算的输入值和结果
  - `Regs`: 寄存器写回详情（rd ← 值）
  - `Mem`: 内存访问详情（LD/ST 地址和数据）
- 标量 trace 示例验证：`MOV r1, 5` → `ADD r2, r1, r1` 序列正确显示 r1=5, r2=10
- trace 输出可在 phase0-4 间保持一致格式，便于跨 phase 比较执行行为

---

## 1. 实验概述

**目标**: 实现一个标量处理器（Scalar CPU），作为 toygpgpu 项目的起点。

**对标**: GPGPU-Sim 中最底层的 SP（Streaming Processor）执行单元。

## 2. 实现内容

### 2.1 指令集
实现了 8 条指令的完整执行：

| 指令 | 格式 | 功能 |
|------|------|------|
| ADD | R-type | 有符号加法 |
| SUB | R-type | 有符号减法 |
| MUL | R-type | 有符号乘法（低32位）|
| DIV | R-type | 有符号除法（除零返回0）|
| LD | I-type | 内存加载 |
| ST | S-type | 内存存储 |
| MOV | I-type | 立即数加载 |
| HALT | — | 停止执行 |

### 2.2 硬件组件

| 组件 | 代码文件 | 规模 | GPGPU-Sim 对应 |
|------|---------|------|---------------|
| ISA/Decoder | isa.py | 110行 | ptx_parser + instructions.cc |
| RegisterFile | register_file.py | 60行 | shader_core_ctx 寄存器堆 |
| ALU | alu.py | 50行 | simd_function_unit (SP pipe) |
| Memory | memory.py | 60行 | memory_partition_unit (简化) |
| CPU | cpu.py | 120行 | gpgpu_sim + shader_core_ctx |
| Assembler | assembler.py | 110行 | ptx_parser |

### 2.3 指令编码 (RISC-V 风格)

32-bit 定长编码，8-bit opcode + 三个 4-bit 寄存器字段 + 12-bit 立即数。

### 2.4 执行模型

单周期顺序执行：Fetch → Decode → Execute → Writeback
无流水线重叠（Phase 1+ 加入）。

## 3. 测试结果

- **单元测试**: 38 项，100% 通过
- **集成测试**: 6 个汇编程序，7 个检查点，100% 通过
- **指令覆盖率**: 8/8 (100%)
- **代码总量**: ~500 行 Python

## 4. 参考来源

本实现参考了以下开源项目：

1. **GPGPU-Sim** (C++): 模块划分（ISA/Decoder, ALU, RegisterFile, Memory 的职责拆解）
2. **TinyGPU** (Python): Python 实现 GPU 模拟器的代码风格和组织方式
3. **RISC-V RV32I**: 指令编码格式（8-4-4-4-12 bit field 分配）
4. **SoftGPU** (Python): Warp/Thread 概念（为 Phase 2 做准备）

## 5. 已知限制

1. **无分支**: 仅顺序执行，无 JMP/BEQ 等控制流指令
2. **无流水线**: 单周期执行模型
3. **单线程**: 1 线程 = 1 处理器（无 SIMD/SIMT）
4. **平坦内存**: 256 words，无层次结构
5. **仅整数**: 不支持浮点运算

这些限制将在后续 Phase 中逐步解决。

## 6. 下一步

Phase 1: SIMD 向量处理器
- 将 ALU 扩展为多 lane
- 向量寄存器堆
- 向量指令 (VADD, VSUB, VMUL, VLD, VST)
