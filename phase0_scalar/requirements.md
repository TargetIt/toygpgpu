# Phase 0: 标量处理器 — 需求分解

## New Features (2026-05-15)

The following features were added after the initial release:

- **learning_console.py**: Added an interactive step-through debugger that allows users to single-step through program execution, inspect register state, and observe the fetch-decode-execute-writeback cycle in real time.
- **--trace mode**: Added support for `cpu.run(trace=True)` in Python and `run.sh --trace` from the command line, enabling per-instruction execution tracing with register and memory state dumps.
- **Bilingual comments and ASCII flow diagrams**: All `.asm` programs were updated with Chinese/English bilingual comments and ASCII flow diagrams illustrating program logic and data flow.

## 1. 目标

实现一个最简标量处理器（Scalar CPU），作为 toygpgpu 的起点。

对标 GPGPU-Sim 的最小抽象单元：**一条指令在一个数据上操作**。

## 2. 功能需求

### FR-01: 指令集
处理器必须支持以下 8 条指令：

| 指令 | 格式 | 功能 | 编码 |
|------|------|------|------|
| ADD | ADD rd, rs1, rs2 | rd = rs1 + rs2 | 0x01 |
| SUB | SUB rd, rs1, rs2 | rd = rs1 - rs2 | 0x02 |
| MUL | MUL rd, rs1, rs2 | rd = rs1 * rs2 (低32位) | 0x03 |
| DIV | DIV rd, rs1, rs2 | rd = rs1 / rs2 | 0x04 |
| LD  | LD rd, addr | rd = mem[addr] | 0x05 |
| ST  | ST rs, addr | mem[addr] = rs | 0x06 |
| MOV | MOV rd, imm | rd = imm (符号扩展) | 0x07 |
| HALT| HALT | 停止执行 | 0x00 |

### FR-02: 指令编码
每条指令 32-bit 定长编码：
```
[31:24] opcode   (8 bits)
[23:20] rd       (4 bits, 目的寄存器 0-15)
[19:16] rs1      (4 bits, 源寄存器1)
[15:12] rs2      (4 bits, 源寄存器2)
[11:0]  imm/addr (12 bits, 立即数或地址)
```

### FR-03: 寄存器堆
- 16 个 32-bit 通用寄存器 (r0-r15)
- r0 硬连线为 0（写忽略，读恒为 0）
- 支持双读单写（1 cycle 内可读 2 个寄存器 + 写 1 个）

### FR-04: 内存
- 256 words × 32-bit (1KB)
- 按字寻址（地址 0-255）
- 初始化为全 0

### FR-05: 执行模型
- 单步执行：fetch → decode → execute → writeback
- PC 从 0 开始，每条指令后 +1
- HALT 指令停止执行
- 无分支指令（Phase 0 仅顺序执行）

### FR-06: 汇编器
- 文本汇编 → 机器码转换
- 支持标签和注释
- 支持寄存器名 (r0-r15) 和数字立即数

## 3. 非功能需求

### NFR-01: 可测试性
- 每个模块独立可测
- 提供至少 5 个测试程序

### NFR-02: 可读性
- 每个类/函数有中文 docstring
- 关键逻辑有注释

### NFR-03: 可运行
- 一键脚本运行所有测试
- 仅依赖 Python 3.9+ 标准库

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | 基本运算程序正确：MOV + ADD + ST 输出预期结果 |
| AC-02 | 乘除程序正确 |
| AC-03 | LD/ST 访存正确 |
| AC-04 | r0 写保护有效 |
| AC-05 | HALT 正常终止 |
| AC-06 | 5+ 个测试程序全部通过 |
| AC-07 | `bash run.sh` 一键运行全部测试 |
