# Phase 0 测试报告

## 2026-05-15 更新

### learning_console.py 测试
- **交互式调试器验证**: 确认 learning_console.py 可正确加载 phase0_scalar 模块，支持单步执行（step）、运行（run）、反汇编（disasm）等命令
- **寄存器/内存查看**: `regs` 命令正确显示 r0-r15 状态，`mem` 命令可读取指定地址内容
- **断点功能**: `break` 命令在指定 PC 处正确暂停执行
- **trace 模式**: `trace on` 启用后每 cycle 输出指令执行详情，包括 ALU 操作数、寄存器更新和内存访问

### trace 模式验证
- 验证了 trace 输出格式包含以下字段：cycle 数、当前 PC、指令助记符、ALU 输入操作数、结果值、寄存器写回和内存读写
- 标量指令在 trace 中正确显示单线程执行路径
- 验证 trace 日志可与手动计算结果交叉比对，确认 ALU 运算正确性

### 回归测试
- 原有 45/45 测试全部通过，learning_console.py 和 trace 模式未引入回归问题

---

**日期**: 2026-05-05
**测试套件**: tests/test_phase0.py
**结果**: ✅ 45/45 全部通过

## 单元测试 (38 项)

### ALU (9 项)
| 测试 | 结果 |
|------|------|
| ADD 5+3=8 | ✅ |
| SUB 10-7=3 | ✅ |
| MUL 6*7=42 | ✅ |
| DIV 42/6=7 | ✅ |
| DIV by zero → 0 | ✅ |
| ADD overflow wrap | ✅ |
| SUB underflow wrap | ✅ |
| ADD negative | ✅ |
| MUL negative | ✅ |

### RegisterFile (5 项)
| 测试 | 结果 |
|------|------|
| r0 reads as 0 | ✅ |
| r0 write ignored | ✅ |
| write/read r1 | ✅ |
| 32-bit value preserved | ✅ |
| out-of-range read raises IndexError | ✅ |

### Memory (4 项)
| 测试 | 结果 |
|------|------|
| write/read word | ✅ |
| last address | ✅ |
| out-of-range read raises IndexError | ✅ |
| byte order preserved | ✅ |

### ISA Encoding/Decoding (12 项)
| 测试 | 结果 |
|------|------|
| R-type opcode | ✅ |
| R-type rd | ✅ |
| R-type rs1 | ✅ |
| R-type rs2 | ✅ |
| I-type opcode | ✅ |
| I-type rd | ✅ |
| I-type imm | ✅ |
| S-type opcode | ✅ |
| S-type rs1 | ✅ |
| S-type imm | ✅ |
| HALT opcode | ✅ |
| negative imm sign-extended | ✅ |

### Assembler (8 项)
| 测试 | 结果 |
|------|------|
| 5 instructions assembled | ✅ |
| MOV opcode correct | ✅ |
| MOV rd=r1 correct | ✅ |
| MOV imm=5 correct | ✅ |
| ADD opcode correct | ✅ |
| HALT opcode correct | ✅ |

## 集成测试 (7 项)

| 程序 | 检查点 | 预期 | 结果 |
|------|--------|------|------|
| 01_basic_arith.asm | mem[0] | 8 | ✅ |
| 02_mul_div.asm | mem[0] | 42 | ✅ |
| 02_mul_div.asm | mem[1] | 7 | ✅ |
| 03_memory.asm | mem[30] | 300 | ✅ |
| 04_r0_protect.asm | mem[0] | 42 | ✅ |
| 05_negative.asm | mem[0] | -30 | ✅ |
| 05_negative.asm | mem[1] | -15 | ✅ |
| 05_negative.asm | mem[2] | -5 | ✅ |
| 06_complex.asm | mem[0] | 45 | ✅ |

## 覆盖率

| 指令 | 测试程序 |
|------|---------|
| ADD | 01, 03, 04, 06 |
| SUB | 06 |
| MUL | 02, 05, 06 |
| DIV | 02, 05, 06 |
| LD | 03 |
| ST | 01, 02, 03, 04, 05, 06 |
| MOV | 01-06 |
| HALT | 01-06 |

所有 8 条指令均有覆盖。
