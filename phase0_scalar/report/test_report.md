# Phase 0 测试报告

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
