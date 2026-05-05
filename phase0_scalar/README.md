# Phase 0: 标量处理器

实现一个 8 指令的最简 CPU，对标 GPGPU-Sim 的 `simd_function_unit`（单 SP）。

## 核心概念
- ISA 设计 (RISC-V 风格 32-bit 定长编码)
- 寄存器堆 16×32bit (r0 硬连线 0)
- ALU (有符号 32-bit 加减乘除)
- 汇编器 (文本 asm → 机器码)

## 运行

```bash
cd phase0_scalar && bash run.sh
```

## 测试
45 项单元+集成测试，6 个汇编程序。

## 文件

| 文件 | 说明 |
|------|------|
| `src/isa.py` | 指令集定义与译码 |
| `src/register_file.py` | 寄存器堆 |
| `src/alu.py` | 算术逻辑单元 |
| `src/memory.py` | 256字内存 |
| `src/cpu.py` | 标量处理器顶层 |
| `src/assembler.py` | 汇编器 |
| `tests/` | 测试套件 + 6 个 asm 程序 |

## 对标 GPGPU-Sim
`simd_function_unit` (SP pipeline)
