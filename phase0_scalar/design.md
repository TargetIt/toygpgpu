# Phase 0: 标量处理器 — 设计文档

## 1. 架构总览

对标 GPGPU-Sim 架构中最底层的执行单元。Phase 0 的标量处理器 = GPGPU-Sim 中一个 SP（Streaming Processor）的简化版。

```
┌─────────────────────────────────────────────────┐
│                    CPU (cpu.py)                  │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Register │   │   ALU    │   │   Memory   │  │
│  │   File   │   │ (alu.py) │   │ (memory.py)│  │
│  │ (reg.py) │   │          │   │            │  │
│  │ 16×32bit │   │ + - * /  │   │ 256×32bit  │  │
│  └──────────┘   └──────────┘   └────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         ISA / Decoder (isa.py)            │   │
│  │    Instruction fetch → decode → execute   │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │       Assembler (assembler.py)            │   │
│  │    Text assembly → Machine code           │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 2. 指令编码设计（参考 RISC-V RV32I 格式）

每条指令 32 bits，采用 RISC-V 风格的定长编码：

```
 31    24 23   20 19   16 15   12 11         0
┌────────┬───────┬───────┬───────┬────────────┐
│ opcode │  rd   │  rs1  │  rs2  │ imm/addr   │
│  8-bit │ 4-bit │ 4-bit │ 4-bit │   12-bit   │
└────────┴───────┴───────┴───────┴────────────┘
```

各字段说明：
- **opcode[7:0]**: 操作码，决定指令类型
- **rd[3:0]**: 目的寄存器编号
- **rs1[3:0]**: 源寄存器1编号
- **rs2[3:0]**: 源寄存器2编号（仅 R-type 指令使用）
- **imm[11:0]**: 12-bit 立即数或地址偏移

指令类型：
- **R-type** (ADD/SUB/MUL/DIV): 使用 rs1, rs2, rd。imm 忽略
- **I-type** (MOV/LD): 使用 rs1(或忽略), rd, imm
- **S-type** (ST): 使用 rs1(源), imm(地址)。rd 忽略
- **HALT**: 仅 opcode 有效

## 3. 模块设计

### 3.1 RegisterFile (reg.py)

```python
class RegisterFile:
    regs: list[int]  # 16 × 32-bit, regs[0] always 0

    def read(reg_id: int) -> int      # 读寄存器
    def write(reg_id: int, value: int) # 写寄存器 (r0 写忽略)
```

**设计参考**: TinyGPU 的 register file 使用 Python list 存储，每个线程一个独立 register file。Phase 0 只有 1 个线程，所以只有 1 个 register file。

### 3.2 ALU (alu.py)

```python
class ALU:
    @staticmethod
    def add(a, b)    # 有符号加
    def sub(a, b)    # 有符号减
    def mul(a, b)    # 有符号乘 (取低32位)
    def div(a, b)    # 有符号除 (除零返回0)
```

所有操作将操作数视为 **有符号 32-bit 整数**。

**设计参考**: GPGPU-Sim 的 `simd_function_unit` 执行 SP/SFU/INT 运算。Phase 0 仅有 ALU 部分。

### 3.3 Memory (memory.py)

```python
class Memory:
    data: bytearray  # 1024 bytes (256 × 32-bit words)

    def read_word(addr: int) -> int   # 读字
    def write_word(addr: int, value: int) # 写字
```

按字寻址（0-255）。内部用 bytearray 存储（1024 bytes = 256 words × 4 bytes/word）。

**设计参考**: GPGPU-Sim 的 `memory_partition_unit` 管理 L2 + DRAM。Phase 0 只有平坦内存。

### 3.4 ISA & Decoder (isa.py)

```python
class Instruction:
    opcode: int
    rd: int
    rs1: int
    rs2: int
    imm: int

def decode(instruction_word: int) -> Instruction

# Opcode constants
OP_HALT = 0x00
OP_ADD  = 0x01
OP_SUB  = 0x02
OP_MUL  = 0x03
OP_DIV  = 0x04
OP_LD   = 0x05
OP_ST   = 0x06
OP_MOV  = 0x07
```

**设计参考**: GPGPU-Sim 的 `ptx_parser` + `instructions.cc` 译码和执行 PTX 指令。Phase 0 用简化的自定义 ISA。

### 3.5 CPU (cpu.py)

```python
class CPU:
    pc: int
    reg_file: RegisterFile
    alu: ALU
    memory: Memory
    program: list[int]  # 程序内存 (机器码)

    def load_program(program: list[int])
    def step() -> bool     # 执行一条指令, 返回 True=继续, False=HALT
    def run()              # 运行到 HALT
    def dump_state()       # 打印当前状态
```

执行循环：
```
while True:
    instr = decode(program[pc])
    pc += 1
    execute(instr)
    if instr.opcode == HALT: break
```

**设计参考**: GPGPU-Sim 中 `shader_core_ctx::cycle()` 执行 6 级流水线。Phase 0 简化为单周期执行。

### 3.6 Assembler (assembler.py)

```python
def assemble(source: str) -> list[int]
# "ADD r1, r2, r3" → [0x01230000]
```

支持的语法：
- `ADD r1, r2, r3` — 寄存器-寄存器运算
- `MOV r1, 42` — 加载立即数
- `LD r1, [100]` — 加载内存
- `ST r1, [100]` — 存储到内存
- `HALT` — 停止
- `# comment` — 注释
- `label:` — 标签（解析但不跳转，Phase 0 顺序执行）

## 4. 执行示例

```asm
; 计算 5 + 3，结果存入 memory[100]
MOV r1, 5        ; 0x07100005  r1 = 5
MOV r2, 3        ; 0x07200003  r2 = 3
ADD r3, r1, r2   ; 0x01312000  r3 = r1 + r2 = 8
ST r3, [100]     ; 0x06300064  mem[100] = r3
HALT             ; 0x00000000  stop
```

## 5. 与 GPGPU-Sim 的对应关系

| GPGPU-Sim 概念 | Phase 0 实现 | 未来扩展 |
|---------------|-------------|---------|
| 1 thread / 1 SP | CPU (cpu.py) | → SIMD lanes (Phase 1) |
| shd_warp_t | 无 | → warp.py (Phase 2) |
| simt_stack | 无 | → simt_stack.py (Phase 3) |
| scoreboard | 无 | → scoreboard.py (Phase 4) |
| memory_partition | Memory (memory.py) | → 多级缓存 (Phase 5) |
| ptx_parser | Assembler (assembler.py) | → 更丰富的 ISA |
