# Phase 1: SIMD 向量处理器 — 设计文档

## 1. 架构演进

从 Phase 0 的标量处理器，增加向量化能力：

```
Phase 0 (标量)                  Phase 1 (SIMD)
┌──────────┐                   ┌──────────────────────────┐
│ Register │                   │ Scalar RegFile  Vector RF│
│  File    │                   │  r0-r15 (16)   v0-v7 (8)│
│ 16×32bit │                   │  每个32bit     每个VLEN×│
└──────────┘                   │                32bit    │
     │                         └──────┬──────────┬───────┘
     ▼                                │          │
┌──────────┐                   ┌──────▼──┐  ┌───▼───────┐
│   ALU    │                   │ Scalar  │  │ Vector ALU│
│ 1 lane   │                   │   ALU   │  │ VLEN lanes│
└──────────┘                   └──────────┘  └───────────┘
```

## 2. 指令编码扩展

复用 Phase 0 的 32-bit 编码，新增向量指令的操作码：

```
 31    24 23   20 19   16 15   12 11         0
┌────────┬───────┬───────┬───────┬────────────┐
│ opcode │  rd   │  rs1  │  rs2  │ imm/addr   │
└────────┴───────┴───────┴───────┴────────────┘
```

新增向量操作码：

| 指令 | 编码 | 类型 |
|------|------|------|
| VADD | 0x11 | VR-type (vd, vs1, vs2) |
| VSUB | 0x12 | VR-type |
| VMUL | 0x13 | VR-type |
| VDIV | 0x14 | VR-type |
| VLD  | 0x15 | VI-type (vd, addr) |
| VST  | 0x16 | VS-type (vs, addr) |
| VMOV | 0x17 | VI-type (vd, imm) |

对于向量指令，rd/rs1/rs2 字段指定向量寄存器 v0-v7（复用相同的 4-bit 编码空间）。

## 3. 模块设计

### 3.1 VectorRegisterFile（新增）

```python
class VectorRegisterFile:
    vlen: int  # 向量长度（lane 数）
    regs: list[list[int]]  # 8 × VLEN × 32-bit

    def read(reg_id) -> list[int]
    def write(reg_id, values: list[int])
```

### 3.2 VectorALU（新增）

```python
class VectorALU:
    vlen: int

    def vadd(a, b) -> list[int]  # VLEN 路并行加法
    def vsub(a, b) -> list[int]
    def vmul(a, b) -> list[int]
    def vdiv(a, b) -> list[int]
```

### 3.3 CPU 扩展

在 Phase 0 CPU 基础上增加：
- `vec_reg_file: VectorRegisterFile`
- `vec_alu: VectorALU`
- `_execute` 方法增加向量指令处理
- 内存大小从 256 扩展到 1024 words

### 3.4 Assembler 扩展

新增对向量指令的解析：
- `VADD v1, v2, v3`
- `VMOV v1, 42`
- `VLD v1, [addr]`
- `VST v1, [addr]`

## 4. 与 GPGPU-Sim 的对应关系

| GPGPU-Sim | Phase 1 |
|-----------|---------|
| Warp 内 32 threads | VLEN lane 向量 |
| simd_function_unit 的 SP pipeline | VectorALU (VLEN lane 并行) |
| warp 统一 PC | 单一 PC，全 lane 同指令 |
| 尚无 SIMT stack | 无（Phase 3 加入） |
| 尚无 scoreboard | 无（Phase 4 加入） |

GPGPU-Sim 的关键 insight：**warp 本质就是 SIMD**。32 个线程共享一个 PC，执行同一条指令。Phase 1 把"线程"抽象为"lane"，为 Phase 2 的 warp 模型打基础。

## 5. 典型程序：向量加法

```asm
; C[i] = A[i] + B[i], i = 0..7
; A 在 mem[0..7], B 在 mem[8..15], C 存到 mem[16..23]

VLD  v1, [0]     ; v1 = mem[0..7]   (A)
VLD  v2, [8]     ; v2 = mem[8..15]  (B)
VADD v3, v1, v2   ; v3 = A + B
VST  v3, [16]    ; mem[16..23] = C
HALT
```

仅 4 条指令完成 8 对数据的加法——这就是 SIMD 的威力。

## 6. 配置

```python
# config.py
VLEN = 8           # 向量长度（lane 数）
MEM_SIZE = 1024    # 内存大小 (words)
NUM_VEC_REGS = 8   # 向量寄存器数
```
