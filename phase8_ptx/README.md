## Quick Start

Interactive debugging with learning_console.py (PTX translation pipeline):

```bash
# Interactive debugging with PTX program
python src/learning_console.py tests/programs/01_vector_add.ptx --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_vector_add.ptx --trace --warp-size 4

# Scale kernel demo
python src/learning_console.py tests/programs/02_scale.ptx --warp-size 4

# Run with more cycles for longer programs
python src/learning_console.py tests/programs/03_mov_imm.ptx --max-cycles 300
```

## New in this update

- **learning_console.py**: Interactive PTX debugger with virtual-to-physical register allocation display, PTX->ISA translation pipeline inspection
- **Trace mode**: `--trace` for batch execution with translation and execution trace
- **PRED/Predication**: `@p0` predicate support in PTX translator, SETP mapping from PTX setp instructions
- **vec4/float4 instructions**: V4PACK/V4ADD/V4MUL/V4UNPACK support in the PTX-to-ISA translation layer
- **Warp-level registers**: WREAD/WWRITE accessible from translated PTX kernels

# Phase 8: PTX Frontend

PTX 解析器 + 翻译器，将 CUDA PTX 子集编译为 toygpgpu 内部 ISA。

## 新增
- Tokenizer: 正则词法分析
- Parser: 解析 .entry blocks, 指令, 操作数
- Register Allocator: 虚拟 %r → 物理 r 线性扫描
- Translator: PTX ops → internal ISA
- `assemble_ptx()`: 完整 PTX → 机器码 pipeline

## 支持的 PTX 指令
mov, add, mul, ld/st.global, bra, ret, setp, %tid.x

## 运行

```bash
cd phase8_ptx && bash run.sh
```

## 对标 GPGPU-Sim
`cuda-sim/ptx_parser` (ptx.l/ptx.y)
