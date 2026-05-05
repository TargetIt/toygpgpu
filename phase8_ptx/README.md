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
