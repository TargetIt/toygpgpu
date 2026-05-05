# Phase 8 设计: PTX 前端

## 编译管道
```
.ptx file → Tokenize → Parse → PTX IR → Translate → asm → Machine Code
```

## Tokenizer
- 正则匹配: 虚拟寄存器 %r\d+, 特殊寄存器 %tid.x 等
- 注释处理: ; 行注释, // 行内注释
- 括号合并: [ N ] → [N]

## Parser
- 跳过 .entry/.reg 等声明
- 收集操作数直到 ; (合并括号地址)
- 构建 PtxProgram IR

## Register Allocator
- 线性扫描: virtual reg_id + 1 = physical register
- 上限 10 个物理寄存器 (r1-r10, r0 保留)
- %tid.x → TID 指令

## Translator
- mov.u32 %r, imm → MOV r, imm
- mov.u32 %r, %tid.x → TID r
- ld.global.u32 %r, [addr] → LD r, [addr]
- ret → HALT
