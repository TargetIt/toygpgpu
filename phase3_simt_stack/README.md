# Phase 3: SIMT Stack (分支发散)

实现 SIMT Stack 处理 warp 内分支发散与重汇聚。对标 GPGPU-Sim `simt_stack`。

## 新增概念
- JMP/BEQ/BNE 分支指令
- Two-pass 汇编器 (label 解析)
- SIMT Stack: push(发散) → pop(重汇聚)
- Active mask 分拆与恢复
- JMP merge 模式

## 运行

```bash
cd phase3_simt_stack && bash run.sh
```

## 测试
32 项测试，含 if/else 偶数/奇数线程发散、tid 分组分支。

## 对标 GPGPU-Sim
`simt_stack` (IPDOM 后支配栈)
