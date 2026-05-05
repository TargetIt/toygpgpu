# Phase 7: 流水线解耦 — I-Buffer + Operand Collector

## 1. 目标

将 Phase 6 的简化流水线（fetch→execute 合一）升级为 GPGPU-Sim 风格的多级流水线：

```
原:  FETCH+EXECUTE (1 stage)
新:  FETCH → DECODE→I-Buffer → ISSUE(check bank) → EXECUTE → WRITEBACK
```

## 2. I-Buffer (per-warp 指令缓冲)

对标 GPGPU-Sim 的 I-Buffer：fetch 和 issue 解耦。

| 属性 | 说明 |
|------|------|
| 容量 | 每 warp 2 条指令槽 |
| 分区 | 静态分区（各 warp 独立） |
| valid 位 | fetch 写入后置 1 |
| ready 位 | decode + scoreboard check 后置 1 |
| 调度 | Scheduler 从 I-Buffer 中选 ready 的 warp |

## 3. Operand Collector (banked register file)

对标 GPGPU-Sim 的 `opndcoll_rfu_t`。

| 属性 | 说明 |
|------|------|
| Bank 数 | 4 banks |
| 寄存器→bank | bank_id = reg_id % 4 |
| 读端口 | 每 bank 1 个读端口/cycle |
| Bank conflict | rs1 和 rs2 同 bank → 需 2 cycles 串行读 |
| No conflict | rs1 和 rs2 不同 bank → 1 cycle 并行读 |

## 4. 验收标准

| 编号 | 标准 |
|------|------|
| AC-01 | I-Buffer fetch→decode→issue 解耦正确 |
| AC-02 | Bank conflict 检测正确 (同 bank → +1 cycle) |
| AC-03 | No conflict 时操作数 1 cycle 收集 |
| AC-04 | Phase 0-6 测试保持通过 |
