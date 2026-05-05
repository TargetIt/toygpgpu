# Phase 7 设计: 流水线解耦

## I-Buffer
- capacity=2 entries/warp
- write(pc) → valid=True, ready=False
- set_ready(pc) → ready=True (decode done)
- consume() → FIFO by min PC
- flush() → branch/reconv 后清空

## Operand Collector
- 4 banks: reg_id % 4
- can_read_operands(rs1, rs2): 检查 bank 占用
- reserve_banks(rs1, rs2): 占用 + 检测 bank conflict (同 bank)
- release_banks(): 每周期释放

## 流水线执行顺序
1. advance scoreboards + release banks
2. ISSUE: scheduler → I-Buffer → scoreboard → bank → execute
3. FETCH/DECODE: fill I-Buffer (每周期 1 warp 1 条)
