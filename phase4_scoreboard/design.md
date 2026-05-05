# Phase 4: Scoreboard — 设计文档

## 1. 架构

对标 GPGPU-Sim scoreboard 的实现。

```
Warp
├── simt_stack (Phase 3)
├── scoreboard (Phase 4)  ← NEW
│   ├── reserved[16]: int  # 每个寄存器的剩余延迟周期
│   │    0 = 没有 pending write
│   │    >0 = 有 pending write, 还需要 N 个周期
│   └── advance()   # 每周期减 1
│   └── check_raw(rs1, rs2) → bool   # RAW 冒险?
│   └── check_waw(rd) → bool          # WAW 冒险?
│   └── reserve(rd, latency)          # 标记 pending write
```

## 2. 执行流程

```
SIMTCore::step():
  1. For each warp: scoreboard.advance()  # 推进延迟
  2. scheduler.select_warp()  # 跳过 stalled warps
  3. if warp is stalled by scoreboard: skip
  4. check scoreboard before issue:
     - check_raw(rs1, rs2) → if True, stall
     - check_waw(rd) → if True, stall
  5. execute instruction
  6. reserve destination register with pipeline latency
```

## 3. Scoreboard 数据结构

```python
class Scoreboard:
    reserved: dict[int, int]  # reg_id → remaining_cycles
    
    def advance():
        """每周期减少所有 pending 的剩余周期"""
        for reg in list(reserved):
            reserved[reg] -= 1
            if reserved[reg] <= 0:
                del reserved[reg]
    
    def check_raw(rs1, rs2) -> bool:
        """源寄存器是否有 pending write (RAW)"""
        return (rs1 != 0 and rs1 in reserved) or (rs2 != 0 and rs2 in reserved)
    
    def check_waw(rd) -> bool:
        """目的寄存器是否有 pending write (WAW)"""
        return rd != 0 and rd in reserved
    
    def reserve(rd, latency):
        """标记寄存器有 pending write"""
        if rd != 0:
            reserved[rd] = latency
```

## 4. 流水线延迟配置

```python
PIPELINE_LATENCY = {
    OP_ADD: 1, OP_SUB: 1, OP_MUL: 1, OP_DIV: 1,
    OP_LD: 4, OP_ST: 4,   # 访存延迟
    OP_MOV: 1, OP_TID: 1, OP_WID: 1, OP_BAR: 1,
    OP_JMP: 1, OP_BEQ: 1, OP_BNE: 1,
}
```

## 5. Scheduler 修改

`select_warp()` 需跳过 scoreboard-stalled warps：
```python
def select_warp(self):
    for _ in range(num_warps):
        warp = self.warps[self.current_idx]
        self.current_idx = (self.current_idx + 1) % num_warps
        if not warp.done and not warp.at_barrier and not warp.scoreboard_stalled:
            return warp
    return None
```

## 6. GPGPU-Sim 对照

| GPGPU-Sim | Phase 4 |
|-----------|---------|
| `scoreboard::check_collision()` | `check_raw()` + `check_waw()` |
| `scoreboard::reserve_reg()` | `reserve()` |
| `scoreboard::release_reg()` | auto on cycle expiration |
| `shader_core_ctx::issue()` 中的 scoreboard 检查 | `step()` 中 issue 前检查 |
