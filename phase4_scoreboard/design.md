# Phase 4: Scoreboard — Design Document / 设计文档

> **对应 GPGPU-Sim**: `scoreboard` 类 (`gpgpu-sim/scoreboard.h`), `shader_core_ctx::issue()` 中的 scoreboard 检查
> **参考**: GPGPU-Sim `gpgpu-sim/scoreboard.h`, `shader_core_ctx::cycle()`

## 1. Introduction / 架构概览

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     SIMTCore (Phase 4)                       │
  │                                                              │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
  │  │  Warp 0  │    │  Warp 1  │    │  Warp N  │               │
  │  │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │               │
  │  │ │Scoreb│ │    │ │Scoreb│ │    │ │Scoreb│ │               │
  │  │ │ oard │ │    │ │ oard │ │    │ │ oard │ │               │
  │  │ │reserv│ │    │ │reserv│ │    │ │reserv│ │               │
  │  │ │ed[r] │ │    │ │ed[r] │ │    │ │ed[r] │ │               │
  │  │ └──────┘ │    │ └──────┘ │    │ └──────┘ │               │
  │  │ SIMTStack│    │ SIMTStack│    │ SIMTStack│               │
  │  │ Threads  │    │ Threads  │    │ Threads  │               │
  │  └──────────┘    └──────────┘    └──────────┘               │
  │         \              |             /                       │
  │          ┌─────────────┴─────────────┐                       │
  │          │   WarpScheduler (RR)      │                       │
  │          │   skip stalled warps      │                       │
  │          └─────────────┬─────────────┘                       │
  │                        v                                     │
  │          ┌─────────────────────────┐                         │
  │          │   ALU / Vec4ALU        │                         │
  │          │   Memory / Scoreboard   │                         │
  │          └─────────────────────────┘                         │
  └──────────────────────────────────────────────────────────────┘
```

Phase 4 为 toygpgpu 引入寄存器记分板 (Scoreboard)，实现数据冒险检测和流水线延迟建模。每个 warp 拥有独立的 Scoreboard 实例，跟踪目的寄存器的 pending write 状态。在指令发射前检查 RAW (Read After Write) 和 WAW (Write After Write) 冒险，检测到冲突时 stall warp 直到写回完成。

## 2. Motivation / 设计动机

Phase 3 之前的流水线是"理想化"的：所有指令在一个周期内完成执行和写回，不存在任何数据冒险。真实的 GPU 流水线具有多周期执行单元——例如内存加载 (LD) 通常需要数十到数百个周期——因此 RAW 和 WAW 冲突是必须解决的现实问题。

- **RAW (Read After Write)**：后续指令读取前一条指令尚未写回的寄存器。没有检测机制会导致读取到过时值。
- **WAW (Write After Write)**：两条连续指令写入同一寄存器，第二条可能在第一条完成前覆盖，最终值不确定。
- **寄存器 r0 豁免**：r0 硬连线为常数 0，不参与冒险检测。

GPGPU-Sim 用 `scoreboard::check_collision()` 实现 RAW+WAW 检测，用 `reserve_reg()` 标记 pending 寄存器，用 `release_reg()` 在写回后清除。Phase 4 简化了 release 机制——通过 `advance()` 每周期自动递减计数器，到期自动清除。

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 记分板核心算法

每个 warp 配备一个 `Scoreboard` 实例，内含 `reserved: dict[int, int]` 映射 `{寄存器编号: 剩余周期数}`。

```
算法: Scoreboard 冒险检测与延迟管理
────────────────────────────────────────────────────
初始化: reserved = {}

advance():
    每周期调用一次
    for each (reg_id, remaining_cycles) in reserved:
        remaining_cycles -= 1
        if remaining_cycles <= 0:
            delete reserved[reg_id]   # 写回完成

check_raw(rs1, rs2) → bool:
    return (rs1 != 0 AND rs1 in reserved) OR (rs2 != 0 AND rs2 in reserved)

check_waw(rd) → bool:
    return rd != 0 AND rd in reserved

reserve(rd, latency):
    if rd != 0:
        reserved[rd] = latency   # 标记为 pending write
```

### 3.2 流水线延迟配置

```python
PIPELINE_LATENCY = {
    'alu': 1,       # ALU 指令 (ADD, SUB, MUL, DIV): 1 周期
    'mem': 4,       # LD/ST 指令: 4 周期 (地址计算+缓存访问+写回)
    'default': 1,   # 其他指令: 1 周期
}
```

控制指令 (HALT, BAR, JMP, BEQ, BNE, WWRITE) 不写寄存器，延迟为 0，不触发冒险。

### 3.3 调度器集成

`select_warp()` 跳过 scoreboard-stalled 的 warp：

```
select_warp():
    for _ in range(num_warps):
        warp = warps[current_idx]
        current_idx = (current_idx + 1) % num_warps
        if warp.done: continue
        if warp.at_barrier: continue
        if warp.scoreboard_stalled: continue   # ← Phase 4 新增
        return warp
    return None
```

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| 文件 | 作用 |
|------|------|
| `scoreboard.py` | Scoreboard 类 + PIPELINE_LATENCY 配置 |
| `simt_core.py` | SIMTCore 集成 scoreboard 检查和 reserve |
| `warp.py` | Warp 类新增 scoreboard 和 scoreboard_stalled 属性 |
| `scheduler.py` | WarpScheduler 跳过 scoreboard_stalled warp |
| `learning_console.py` | 交互式调试控制台，新增 `sb` 命令查看 scoreboard |
| `isa.py` | 指令集编码/译码 (无变更) |
| `alu.py` | ALU 运算单元 (无变更) |
| `memory.py` | 全局内存模型 (无变更) |
| `assembler.py` | 汇编器 (无变更) |

### 4.2 Key Data Structures / 关键数据结构

```python
class Scoreboard:
    """寄存器记分板 (每个 warp 一个实例)
    
    Attributes:
        reserved: dict[int, int]  — {reg_id: remaining_cycles}
                                    0 = 没有 pending write
                                    >0 = 还有 N 周期写回
    """
    def advance()                    # 每周期递减计数器
    def check_raw(rs1, rs2) -> bool  # RAW 冒险检测
    def check_waw(rd) -> bool        # WAW 冒险检测
    def reserve(rd, latency)         # 标记 pending write
    stalled: bool  (property)        # 是否有任何 pending write
```

### 4.3 Module Interfaces / 模块接口

```
SIMTCore::step():
    1. for each warp: scoreboard.advance()           # 推进延迟
       if not scoreboard.stalled: clear stall flag   # 恢复调度
    2. warp = scheduler.select_warp()                # 跳过 stalled warps
    3. if warp.simt_stack.at_reconvergence(warp.pc): # 重汇聚处理
    4. decode instruction
    5. if check_waw(rd) or check_raw(rs1, rs2):      # scoreboard 检查
         warp.scoreboard_stalled = True → stall
    6. execute instruction
    7. reserve(rd, latency)                          # 标记 pending
```

## 5. Functional Processing Flow / 功能处理流程

### 示例：RAW 冒险 (`01_raw_hazard.asm`)

程序: `MOV r1,10; MOV r2,5; ADD r3,r1,r2; ST r3,[0]; HALT`

```
周期  SIMT Core 行为                        Scoreboard 状态
─────────────────────────────────────────────────────────────
0     Warp0: advance (无 pending)           (clean)
      scheduler → Warp0, PC=0
      decode MOV r1,10
      check: no hazard (r1=0 not in reserved)
      execute: r1=10
      reserve(r1, latency=1)                r1=1
      
1     Warp0: advance (r1: 1→0, 到期清除)   (clean)
      scheduler → Warp0, PC=1
      decode MOV r2,5
      check: no hazard (r2=0 not in reserved)
      execute: r2=5
      reserve(r2, latency=1)                r2=1
      
2     Warp0: advance (r2: 1→0, 到期清除)   (clean)
      scheduler → Warp0, PC=2
      decode ADD r3,r1,r2
      check_raw(r1,r2) → r2 在 reserved 中? 
      → YES! r2 还有 1 周期 → STALL          r2=1
      
3     Warp0: advance (r2: 1→0, 到期清除)   (clean)
      scheduler → Warp0 (不再 stalled)
      decode ADD r3,r1,r2
      check: no hazard                      (clean)
      execute: r3=15
      reserve(r3, latency=1)                r3=1
      
4     Warp0: advance (r3: 1→0, 到期清除)   (clean)
      scheduler → Warp0, PC=3
      ST r3,[0] → mem[0]=15
      
5     Warp0: HALT → done
```

## 6. Comparison with Phase 3 / 与前一版本的对比

| Aspect | Phase 3 (SIMT Stack) | Phase 4 (Scoreboard) | Change |
|--------|----------------------|----------------------|--------|
| 依赖检测 | 无 | RAW + WAW 检测 | 新增 scoreboard |
| 流水线延迟 | 假设所有指令 1 周期 | 可配置 (ALU=1, MEM=4) | 延迟模型 |
| 写回模型 | 立即写回 | 延迟写回 (pending) | 延迟管理 |
| warp stall 原因 | barrier 等待 | barrier + scoreboard | 双重 stall |
| r0 处理 | 普通寄存器 | 硬连线 0, 豁免检测 | 特殊处理 |
| 调度器 | 跳过 barrier warp | 跳过 barrier + scoreboard | 跳过多条件 |
| 学习控制台 | `stack` 命令 | 新增 `sb` 命令 | 调试能力 |
| 新增文件 | — | `scoreboard.py` | 1 个新文件 |
| 修改文件 | — | `warp.py`, `simt_core.py`, `scheduler.py` | 集成 |

## 7. Known Issues and Future Work / 遗留问题与后续工作

- **简化释放机制**：当前 `advance()` 按固定周期递减，未模拟真实硬件写回总线的时序。真实 GPU 的 scoreboard 在写回阶段显式调用 `release_reg()`。
- **单周期 ALU**：所有 ALU 指令延迟均为 1，未区分不同执行单元（如 INT vs FP）的延迟差异。
- **无转发 (forwarding)**：未实现寄存器值转发（bypass）。GPGPU-Sim 支持写前推以减少 stall 周期。
- **Per-warp scoreboard**：当前每个 warp 独立记分板，未模拟寄存器文件级别的依赖（不同 warp 访问同一寄存器在真实硬件中通常通过 bank 仲裁，参见 Phase 7）。
- **非阻塞访问**：LD/ST 延迟期间其他 warp 仍可执行，但 Phase 4 未显式建模内存请求队列。
