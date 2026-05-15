# Phase 7: Pipeline — Design Document / 设计文档

> **对应 GPGPU-Sim**: I-Buffer (`shader_core_ctx` fetch/decode/issue), Operand Collector (`opndcoll_rfu_t`, `gpgpu-sim/shader.cc`)
> **参考**: GPGPU-Sim inverse pipeline, banked register file arbitration, I-Buffer static partitioning

## 1. Introduction / 架构概览

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         SIMTCore (Phase 7 Pipeline)                      │
  │                                                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐     │
  │  │  Inverse Pipeline (每周期从后往前推进)                           │     │
  │  │                                                                  │     │
  │  │  WRITEBACK ← EXECUTE ← READ_OP ← ISSUE ← DECODE ← FETCH        │     │
  │  │     ↓           ↓           ↓         ↓        ↓        ↓       │     │
  │  │  advance()   _execute_  op_collector ibuffer  set_ready write   │     │
  │  │  release()   _warp()    reserve()    .consume() .set_ready()    │     │
  │  └─────────────────────────────────────────────────────────────────┘     │
  │                                                                          │
  │  ┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐    │
  │  │    Warp 0     │    │     Warp 1        │    │     Warp N        │    │
  │  │ ┌──────┐      │    │  ┌──────┐         │    │  ┌──────┐         │    │
  │  │ │I-Buf │      │    │  │I-Buf │         │    │  │I-Buf │         │    │
  │  │ │[0][1]│      │    │  │[0][1]│         │    │  │[0][1]│         │    │
  │  │ └──────┘      │    │  └──────┘         │    │  └──────┘         │    │
  │  │ Scoreboard    │    │  Scoreboard        │    │  Scoreboard       │    │
  │  │ SIMTStack     │    │  SIMTStack         │    │  SIMTStack        │    │
  │  └───────────────┘    └───────────────────┘    └───────────────────┘    │
  │                                                                          │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │           OperandCollector (4-bank Register File)                │   │
  │  │  Bank0: r0,r4,r8,r12   Bank1: r1,r5,r9,r13                      │   │
  │  │  Bank2: r2,r6,r10,r14  Bank3: r3,r7,r11,r15                     │   │
  │  │                                                                  │   │
  │  │  Bank conflict detection + non-blocking access (1 read/bank/cyc) │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  │                                                                          │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │  L1 Cache | Shared Memory | Global Memory | ALU | Vec4ALU       │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────────────────┘
```

Phase 7 是 toygpgpu 最大的架构变革：从"单周期 execute"演变为完整的 6 级流水线 (FETCH → DECODE → ISSUE → READ_OPERANDS → EXECUTE → WRITEBACK)。引入 I-Buffer (每 warp 2 entries) 实现取指与发射解耦，Operand Collector (4-bank) 实现寄存器 bank 冲突检测。这是对 GPGPU-Sim 逆序流水线 (inverse pipeline) 的核心建模。

## 2. Motivation / 设计动机

Phase 6 之前的"一步式"执行模型 (prepare → select → check → execute → reserve) 虽然功能正确，但隐藏了 GPU 流水线的关键微架构细节：

- **取指-发射解耦**：真实 GPU 的取指 (fetch) 和发射 (issue) 是独立的流水线阶段。I-Buffer 作为 FIFO 缓冲解耦两者，允许预取后续指令隐藏取指延迟。遇到分支时 I-Buffer 必须刷新 (flush)。
- **Banked Register File**：GPU 寄存器堆分为多个 bank (通常 32 banks for Fermi)，每 bank 每周期仅一个读端口。当指令的多个源寄存器在同一 bank 时发生 bank conflict，需要额外周期。
- **逆序流水线 (Inverse Pipeline)**：GPGPU-Sim 采用独特的逆序执行方式——在一个周期内从写回阶段向取指阶段反向推进。这是为了简化 writeback 和 scoreboard release 的顺序依赖。

GPGPU-Sim 对应实现：
- I-Buffer: `shader_core_ctx` 中的 `shader_core_ctx::fetch()` / `shader_core_ctx::decode()` / `shader_core_ctx::issue()`
- Operand Collector: `opndcoll_rfu_t` 类的 `allocate_cu()` / `dispatch()` 方法

## 3. Algorithm and Theory / 核心算法与理论

### 3.1 I-Buffer (Per-Warp 指令缓冲)

```
I-Buffer: 每个 warp 2 个 entry

数据结构 IBufferEntry:
    valid: bool     # fetch 已完成
    ready: bool     # decode + scoreboard 通过
    instruction_word: int
    pc: int

核心操作:
    write(instr_word, pc):
        # fetch 完成时调用
        if has_free():
            entry = find first invalid slot
            entry.valid = True, entry.ready = False
            entry.instr_word = instr_word, entry.pc = pc

    set_ready(pc):
        # decode 完成时调用 (Phase 7 中与 write 合并)
        entry = find valid entry with matching pc
        entry.ready = True

    consume() -> Optional[IBufferEntry]:
        # issue 时调用 — FIFO, 选最老的 ready 指令
        entry = first valid AND ready entry
        if entry: clear slot, return entry

    peek() -> Optional[IBufferEntry]:
        # 查看下一条就绪指令 (不消费, 用于重汇聚判断)
        return first valid AND ready entry

    flush():
        # 分支/跳转后调用
        clear all entries

    has_free() -> bool:  任何 invalid slot?
    has_ready() -> bool: 任何 valid AND ready?
```

### 3.2 Operand Collector (Banked Register File)

4-bank 寄存器堆：`reg_id % 4` 决定 bank 归属。

```
Bank 0: r0, r4, r8,  r12
Bank 1: r1, r5, r9,  r13
Bank 2: r2, r6, r10, r14
Bank 3: r3, r7, r11, r15

注意: r0 硬连线 0, 读 r0 不消耗 bank 端口

can_read_operands(rs1, rs2) -> (bool, reason):
    banks_needed = {}
    if rs1 != 0: banks_needed.add(bank_of(rs1))
    if rs2 != 0: banks_needed.add(bank_of(rs2))
    busy = [b for b in banks_needed if bank_busy[b]]
    if busy: return (False, "bank conflict: banks X busy")
    return (True, None)

reserve_banks(rs1, rs2):
    if rs1 != 0: bank_busy[bank_of(rs1)] = True
    if rs2 != 0:
        if bank_of(rs1) == bank_of(rs2): conflict_count++  # 同 bank!
        bank_busy[bank_of(rs2)] = True

release_banks():  # 每周期调用
    bank_busy = [False] * num_banks
```

### 3.3 Inverse Pipeline

每周期从后往前推进 6 个阶段：

```
step() 逆序执行:

 ===== 阶段 1: WRITEBACK (写回 + 资源释放) =====
    for each warp:
        scoreboard.advance()      # 递减 pending 计数器
        clear stall flag
    op_collector.release_banks()  # 释放 bank 端口

 ===== 阶段 2: ISSUE (发射) =====
    for _ in range(num_warps):
        warp = scheduler.select_warp()
        
        # 重汇聚检测 — 使用 I-Buffer.peek() 而非 warp.pc
        peek_entry = warp.ibuffer.peek()
        check_pc = peek_entry.pc if peek_entry else warp.pc
        if simt_stack.at_reconvergence(check_pc):
            handle_reconvergence → flush I-Buffer
        
        entry = warp.ibuffer.consume()
        if entry is None: continue  # I-Buffer 空
        
        decode(entry.instruction_word)
        
        # Scoreboard 检查
        if check_waw(rd) or check_raw(rs1, rs2):
            write entry back to I-Buffer → stall
        
        # Operand Collector bank 检查
        if not can_read_operands(rs1, rs2):
            write entry back to I-Buffer → bank conflict stall
        
        # Reserve banks + 执行
        reserve_banks(rs1, rs2)
        _execute_warp(...)
        sb.reserve(rd, latency)
        break  # 每周期最多发射 1 条

 ===== 阶段 3: FETCH + DECODE (取指 + 译码) =====
    for each warp (find first with free I-Buffer slot):
        if ibuffer.has_free():
            read instruction at warp.pc
            ibuffer.write(raw_word, warp.pc)
            ibuffer.set_ready(warp.pc)  # decode 完成
            warp.pc++
            break  # 每周期最多为 1 个 warp 取指
```

关键设计决策：使用 `IBuffer.peek()` 检查重汇聚，因为 FETCH 阶段已经递增了 `warp.pc`，可能导致跳过重汇聚点。

## 4. Architecture / 架构设计

### 4.1 Module Breakdown / 模块分解

| 文件 | 作用 |
|------|------|
| `ibuffer.py` | IBuffer + IBufferEntry 类 (2-entry per warp 指令缓冲) |
| `operand_collector.py` | OperandCollector 类 (4-bank register file, conflict detection) |
| `simt_core.py` | 完全重写 step() 实现逆序流水线，集成 I-Buffer + OperandCollector |
| `warp.py` | Warp 类新增 ibuffer 属性 + `_fetched_this_cycle` 标记 |
| `learning_console.py` | 流水线感知的交互式调试，新增 `ib`/`oc`/`sb` 命令 |
| `scheduler.py` | WarpScheduler (同 Phase 5/6, RR 策略) |
| `scoreboard.py` | Scoreboard (同 Phase 4) |
| `cache.py` | L1 数据缓存 (同 Phase 5) |
| `shared_memory.py` | 共享内存 (同 Phase 5) |
| `thread_block.py` | ThreadBlock (同 Phase 5) |
| `gpu_sim.py` | GPUSim (同 Phase 6) |

### 4.2 Key Data Structures / 关键数据结构

```python
@dataclass
class IBufferEntry:
    """单条指令槽"""
    valid: bool = False           # fetch 已写入
    ready: bool = False           # decode 完成, 可发射
    instruction_word: int = 0     # 32-bit 原始编码
    pc: int = 0                   # 指令地址

class IBuffer:
    """Per-Warp 指令缓冲 (2 entries)"""
    entries: list[IBufferEntry]   # 指令槽
    capacity: int = 2
    
    def has_free() -> bool
    def has_ready() -> bool
    def write(instr_word, pc) -> bool
    def set_ready(pc)
    def peek() -> Optional[IBufferEntry]   # 查看不消费
    def consume() -> Optional[IBufferEntry] # FIFO 取出
    def flush()                              # 清空

class OperandCollector:
    num_banks: int = 4
    bank_busy: list[bool]         # 每 bank 占用状态
    conflict_count: int           # bank conflict 累计
    total_reqs: int               # 总操作数请求
    
    @staticmethod bank_of(reg_id) -> int  # 寄存器 → bank 映射
    def can_read_operands(rs1, rs2) -> (bool, str)
    def reserve_banks(rs1, rs2)           # 占用 + conflict 检测
    def release_banks()
    def bank_conflict_rate() -> float
```

### 4.3 Module Interfaces / 模块接口

```
SIMTCore 构造函数:
    self.op_collector = OperandCollector(4)   # 新增

SIMTCore::step():
    # 1. WRITEBACK stage
    for w in warps: w.scoreboard.advance()
    self.op_collector.release_banks()
    
    # 2. ISSUE stage (最多 1 条)
    for warp in scheduler: 
        check reconv via ibuffer.peek()
        entry = warp.ibuffer.consume()
        if None: continue
        decode → scoreboard check → bank check → execute → reserve
        break
    
    # 3. FETCH stage (最多 1 条)
    for warp in warps:
        if ibuffer.has_free():
            ibuffer.write(program[warp.pc], warp.pc)
            ibuffer.set_ready(warp.pc)
            warp.pc++
            break

Warp 新增属性:
    self.ibuffer = IBuffer(capacity=2)
    self._fetched_this_cycle = False   # 防止重复取指

Warp::_execute_branch 修改:
    if divergence or taken branch:
        warp.ibuffer.flush()   # 分支后清空 I-Buffer
```

## 5. Functional Processing Flow / 功能处理流程

### 示例 1: I-Buffer 基本操作 (`01_ibuffer_basic.asm`)

```
程序: MOV r1,10; MOV r2,3; ADD r3,r1,r2; ST r3,[0]; HALT
warp_size=4, num_warps=1

Trace 输出:
──────────────────────────────────────────────────────────────────
FETCH:W0 PC=0 -> IB    | ISSUE:--    | SB:--    | IB:W0:_ _
FETCH:W0 PC=1 -> IB    | ISSUE:--    | SB:--    | IB:W0: MOV@PC0 _
FETCH:W0 PC=2 -> IB    | ISSUE:W0 PC=0:MOV rd=r1...  | SB: r1=1 | IB:W0: MOV@PC1 MOV@PC2
                         (W0: issue MOV r1,10, reserve r1=1 cycle)
FETCH:--                | ISSUE:W0 PC=1:MOV rd=r2...  | SB: r1=0 r2=1 | IB:W0: MOV@PC2 _
                         (W0: issue MOV r2,3, reserve r2=1 cycle)
FETCH:--                | ISSUE:W0 PC=2:ADD rd=r3...  | SB: -- | IB:W0:_ _
                         (W0: issue ADD r3,r1,r2=13)
FETCH:--                | ISSUE:W0 PC=3:ST ...        | IB:W0:_ _
                         (W0: ST r3,[0] → mem[0]=13)
FETCH:W0 PC=4 -> IB    | ISSUE:--                     | IB:W0:HALT@PC4 _
                         (W0: HALT → done)

说明:
- 前 2 周期只 FETCH 未 ISSUE: I-Buffer 预取 2 条指令后发射才开始
- I-Buffer 满 (2 entries) 时 FETCH 暂停, 等待 ISSUE 消费后继续预取
- 最后一个周期 FETCH HALT 后 ISSUE 消费, warp done
```

### 示例 2: Bank Conflict (`02_bank_conflict.asm`)

```
程序: MOV r1,10; MOV r5,20; ADD r2,r1,r5; ST r2,[0]; ...

寄存器 bank 映射:
  r1 → bank 1,  r5 → bank 1   (同 bank! 冲突!)
  r2 → bank 2,  r3 → bank 3   (不同 bank, 无冲突)

Trace 关键周期:
Cycle X: ISSUE MOV r1,10  → reserve bank 1
Cycle X+1: ISSUE MOV r5,20  → reserve bank 1 (释放后重新占用)
Cycle X+2: ISSUE ADD r2,r1,r5 → bank check: rs1=r1(bank1), rs2=r5(bank1)
           → BANK CONFLICT! 3"读取 r1 和 r5 在同一 bank, 放回 I-Buffer
Cycle X+3: ISSUE ADD r2,r1,r5 → bank check → OK
           → execute: r2 = 10+20 = 30

OpCollector stats:  reads=XX, conflicts=1, conflict rate=XX%
```

### 示例 3: 分支 + I-Buffer (`03_branch_ibuffer.asm`)

```
程序: MOV r1,10; MOV r2,5; BEQ r1,r2,skip; MOV r3,1; ST r3,[0]; JMP done; skip:...; done:...

BEQ  条件: r1(10) != r2(5) → NOT taken → fall through

关键行为:
- BEQ 执行前 I-Buffer 可能已预取 fall-through 路径指令 (MOV r3,1 等)
- BEQ NOT taken → fall through → I-Buffer 内容正确, 无需 flush
- 如果 BEQ taken → warp.pc 跳转 → ibuffer.flush() 清除预取内容
- flush 后从目标地址重新取指

对比 JMP (always taken):
- JMP 执行后 warp.pc 跳转 → ibuffer.flush()
- 下周期从目标地址取指
```

## 6. Comparison with Phase 6 / 与前一版本的对比

| Aspect | Phase 6 (Kernel) | Phase 7 (Pipeline) | Change |
|--------|------------------|---------------------|--------|
| 执行模型 | 单步 (fetch+decode+execute 在一个步骤) | 6 级逆序流水线 | 架构重构 |
| 取指 | 直接 `program[warp.pc]` | → I-Buffer 缓冲 + 预取 | 解耦 |
| 发射 | scheduler 选 warp 后立即执行 | I-Buffer.consume() 取指令 | 双缓冲 |
| 寄存器文件 | 统一访问，无冲突 | 4-bank，冲突检测+stall | bank 化 |
| 重汇聚检测 | 检查 warp.pc | 检查 ibuffer.peek().pc | 适配流水线 |
| 分支影响 | 仅改 pc 和 mask | 额外 ibuffer.flush() | 清 I-Buffer |
| 学习控制台 | `sb`/`cache`/`smem`/`perf` | 新增 `ib`/`oc` 命令 | 流水线调试 |
| trace 模式 | 指令级 + 寄存器 | 流水线级 (FETCH/ISSUE/SB/IB) | 更详细 |
| Warp 属性 | scoreboard, stack | + ibuffer, _fetched_this_cycle | 扩展 |
| 新增文件 | — | `ibuffer.py`, `operand_collector.py` | 2 个新文件 |
| 核心改动 | simt_core.py 小幅 | simt_core.py 完全重写 step() | 重构 |

## 7. Known Issues and Future Work / 遗留问题与后续工作

- **简化的取指仲裁**：每周期仅 1 个 warp 取指 1 条指令，未模拟真实 GPU 的 wider fetch (每周期 2+ warp)。
- **流水线深度固定**：所有指令的流水线阶段数相同（6 级），未区分不同执行单元的延迟差异。
- **无分支预测**：I-Buffer 在分支前不会预测目标，分支 miss 导致 flush + 重新取指的惩罚。真实 GPU 使用分支预测器或双路径执行。
- **寄存器文件 bank 数少**：4 banks 是教学简化。Fermi GPU 有 32 banks (每组 128 个寄存器)。
- **Operand Collector 简化**：真实 GPU 的 opndcol 具有多个 collector unit pool（每个执行单元管线专用），Phase 7 使用统一的单 collector。
- **无 operand 写**：仅模拟读端口冲突，未模拟写端口（writeback 阶段的写端口仲裁）。
- **I-Buffer 容量固定**：2 entries 不能动态扩展。真实 GPU 的 I-Buffer 每 warp 分配更多 slots。
- **未模拟 fetch stall**：Instruction cache miss、指令对齐限制等 fetch stall 原因未建模。

