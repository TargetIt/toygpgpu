# Phase 3: SIMT Stack — 设计文档

## 1. SIMT Stack 原理

GPGPU-Sim 使用 **IPDOM（Immediate Post-Dominator）栈** 处理分支发散。Phase 3 实现简化版。

### 示例：if/else 发散

```asm
TID r1
MOV r2, 2
DIV r3, r1, r2    ; r3 = tid / 2
MUL r4, r3, r2    ; r4 = (tid/2) * 2

MOV r5, 0
BEQ r4, r5, else_path  ; 偶数线程: r4==0 → 跳转

; then path (奇数线程)
MOV r6, 1
ST r6, [100]
JMP endif

else_path:          ; 偶数线程
MOV r6, 2
ST r6, [100]

endif:              ; 重汇聚点
; 所有线程继续
HALT
```

### 执行时间线

```
PC  active_mask    操作
0   11111111       TID r1
1   11111111       MOV r2, 2
2   11111111       DIV r3, r1, r2
3   11111111       MUL r4, r3, r2
4   11111111       MOV r5, 0
5   11111111       BEQ → 偶数取跳转,奇数走fallthrough
    → push(reconv=endif, orig=11111111, taken=01010101(偶数), fall=PC+1=6)
    → active=01010101, PC=else_path

... (执行偶数路径)
n   01010101       JMP endif → active=01010101, PC=endif

endif:             → PC == reconv_pc! pop!
    → push(reconv=endif, orig=01010101(剩下未执行的奇数), taken=奇数, fall=6)
    → active=10101010(奇数), PC=6

... (执行奇数路径)
m   10101010       到达 endif → PC == reconv_pc! pop!
    → 恢复 active=11111111, PC=endif

endif:
    11111111       HALT
```

## 2. SIMT Stack 数据结构

```python
@dataclass
class SIMTStackEntry:
    reconv_pc: int       # 重汇聚 PC
    orig_mask: int       # 发散前 active_mask
    taken_mask: int      # 跳转路径的线程掩码
    fallthrough_pc: int  # 非跳转路径的起始 PC

class SIMTStack:
    entries: list[SIMTStackEntry]
    
    def push(entry)        # 压栈
    def pop() -> entry     # 弹栈
    def top() -> entry     # 查看栈顶
    def at_reconv(pc) -> bool  # PC 是否在重汇聚点
```

## 3. 分支指令编码

复用 R-type 格式：
- `JMP label` → opcode=0x08, rd=0, rs1=0, rs2=0, imm=target_offset
- `BEQ rs1, rs2, label` → opcode=0x09, rd=0, rs1, rs2, imm=target_offset
- `BNE rs1, rs2, label` → opcode=0x0A, rd=0, rs1, rs2, imm=target_offset

target_offset = label_PC - current_PC（相对偏移）

## 4. 汇编器 Label 支持

两遍汇编：
1. **Pass 1**: 扫描源程序，记录所有 label 的 PC
2. **Pass 2**: 用 label→PC 映射翻译指令

```
Source → [Pass1: collect labels] → [Pass2: emit code with resolved labels]
```

## 5. SIMTCore 修改

`_execute_warp` 中新增分支处理：

```python
if op in (OP_JMP, OP_BEQ, OP_BNE):
    taken_mask = compute_taken_mask(warp, instr)  # 哪些线程跳转
    not_taken = warp.active_mask & ~taken_mask
    
    if taken_mask and not_taken:
        # 发散！
        warp.simt_stack.push(SIMTStackEntry(
            reconv_pc=warp.pc,  # 下一条指令 = 重汇聚点
            orig_mask=warp.active_mask,
            taken_mask=taken_mask,
            fallthrough_pc=warp.pc  # 非跳转路径从这里开始
        ))
        warp.active_mask = taken_mask
        warp.pc = target_pc
    elif taken_mask:
        # 全跳转 → 无发散
        warp.pc = target_pc
    else:
        # 全不跳转 → 执行下一条
        pass
```

重汇聚检测在 `step()` 顶部：
```python
if warp.simt_stack and warp.simt_stack.top().reconv_pc == warp.pc:
    entry = warp.simt_stack.pop()
    remaining = entry.orig_mask & ~entry.taken_mask
    if remaining:
        warp.simt_stack.push(SIMTStackEntry(
            reconv_pc=entry.reconv_pc,
            orig_mask=remaining,
            taken_mask=remaining,
            fallthrough_pc=0
        ))
        warp.active_mask = remaining
        warp.pc = entry.fallthrough_pc
    else:
        warp.active_mask = entry.orig_mask
```

## 6. 与 GPGPU-Sim 对照

| GPGPU-Sim | Phase 3 |
|-----------|---------|
| `simt_stack` (C++) | `SIMTStack` (Python) |
| `shd_warp_t::m_shader_ips` | `warp.simt_stack` |
| IPDOM 分析 (编译时) | 简化为：标签处 = 可能的重汇聚点 |
| `simt_stack::push/pop` | 同名方法 |
