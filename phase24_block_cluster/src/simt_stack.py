"""
SIMTStack — SIMT 分支发散栈
=============================
对标 GPGPU-Sim 中 simt_stack 类（基于 IPDOM 的后支配栈）。

GPGPU-Sim 的 simt_stack 使用编译时分析得到的 IPDOM 信息
来决定重汇聚点。Phase 3 简化为：紧接分支后的下一条指令 = 重汇聚点。

原理：
  遇到分支 → push(重汇聚点, 原始mask, 跳转mask)
  到达重汇聚点 → 切换到未执行的路径
  所有路径执行完 → pop, 恢复原始mask

参考: GPGPU-Sim gpgpu-sim/stack.cc 中 simt_stack 实现
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SIMTStackEntry:
    """SIMT 栈条目

    对应 GPGPU-Sim 中 simt_stack_entry 结构体。

    Attributes:
        reconv_pc: 重汇聚 PC 地址（发散前 PC+1）
        orig_mask: 发散前的 active_mask
        taken_mask: 已执行路径的线程掩码
        fallthrough_pc: 未执行路径的起始 PC
    """
    reconv_pc: int
    orig_mask: int
    taken_mask: int
    fallthrough_pc: int


class SIMTStack:
    """SIMT 发散/重汇聚栈

    每个 Warp 一个独立的 SIMT Stack。

    GPGPU-Sim 中 simt_stack 包含多个条目（支持嵌套分支）。
    Phase 3 同样支持嵌套。
    """

    def __init__(self):
        self.entries: List[SIMTStackEntry] = []

    def push(self, entry: SIMTStackEntry):
        self.entries.append(entry)

    def pop(self) -> Optional[SIMTStackEntry]:
        if self.entries:
            return self.entries.pop()
        return None

    def top(self) -> Optional[SIMTStackEntry]:
        if self.entries:
            return self.entries[-1]
        return None

    def at_reconvergence(self, pc: int) -> bool:
        """检查当前 PC 是否在栈顶的重汇聚点"""
        top = self.top()
        return top is not None and top.reconv_pc == pc

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0

    def __len__(self) -> int:
        return len(self.entries)
