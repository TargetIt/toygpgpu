"""
I-Buffer — Per-Warp 指令缓冲
===============================
对标 GPGPU-Sim 中 shader_core_ctx 的 I-Buffer。

fetch 和 issue 解耦：fetch 先取指→译码→写入 I-Buffer，
scheduler 从 I-Buffer 中选 ready 的 warp 发射指令。

GPGPU-Sim 的 I-Buffer: 静态分区, 每 warp 2 entries。
valid=已填充, ready=译码+scoreboard检查完成。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中的 fetch/decode/issue 实现
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class IBufferEntry:
    """I-Buffer 中的单条指令槽"""
    valid: bool = False       # fetch 已写入
    ready: bool = False       # decode + scoreboard check 通过
    instruction_word: int = 0  # 32-bit 原始指令字
    pc: int = 0                # 该指令的 PC


class IBuffer:
    """Per-Warp 指令缓冲 (I-Buffer)

    Attributes:
        entries: 指令槽列表 (默认 2 entries)
        next_write: 下一次写入的槽索引
    """

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self.entries = [IBufferEntry() for _ in range(capacity)]
        self.next_write = 0

    def has_free(self) -> bool:
        """是否有空闲槽 (可以 fetch)"""
        return any(not e.valid for e in self.entries)

    def has_ready(self) -> bool:
        """是否有就绪指令 (可以 issue)"""
        return any(e.valid and e.ready for e in self.entries)

    def write(self, instruction_word: int, pc: int):
        """写入一条新指令 (fetch 完成后调用)"""
        if not self.has_free():
            return False
        # 找第一个空闲槽
        for i in range(self.capacity):
            if not self.entries[i].valid:
                self.entries[i].valid = True
                self.entries[i].ready = False
                self.entries[i].instruction_word = instruction_word
                self.entries[i].pc = pc
                return True
        return False

    def set_ready(self, pc: int):
        """标记指指令为 ready (decode + scoreboard check 完成后调用)"""
        for e in self.entries:
            if e.valid and e.pc == pc:
                e.ready = True
                return

    def peek(self) -> Optional[IBufferEntry]:
        """查看下一条就绪指令 (不消费, 用于重汇聚判断等)"""
        best_idx = -1
        best_pc = float('inf')
        for i in range(self.capacity):
            if self.entries[i].valid and self.entries[i].ready:
                if self.entries[i].pc < best_pc:
                    best_pc = self.entries[i].pc
                    best_idx = i
        if best_idx >= 0:
            return self.entries[best_idx]
        return None

    def consume(self) -> Optional[IBufferEntry]:
        """取出一条就绪指令 (FIFO: 最小 PC = 最老指令)"""
        best_idx = -1
        best_pc = float('inf')
        for i in range(self.capacity):
            if self.entries[i].valid and self.entries[i].ready:
                if self.entries[i].pc < best_pc:
                    best_pc = self.entries[i].pc
                    best_idx = i
        if best_idx >= 0:
            entry = self.entries[best_idx]
            self.entries[best_idx] = IBufferEntry()
            return entry
        return None

    def flush(self):
        """清空所有槽位 (分支后调用)"""
        for i in range(self.capacity):
            self.entries[i] = IBufferEntry()

    def __repr__(self):
        slots = []
        for e in self.entries:
            if e.valid:
                s = f"v={e.valid},r={e.ready},pc={e.pc}"
            else:
                s = "empty"
            slots.append(s)
        return f"IBuffer({', '.join(slots)})"
