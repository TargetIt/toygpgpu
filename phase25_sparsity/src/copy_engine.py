"""
Copy Engine — Async DMA Copy Unit (Phase 17)
=============================================
对标 GPU 中的 DMA Copy Engine, 可与 SM 计算并行执行。

Copy Engine 独立于 SIMTCore 运行, 执行异步内存拷贝。
多个 stream 的 copy 操作可以 overlap。

参考: NVIDIA GPU copy engine (docs.nvidia.com/cuda)
      GPGPU-Sim memory_partition_unit DMA
"""

from typing import Optional, Tuple
from stream import Stream


class AsyncCopy:
    """一个异步拷贝操作。"""

    def __init__(self, src_addr: int, dst_addr: int, size: int,
                 stream_id: int = 0):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size
        self.stream_id = stream_id
        self.progress = 0  # words copied so far
        self.done = False


class CopyEngine:
    """异步 DMA 拷贝引擎。

    每个周期可以拷贝 bandwidth 个字。
    与 SIMTCore 并行工作。

    Attributes:
        bandwidth: 每周期拷贝字数
        pending: 待执行的拷贝队列
        completed: 已完成的拷贝数量
    """

    def __init__(self, bandwidth: int = 2):
        self.bandwidth = bandwidth  # words per cycle
        self.pending: list[AsyncCopy] = []
        self.memory = None  # reference to global memory
        self.completed = 0
        self.overlap_cycles = 0

    def submit(self, src: int, dst: int, size: int, stream_id: int = 0):
        """提交异步拷贝操作。"""
        self.pending.append(AsyncCopy(src, dst, size, stream_id))

    def step(self, memory) -> bool:
        """执行一个周期的拷贝工作。

        Args:
            memory: 全局内存对象 (用于读写)

        Returns:
            True if there is still work pending
        """
        self.memory = memory
        words_copied = 0

        for cp in list(self.pending):
            if cp.done:
                continue
            while cp.progress < cp.size and words_copied < self.bandwidth:
                src = cp.src_addr + cp.progress
                dst = cp.dst_addr + cp.progress
                if src < memory.size_words and dst < memory.size_words:
                    memory.write_word(dst, memory.read_word(src))
                cp.progress += 1
                words_copied += 1
            if cp.progress >= cp.size:
                cp.done = True
                self.completed += 1

        # Clean up done copies
        self.pending = [cp for cp in self.pending if not cp.done]

        if words_copied > 0 and len(self.pending) > 0:
            self.overlap_cycles += 1

        return len(self.pending) > 0

    def has_pending(self) -> bool:
        return len(self.pending) > 0

    def stats(self) -> str:
        return (f"CopyEngine: completed={self.completed}, "
                f"pending={len(self.pending)}, "
                f"overlap_cycles={self.overlap_cycles}, "
                f"bandwidth={self.bandwidth} words/cycle")
