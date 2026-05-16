"""
L1Cache — 简化版 L1 数据缓存
==============================
对标 GPGPU-Sim 中 gpu-cache 的 L1 缓存模型。

Phase 5 实现直接映射 (direct-mapped) L1 Cache。
16 lines × 4 words/line (64 words = 256 bytes)。

参考: GPGPU-Sim gpgpu-sim/gpu-cache.cc
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheLine:
    valid: bool = False
    tag: int = 0         # 块地址 (addr // line_size_words)
    dirty: bool = False
    data: list = None    # list of 32-bit words

    def __post_init__(self):
        if self.data is None:
            self.data = [0] * 4


class L1Cache:
    """直接映射 L1 数据缓存

    Attributes:
        num_lines: 缓存行数 (默认 16)
        line_size: 每行 word 数 (默认 4)
        lines: 缓存行列表
        hits: 命中计数
        misses: 未命中计数
    """

    def __init__(self, num_lines: int = 16, line_size: int = 4):
        self.num_lines = num_lines
        self.line_size = line_size
        self.lines = [CacheLine() for _ in range(num_lines)]
        self.hits = 0
        self.misses = 0

    def _index(self, addr: int) -> int:
        return (addr // self.line_size) % self.num_lines

    def _tag(self, addr: int) -> int:
        return addr // (self.num_lines * self.line_size)

    def read(self, addr: int) -> Optional[int]:
        """读一个 word。未命中返回 None（调用方从下级存储取）。"""
        idx = self._index(addr)
        tag = self._tag(addr)
        line = self.lines[idx]
        offset = addr % self.line_size

        if line.valid and line.tag == tag:
            self.hits += 1
            return line.data[offset]
        else:
            self.misses += 1
            return None

    def write(self, addr: int, value: int):
        """写一个 word (write-through, no-write-allocate 简化)"""
        idx = self._index(addr)
        tag = self._tag(addr)
        line = self.lines[idx]
        offset = addr % self.line_size

        if line.valid and line.tag == tag:
            self.hits += 1
            line.data[offset] = value & 0xFFFFFFFF
            line.dirty = True
        else:
            self.misses += 1
            # No-write-allocate: don't load the line, just pass through

    def fill_line(self, addr: int, data: list):
        """填充一个缓存行（miss 后从下级存储加载）"""
        idx = self._index(addr)
        tag = self._tag(addr)
        line = self.lines[idx]
        line.valid = True
        line.tag = tag
        line.dirty = False
        for i in range(min(self.line_size, len(data))):
            line.data[i] = data[i]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> str:
        total = self.hits + self.misses
        return (f"L1Cache: hits={self.hits}, misses={self.misses}, "
                f"total={total}, hit_rate={self.hit_rate:.1%}")
