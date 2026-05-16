"""
L2 Cache — Multi-Level Cache Hierarchy (Phase 19)
===================================================
对标 NVIDIA GPU 的 L2 Cache, 位于 L1 和 HBM 之间。
多 SM 共享, 大容量, 高命中率, 写回策略。

Cache Hierarchy / 缓存层次:
  L1 (per-SM, direct-mapped, 16 lines × 4 words, write-through)
    ↓ miss
  L2 (shared, set-associative, 256 lines × 4 words, write-back)
    ↓ miss
  HBM / Global Memory (1024 words)

Bandwidth Model / 带宽模型:
  L1 hit:     1 cycle,  BW = 1024 words/cycle
  L2 hit:    10 cycles, BW = 256 words/cycle
  HBM access: 100 cycles, BW = 32 words/cycle

参考: NVIDIA H100 L2 Cache (50MB, shared across GPCs)
      GPGPU-Sim memory_partition_unit
"""

from typing import Optional, List, Tuple


class L2CacheLine:
    """L2 缓存行。"""

    def __init__(self):
        self.valid = False
        self.dirty = False
        self.tag = 0
        self.data = [0, 0, 0, 0]  # 4 words per line

    def __repr__(self):
        return f"L2Line(valid={self.valid}, dirty={self.dirty}, tag=0x{self.tag:04X})"


class L2Cache:
    """L2 Cache — 多路组相联, 写回, 多 SM 共享。

    256 lines total, 4-way set associative → 64 sets.
    每行 4 words (16 bytes).

    Attributes:
        num_sets: 组数 (64)
        associativity: 相联度 (4)
        line_size: 行大小 (4 words)
        sets: [set][way] → L2CacheLine
        bandwidth: 每周期可读/写字数
        latency: 命中延迟 (cycles)
    """

    def __init__(self, total_lines: int = 256, associativity: int = 4,
                 line_words: int = 4):
        self.line_words = line_words
        self.associativity = associativity
        self.num_sets = total_lines // associativity
        self.total_lines = self.num_sets * associativity

        # Initialize sets
        self.sets: List[List[L2CacheLine]] = [
            [L2CacheLine() for _ in range(associativity)]
            for _ in range(self.num_sets)
        ]

        # LRU tracking: [set][way] → last access time
        self.lru: List[List[int]] = [
            [0] * associativity for _ in range(self.num_sets)
        ]
        self.clock = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.writebacks = 0
        self.evictions = 0

        # Bandwidth / latency modeling
        self.hit_latency = 10    # cycles for L2 hit
        self.miss_latency = 100  # cycles for HBM access
        self.bandwidth = 256     # words/cycle (L2 bandwidth)

    def _get_set_index(self, addr: int) -> int:
        """Get set index from address."""
        return (addr // self.line_words) % self.num_sets

    def _get_tag(self, addr: int) -> int:
        """Get tag from address."""
        return addr // (self.line_words * self.num_sets)

    def _get_line_offset(self, addr: int) -> int:
        """Get word offset within a cache line."""
        return addr % self.line_words

    def read(self, addr: int) -> Tuple[Optional[int], int]:
        """从 L2 读取地址 addr 的数据。

        Returns:
            (value, latency) — value 为 None 表示 miss
        """
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)
        offset = self._get_line_offset(addr)
        self.clock += 1

        # Search set for matching tag
        for way in range(self.associativity):
            line = self.sets[set_idx][way]
            if line.valid and line.tag == tag:
                self.hits += 1
                self.lru[set_idx][way] = self.clock
                return line.data[offset], self.hit_latency

        # Miss — need to fetch from HBM
        self.misses += 1
        return None, self.miss_latency

    def write(self, addr: int, value: int):
        """写入 L2 (来自 L1 write-through 或 store 操作)。

        写入分配: miss 时先加载行, 再写入。
        """
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)
        offset = self._get_line_offset(addr)
        self.clock += 1

        # Search for existing line
        for way in range(self.associativity):
            line = self.sets[set_idx][way]
            if line.valid and line.tag == tag:
                self.hits += 1
                self.lru[set_idx][way] = self.clock
                line.data[offset] = value
                line.dirty = True
                return

        # Miss — allocate
        self.misses += 1
        self._allocate(addr, value, set_idx, tag, offset)

    def _allocate(self, addr: int, value: int, set_idx: int,
                  tag: int, offset: int):
        """分配一个新的 L2 缓存行 (LRU 替换)。

        Args:
            addr: 完整地址 (用于从 HBM 加载整行)
            value: 要写入的值
            set_idx: 组索引
            tag: 标记
            offset: 行内偏移
        """
        # Find LRU way
        lru_way = min(range(self.associativity),
                      key=lambda w: self.lru[set_idx][w])
        old_line = self.sets[set_idx][lru_way]

        # Writeback if dirty
        if old_line.valid and old_line.dirty:
            self.writebacks += 1
            # In real HW: write old line to HBM

        if old_line.valid:
            self.evictions += 1

        # Fill new line (from HBM in real HW)
        base_addr = addr - offset
        new_line = self.sets[set_idx][lru_way]
        new_line.valid = True
        new_line.dirty = False
        new_line.tag = tag
        # In real HW: load from HBM. Here we initialize to 0.
        new_line.data = [0] * self.line_words
        new_line.data[offset] = value

        # Simulate: if this is a write, mark dirty
        if value != 0:
            new_line.dirty = True

        self.lru[set_idx][lru_way] = self.clock

    def fill_line(self, addr: int, data: List[int]):
        """从 HBM 填充一整行 (带对齐的 4-word 块)。

        Args:
            addr: 行对齐地址
            data: 4 个 word 的数据
        """
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)

        lru_way = min(range(self.associativity),
                      key=lambda w: self.lru[set_idx][w])

        old_line = self.sets[set_idx][lru_way]
        if old_line.valid and old_line.dirty:
            self.writebacks += 1
        if old_line.valid:
            self.evictions += 1

        new_line = self.sets[set_idx][lru_way]
        new_line.valid = True
        new_line.dirty = False
        new_line.tag = tag
        new_line.data = list(data[:self.line_words])
        self.lru[set_idx][lru_way] = self.clock

    def invalidate(self, addr: int):
        """失效指定地址的缓存行 (用于 coherence)。"""
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)
        for way in range(self.associativity):
            line = self.sets[set_idx][way]
            if line.valid and line.tag == tag:
                if line.dirty:
                    self.writebacks += 1
                line.valid = False
                line.dirty = False

    def hit_rate(self) -> float:
        """返回命中率。"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> str:
        """返回缓存统计信息。"""
        total = self.hits + self.misses
        return (f"L2Cache: {self.total_lines} lines ({self.num_sets}×"
                f"{self.associativity}-way), "
                f"hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate()*100:.1f}%, "
                f"writebacks={self.writebacks}, "
                f"evictions={self.evictions}, "
                f"latency: hit={self.hit_latency}c miss={self.miss_latency}c")


class BandwidthModel:
    """多级缓存带宽模型。

    为 perf_model 提供分层延迟数据:
      L1: 1 cycle
      L2: 10 cycles
      HBM: 100 cycles
    """

    def __init__(self):
        self.l1_latency = 1
        self.l2_latency = 10
        self.hbm_latency = 100
        self.l1_bandwidth = 1024  # words/cycle
        self.l2_bandwidth = 256
        self.hbm_bandwidth = 32

    def access_latency(self, cache_level: str) -> int:
        """返回指定缓存级别的访问延迟。"""
        return {
            "L1": self.l1_latency,
            "L2": self.l2_latency,
            "HBM": self.hbm_latency,
        }.get(cache_level, self.hbm_latency)

    def effective_bandwidth(self, l1_hit_rate: float,
                            l2_hit_rate: float) -> float:
        """根据缓存命中率计算有效带宽。

        Args:
            l1_hit_rate: L1 命中率
            l2_hit_rate: L2 命中率 (L1 miss 后)

        Returns:
            有效带宽 (words/cycle)
        """
        l1_hit = l1_hit_rate
        l1_miss_l2_hit = (1 - l1_hit_rate) * l2_hit_rate
        hbm = (1 - l1_hit_rate) * (1 - l2_hit_rate)

        total = (l1_hit * self.l1_bandwidth +
                 l1_miss_l2_hit * self.l2_bandwidth +
                 hbm * self.hbm_bandwidth)
        return total

    def report(self, l2: L2Cache) -> str:
        """生成缓存层次报告。"""
        eff_bw = self.effective_bandwidth(0.8, l2.hit_rate())
        return (
            f"Cache Hierarchy Report:\n"
            f"  L1: {self.l1_latency}c latency, {self.l1_bandwidth} w/c BW\n"
            f"  L2: {self.l2_latency}c latency, {self.l2_bandwidth} w/c BW "
            f"(hit_rate={l2.hit_rate()*100:.1f}%)\n"
            f"  HBM: {self.hbm_latency}c latency, {self.hbm_bandwidth} w/c BW\n"
            f"  Effective BW: {eff_bw:.1f} words/cycle"
        )
