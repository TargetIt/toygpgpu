"""
Memory — 内存模块
==================
对标 GPGPU-Sim 中 memory_partition_unit 管理的 GDDR DRAM。

Phase 0: 平坦内存模型，256 words × 32-bit (1KB)，按字寻址。

参考: GPGPU-Sim gpgpu-sim/memory_partition_unit 的 L2/DRAM 接口
      TinyGPU 的 global memory 实现 (bytearray)
"""


class Memory:
    """256 × 32-bit 平坦内存

    对应 GPGPU-Sim 中的 GDDR DRAM 模型。
    简化版本: 单层、单 bank、零延迟。

    Attributes:
        size_words: 内存大小（字）
        data: 字节级存储 (bytearray)
    """

    def __init__(self, size_words: int = 256):
        self.size_words = size_words
        self.data = bytearray(size_words * 4)  # 每字 4 字节

    def read_word(self, addr: int) -> int:
        """读一个字 (32-bit)

        对应 GPGPU-Sim 中 DRAM 的读操作。

        Args:
            addr: 字地址 (0-255)

        Returns:
            32-bit 值 (小端序)
        """
        if addr < 0 or addr >= self.size_words:
            raise IndexError(f"Memory address out of range: {addr} (0-{self.size_words - 1})")

        offset = addr * 4
        # 小端序: byte[0] 是最低字节
        val = (self.data[offset] |
               (self.data[offset + 1] << 8) |
               (self.data[offset + 2] << 16) |
               (self.data[offset + 3] << 24))
        return val

    def write_word(self, addr: int, value: int):
        """写一个字 (32-bit)

        Args:
            addr: 字地址 (0-255)
            value: 32-bit 值（截断到 32-bit，小端序写入）
        """
        if addr < 0 or addr >= self.size_words:
            raise IndexError(f"Memory address out of range: {addr} (0-{self.size_words - 1})")

        value = value & 0xFFFFFFFF
        offset = addr * 4
        self.data[offset] = value & 0xFF
        self.data[offset + 1] = (value >> 8) & 0xFF
        self.data[offset + 2] = (value >> 16) & 0xFF
        self.data[offset + 3] = (value >> 24) & 0xFF

    def dump(self, start: int = 0, count: int = 16) -> str:
        """打印内存区域（调试用）"""
        lines = []
        for i in range(start, min(start + count, self.size_words)):
            val = self.read_word(i)
            if val != 0:  # 只显示非零值
                lines.append(f"  mem[{i:3d}]: 0x{val:08X} ({val})")
        return "\n".join(lines) if lines else "  (all zero)"
