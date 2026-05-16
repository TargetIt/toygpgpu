"""
SharedMemory — 共享内存模块
=============================
对标 GPGPU-Sim 中 shader_core_ctx 的 shared memory。
每个 Thread Block (CTA) 内所有线程共享。
低延迟、高带宽的片上 SRAM。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中的 shared memory 实现
"""


class SharedMemory:
    """片上共享内存 (per Thread Block)

    对应 NVIDIA GPU 的 __shared__ 内存。
    通常 48KB (可配置)，这里简化为 256 words × 32-bit。

    Attributes:
        data: 字节级存储 (bytearray)
        size_words: 内存大小 (字)
    """

    def __init__(self, size_words: int = 256):
        self.size_words = size_words
        self.data = bytearray(size_words * 4)

    def read_word(self, addr: int) -> int:
        """读一个字 (32-bit)

        Args:
            addr: 字地址 (0 ~ size-1)
        """
        addr = addr % self.size_words
        offset = addr * 4
        return (self.data[offset] |
                (self.data[offset + 1] << 8) |
                (self.data[offset + 2] << 16) |
                (self.data[offset + 3] << 24))

    def write_word(self, addr: int, value: int):
        """写一个字"""
        addr = addr % self.size_words
        value = value & 0xFFFFFFFF
        offset = addr * 4
        self.data[offset] = value & 0xFF
        self.data[offset + 1] = (value >> 8) & 0xFF
        self.data[offset + 2] = (value >> 16) & 0xFF
        self.data[offset + 3] = (value >> 24) & 0xFF
