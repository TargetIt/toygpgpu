"""
ThreadBlock (CTA) — 协作线程阵列
==================================
对标 GPGPU-Sim 的 Cooperative Thread Array (CTA / Thread Block)。

一个 Thread Block 包含多个 warp，共享同一个 Shared Memory。

参考: GPGPU-Sim 的 thread block / CTA 概念
"""

from shared_memory import SharedMemory


class ThreadBlock:
    """协作线程阵列 (CTA)

    对应 CUDA 编程模型中的 thread block。
    多个 warp 组成一个 block，共享 shared memory。

    Attributes:
        block_id: block 编号
        warps: 该 block 内的 warp 列表
        shared_memory: 共享内存 (所有 warp 共享)
    """

    def __init__(self, block_id: int, warps: list, shared_mem_size: int = 256):
        self.block_id = block_id
        self.warps = warps
        self.shared_memory = SharedMemory(shared_mem_size)
