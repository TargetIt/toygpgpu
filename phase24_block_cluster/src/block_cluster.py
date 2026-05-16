"""
Thread Block Cluster — Distributed Shared Memory (Phase 24)
=============================================================
对标 NVIDIA Hopper 架构的 Thread Block Cluster 特性。

核心概念:
  - ThreadBlockCluster: 多个 CTA block 组成集群, 共享分布式共享内存
  - DSM (Distributed Shared Memory): 跨集群中所有 block 的地址空间
    每个 block 贡献一部分 shared memory, 所有 block 可以通过 DSM 访问
  - ClusterBarrier: 跨集群中所有 block 的同步
  - Cross-block reduction pattern: 跨 block 归约操作

Hopper 的 Cluster 特性:
  - 最多 8 个 block 组成一个 cluster
  - cluster.sync() 同步 cluster 内所有线程
  - 分布式共享内存允许 cluster 内 block 互相访问 shared memory
  - cuda:block_dim() 等内置函数获取 cluster 配置

参考:
  - NVIDIA Hopper Architecture whitepaper (Thread Block Cluster)
  - CUDA Programming Guide: Thread Block Clusters
  - GPGPU-Sim's cluster model
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClusterBarrier:
    """集群 Barrier — 同步 cluster 中所有 block 的所有线程。

    对标 CUDA __syncthreads_cluster() 和 cluster.sync()。

    Attributes:
        num_blocks: cluster 中的 block 数量
        arrived_blocks: 已到达 barrier 的 block 计数 (按 phase)
        phase: 当前 barrier phase (用于 double-buffering)
    """

    num_blocks: int = 1
    arrived_blocks: int = 0
    phase: int = 0

    def arrive(self, block_id: int) -> bool:
        """一个 block 到达 barrier。

        Returns:
            所有 block 都到达时返回 True
        """
        self.arrived_blocks += 1
        if self.arrived_blocks >= self.num_blocks:
            self.arrived_blocks = 0
            self.phase += 1
            return True
        return False

    def wait(self, block_id: int) -> bool:
        """等待 barrier (block 到达并检查是否全部到达)。"""
        return self.arrive(block_id)

    def reset(self):
        """重置 barrier。"""
        self.arrived_blocks = 0
        self.phase += 1


class DSMBlockSlice:
    """单个 block 贡献的 DSM 切片 — 这个 block 拥有的 shared memory 部分。

    Attributes:
        block_id: 该 block 在 cluster 中的 ID
        data: 该 block 贡献的共享内存数据 (以 word 为单位)
        size_words: 共享内存大小 (words)
    """

    def __init__(self, block_id: int, size_words: int = 256):
        self.block_id = block_id
        self.size_words = size_words
        self.data = [0] * size_words

    def read(self, offset: int) -> int:
        """从该切片的 offset 处读取一个 word。"""
        if 0 <= offset < self.size_words:
            return self.data[offset]
        return 0

    def write(self, offset: int, value: int):
        """向该切片的 offset 处写入一个 word。"""
        if 0 <= offset < self.size_words:
            self.data[offset] = value


class DistributedSharedMemory:
    """分布式共享内存 — 由 cluster 中所有 block 贡献的地址空间。

    地址映射:
      [block_id * block_size : (block_id+1) * block_size) 表示 block 的切片

    Attributes:
        num_blocks: cluster 中的 block 数量
        block_size: 每个 block 贡献的共享内存大小 (words)
        slices: {block_id: DSMBlockSlice}
    """

    def __init__(self, num_blocks: int = 2, block_size: int = 256):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.slices: Dict[int, DSMBlockSlice] = {
            bid: DSMBlockSlice(bid, block_size) for bid in range(num_blocks)
        }

    def dsm_read(self, global_offset: int) -> int:
        """从 DSM 地址空间读取一个 word。

        Args:
            global_offset: DSM 地址空间中的全局偏移

        Returns:
            word 值
        """
        block_id = global_offset // self.block_size
        local_offset = global_offset % self.block_size
        if block_id in self.slices:
            return self.slices[block_id].read(local_offset)
        return 0

    def dsm_write(self, global_offset: int, value: int):
        """向 DSM 地址空间写入一个 word。

        Args:
            global_offset: DSM 地址空间中的全局偏移
            value: 要写入的值
        """
        block_id = global_offset // self.block_size
        local_offset = global_offset % self.block_size
        if block_id in self.slices:
            self.slices[block_id].write(local_offset, value)

    def dsm_size(self) -> int:
        """DSM 地址空间总大小 (words)。"""
        return self.num_blocks * self.block_size


class ThreadBlockCluster:
    """线程块集群 — 管理 cluster 中的多个 block 和 DSM。

    Attributes:
        num_blocks: cluster 中的 block 数量
        dsm: 分布式共享内存
        barrier: 集群 barrier
        block_data: 每个 block 的本地数据 {block_id: data}
    """

    def __init__(self, num_blocks: int = 2, dsm_block_size: int = 256):
        if num_blocks < 1 or num_blocks > 8:
            raise ValueError("Cluster supports 1-8 blocks")
        self.num_blocks = num_blocks
        self.dsm = DistributedSharedMemory(num_blocks, dsm_block_size)
        self.barrier = ClusterBarrier(num_blocks)
        self.block_data: Dict[int, List[int]] = {
            bid: [] for bid in range(num_blocks)
        }

    def cluster_sync(self, block_id: int) -> bool:
        """跨 cluster 同步 (cluster.sync())。

        Args:
            block_id: 调用线程的 block ID

        Returns:
            True 如果所有 block 都到达 barrier
        """
        return self.barrier.wait(block_id)

    def dsm_load(self, block_id: int, local_smem_offset: int,
                 dsm_global_offset: int, size: int = 1) -> List[int]:
        """从 DSM 加载数据到本 block 的共享内存。

        Args:
            block_id: 本 block ID
            local_smem_offset: 本地共享内存的目标偏移 (模拟)
            dsm_global_offset: DSM 地址空间中的源偏移
            size: 要加载的 word 数量

        Returns:
            加载的数据
        """
        data = []
        for i in range(size):
            val = self.dsm.dsm_read(dsm_global_offset + i)
            data.append(val)
        return data

    def dsm_store(self, block_id: int, dsm_global_offset: int,
                  data: List[int]):
        """从本 block 存储数据到 DSM 地址空间。

        Args:
            block_id: 本 block ID
            dsm_global_offset: DSM 地址空间中的目标偏移
            data: 要存储的数据
        """
        for i, val in enumerate(data):
            self.dsm.dsm_write(dsm_global_offset + i, val)

    def cross_block_reduce(self, op: str = "sum") -> List[int]:
        """跨所有 block 执行归约操作。

        每个 block 贡献数据, 归约结果对每个 block 可见。

        Args:
            op: 归约操作 ("sum", "max", "min")

        Returns:
            每个 block 的归约结果列表
        """
        results = []
        for bid in range(self.num_blocks):
            block_vals = self.block_data.get(bid, [])
            if not block_vals:
                results.append(0)
                continue

            if op == "sum":
                result = sum(block_vals)
            elif op == "max":
                result = max(block_vals)
            elif op == "min":
                result = min(block_vals)
            else:
                result = sum(block_vals)
            results.append(result)

        # Cross-block reduction: write results to DSM
        for bid, val in enumerate(results):
            offset = bid * 4  # Each block writes to a reserved area
            self.dsm.dsm_write(offset, val)

        return results

    def set_block_data(self, block_id: int, data: List[int]):
        """设置一个 block 的数据。"""
        if block_id in self.block_data:
            self.block_data[block_id] = list(data)

    def get_block_data(self, block_id: int) -> List[int]:
        """获取一个 block 的数据。"""
        return list(self.block_data.get(block_id, []))

    def __repr__(self):
        return (f"ThreadBlockCluster(blocks={self.num_blocks}, "
                f"dsm_size={self.dsm.dsm_size()}, "
                f"phase={self.barrier.phase})")
