"""
TMA — Tensor Memory Accelerator (Phase 20)
============================================
对标 NVIDIA Hopper 架构的 TMA (Tensor Memory Accelerator)。

TMA 是 Hopper 架构最大的创新之一。它是一个专用的硬件单元,
可以在后台异步计算张量元素的地址、加载/存储数据,
完全从 SM 指令流水线中卸掉地址计算和数据搬运的开销。

核心功能:
  1. 自动地址生成: 给定 base, shape, strides → 计算每个元素的地址
  2. 边界处理: 自动处理 tile 边界超出张量范围的情况 (zero-pad)
  3. 异步加载: 通过 CopyEngine 异步执行, 与 SM 计算重叠
  4. 多维张量: 支持 1D-3D 张量的 tile 加载

参考: NVIDIA H100 TMA (docs.nvidia.com/cuda/hopper-tuning-guide)
      CUTLASS 3.x TMA abstraction
"""

from typing import List, Tuple, Optional, Dict


class TensorDescriptor:
    """张量描述符 — 描述一个多维张量的形状和布局。

    对标 CUDA CUTENSOR_MAP / cuTensorMap。

    Attributes:
        shape: 各维度大小 [D0, D1, D2]
        strides: 各维度步长 (行优先: strides[-1]=1)
        base_addr: 全局内存基地址
        element_size: 每个元素的字节数
    """

    def __init__(self, shape: List[int], strides: List[int] = None,
                 base_addr: int = 0, element_size: int = 4):
        self.shape = list(shape)
        self.ndim = len(shape)

        # Compute row-major strides if not provided
        if strides:
            self.strides = list(strides)
        else:
            self.strides = [1]
            for s in reversed(shape[1:]):
                self.strides.insert(0, self.strides[0] * s)

        self.base_addr = base_addr
        self.element_size = element_size

    def linear_addr(self, indices: List[int]) -> int:
        """将多维索引转换为线性地址。"""
        addr = self.base_addr
        for i, stride in enumerate(self.strides):
            idx = indices[i] if i < len(indices) else 0
            addr += idx * stride * self.element_size // 4  # word addressing
        return addr

    def tile_bounds(self, start: List[int], size: List[int]) -> Dict:
        """计算 tile 的实际有效范围 (处理越界)。

        Returns:
            {'valid_ranges': [(start, end), ...], 'total_elements': int}
        """
        ranges = []
        total = 1
        for d in range(self.ndim):
            s = start[d] if d < len(start) else 0
            sz = size[d] if d < len(size) else 1
            actual_start = max(0, s)
            actual_end = min(self.shape[d], s + sz)
            ranges.append((actual_start, actual_end))
            total *= max(0, actual_end - actual_start)
        return {'valid_ranges': ranges, 'total_elements': total}


class TMAEngine:
    """TMA 引擎 — 硬件张量地址生成和异步加载。

    Attributes:
        descriptors: 已注册的张量描述符 (类似 cuTensorMap)
        copy_engine: 关联的 CopyEngine (用于异步 DMA)
        stats: 统计信息
    """

    def __init__(self, copy_engine=None):
        self.descriptors: Dict[int, TensorDescriptor] = {}
        self.copy_engine = copy_engine
        self.stats = {"tma_loads": 0, "tma_stores": 0,
                      "addresses_generated": 0, "boundary_elements": 0}

    def register_descriptor(self, desc_id: int, desc: TensorDescriptor):
        """注册张量描述符 (模拟 cuTensorMap 创建)。"""
        self.descriptors[desc_id] = desc

    def compute_addresses(self, desc_id: int, start: List[int],
                          size: List[int]) -> List[int]:
        """计算 tile 中所有元素的线性地址。

        Args:
            desc_id: 张量描述符 ID
            start: tile 起始坐标 [d0_start, d1_start, ...]
            size: tile 大小 [d0_size, d1_size, ...]

        Returns:
            地址列表 (线性 word 地址)
        """
        desc = self.descriptors[desc_id]
        bounds = desc.tile_bounds(start, size)
        addrs = []

        # Iterate over all positions in the tile
        ndim = desc.ndim
        for flat_idx in range(self._prod(size)):
            coords = self._flat_to_coord(flat_idx, size)
            global_coords = [start[d] + coords[d] for d in range(ndim)]

            # Boundary check: out-of-bounds → -1 (skipped/zero-padded)
            in_bounds = all(
                0 <= global_coords[d] < desc.shape[d]
                for d in range(ndim)
            )
            if in_bounds:
                addrs.append(desc.linear_addr(global_coords))
            else:
                addrs.append(-1)  # out-of-bounds marker
                self.stats["boundary_elements"] += 1

        self.stats["addresses_generated"] += len(addrs)
        return addrs

    def tma_load(self, desc_id: int, start: List[int], size: List[int],
                 smem_offset: int, memory,
                 shared_memory=None) -> int:
        """TMA 异步加载: 将 tile 从 global memory 加载到 shared memory。

        这是硬件加速的: 地址计算和加载在一个硬件操作中完成。

        Args:
            desc_id: 张量描述符 ID
            start: tile 起始坐标
            size: tile 大小
            smem_offset: shared memory 目标偏移
            memory: 全局内存对象
            shared_memory: shared memory 对象

        Returns:
            加载的元素数量
        """
        addrs = self.compute_addresses(desc_id, start, size)
        count = 0

        for i, addr in enumerate(addrs):
            if addr >= 0 and shared_memory:
                smem_addr = smem_offset + i
                if smem_addr < shared_memory.size_words:
                    if addr < memory.size_words:
                        shared_memory.write_word(smem_addr,
                                                memory.read_word(addr))
                    count += 1
            elif addr < 0 and shared_memory:
                # Out-of-bounds: zero-pad
                smem_addr = smem_offset + i
                if smem_addr < shared_memory.size_words:
                    shared_memory.write_word(smem_addr, 0)
                count += 1

        self.stats["tma_loads"] += 1
        return count

    def tma_store(self, desc_id: int, start: List[int], size: List[int],
                  smem_offset: int, memory,
                  shared_memory=None) -> int:
        """TMA 异步存储: 将 shared memory 的 tile 写回 global memory。"""
        addrs = self.compute_addresses(desc_id, start, size)
        count = 0

        for i, addr in enumerate(addrs):
            if addr >= 0 and shared_memory:
                smem_addr = smem_offset + i
                if smem_addr < shared_memory.size_words and addr < memory.size_words:
                    memory.write_word(addr, shared_memory.read_word(smem_addr))
                    count += 1

        self.stats["tma_stores"] += 1
        return count

    def stats_report(self) -> str:
        """TMA 统计报告。"""
        return (f"TMA Engine: loads={self.stats['tma_loads']}, "
                f"stores={self.stats['tma_stores']}, "
                f"addrs_gen={self.stats['addresses_generated']}, "
                f"boundary={self.stats['boundary_elements']}")

    @staticmethod
    def _prod(lst: List[int]) -> int:
        p = 1
        for x in lst:
            p *= x
        return p

    @staticmethod
    def _flat_to_coord(flat_idx: int, shape: List[int]) -> List[int]:
        coords = []
        stride = 1
        for s in reversed(shape):
            coords.insert(0, (flat_idx // stride) % s)
            stride *= s
        return coords
