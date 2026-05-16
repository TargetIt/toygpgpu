"""
Structured Sparsity — 2:4 Sparse Tensor Operations (Phase 25)
===============================================================
对标 NVIDIA Ampere+ 架构的 2:4 结构化稀疏特性。

核心概念:
  - 2:4 Structured Sparsity: 在每组 4 个元素中, 恰好有 2 个非零
  - Sparse MMA: 硬件跳过零值乘法, 提供 2x 矩阵乘法的性能提升
  - SparsityMask: 编码 4 个元素中哪 2 个非零的掩码 (2-bit per group)
  - Dense-to-sparse 转换和稀疏矩阵打包

NVIDIA 稀疏张量核心:
  - 在 Ampere (SM 8.0) 引入
  - 对 2:4 结构化稀疏矩阵的 MMA 吞吐量翻倍
  - 压缩格式: 每 4 个元素只存储 2 个非零 + 2-bit 掩码
  - 非结构化稀疏需要软件处理, 2:4 是硬件直接支持的

参考:
  - NVIDIA Ampere Architecture whitepaper (Fine-Grained Structured Sparsity)
  - "Accelerating Matrix Multiplication with 2:4 Structured Sparsity"
  - CUDA Programming Guide: Sparse MMA
"""

from dataclasses import dataclass
from typing import List, Tuple


class SparsityMask:
    """2:4 稀疏掩码 — 编码每组 4 个元素中的非零位置。

    在 2:4 模式中, 每 4 个元素恰有 2 个非零。
    掩码用 2-bit 编码, 指示 4 个位置中哪 2 个是有效的。
    值 0-5 对应 6 种可能的 2:4 模式 (C(4,2)=6)。

    Patterns:
      0: [X, X, ., .]  cols 0,1
      1: [X, ., X, .]  cols 0,2
      2: [X, ., ., X]  cols 0,3
      3: [., X, X, .]  cols 1,2
      4: [., X, ., X]  cols 1,3
      5: [., ., X, X]  cols 2,3
    """

    # 掩码模式 → 有效列索引
    PATTERNS = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (1, 2),
        4: (1, 3),
        5: (2, 3),
    }

    # 有效列索引 → 掩码模式 (逆映射)
    COLS_TO_PATTERN = {
        (0, 1): 0,
        (0, 2): 1,
        (0, 3): 2,
        (1, 2): 3,
        (1, 3): 4,
        (2, 3): 5,
    }

    @classmethod
    def encode(cls, cols: Tuple[int, int]) -> int:
        """编码列索引对为 2-bit 掩码。"""
        return cls.COLS_TO_PATTERN.get(tuple(sorted(cols)), 0)

    @classmethod
    def decode(cls, mask: int) -> Tuple[int, int]:
        """解码 2-bit 掩码为有效列索引。"""
        return cls.PATTERNS.get(mask & 0xF, (0, 1))

    @classmethod
    def is_valid_2to4(cls, values: List[float]) -> bool:
        """检查 4 个元素是否符合 2:4 稀疏 (恰有 2 个非零)。"""
        if len(values) != 4:
            return False
        non_zero = sum(1 for v in values if v != 0)
        return non_zero == 2

    @classmethod
    def find_pattern(cls, values: List[float]) -> int:
        """从 4 个元素中找到 2:4 模式编码。"""
        non_zero_cols = tuple(i for i, v in enumerate(values) if v != 0)
        if len(non_zero_cols) != 2:
            return 0
        return cls.encode(non_zero_cols)


def dense_to_sparse_2to4(dense: List[float]) -> Tuple[List[float], List[int]]:
    """将稠密向量/行转换为 2:4 稀疏格式。

    每组 4 个元素, 保留非零值并记录掩码。
    非零元素被压缩存储, 掩码指示原始位置。

    Args:
        dense: 稠密数据 (长度应为 4 的倍数)

    Returns:
        (稀疏非零值, 掩码列表)
    """
    if len(dense) % 4 != 0:
        # Pad to multiple of 4
        padded = dense + [0.0] * (4 - len(dense) % 4)
    else:
        padded = list(dense)

    sparse_values: List[float] = []
    masks: List[int] = []

    for i in range(0, len(padded), 4):
        group = padded[i:i + 4]
        if SparsityMask.is_valid_2to4(group):
            # Already 2:4 structured: just take the non-zero elements
            non_zero = [v for v in group if v != 0]
            pattern = SparsityMask.find_pattern(group)
        else:
            # Force 2:4: keep the 2 largest absolute values
            indexed = sorted(enumerate(group), key=lambda x: abs(x[1]),
                             reverse=True)
            kept_cols = tuple(sorted([indexed[0][0], indexed[1][0]]))
            pattern = SparsityMask.encode(kept_cols)
            non_zero = [group[col] for col in kept_cols]

        sparse_values.extend(non_zero)
        masks.append(pattern)

    return sparse_values, masks


def sparse_to_dense_2to4(sparse_values: List[float],
                         masks: List[int]) -> List[float]:
    """将 2:4 稀疏格式解包为稠密格式。

    使用压缩的非零值和掩码重建完整的 4 元素组。

    Args:
        sparse_values: 压缩存储的非零值
        masks: 每组的 2-bit 掩码

    Returns:
        稠密数据 (每 4 个元素有 2 个零)
    """
    dense: List[float] = []
    val_idx = 0

    for mask in masks:
        cols = SparsityMask.decode(mask)
        group = [0.0, 0.0, 0.0, 0.0]
        for col in cols:
            if val_idx < len(sparse_values):
                group[col] = sparse_values[val_idx]
                val_idx += 1
        dense.extend(group)

    return dense


def sparse_mma(a_sparse: List[float], b_dense: List[float],
               a_masks: List[int], m: int, k: int, n: int) -> List[float]:
    """稀疏矩阵乘法 C = A_sparse @ B_dense。

    对标 NVIDIA 稀疏张量核心的 Sparse MMA。
    A 是 2:4 结构化稀疏矩阵, B 是稠密矩阵。
    硬件跳过 A 中零元素的乘法, 获得 ~2x 吞吐量提升。

    如果 A 以标准 2:4 格式压缩:
      - A 的非零值: m * (k // 2) 个元素 (因为 2:4 格式, 每行 k/2 个非零)
      - masks: m * (k // 4) 个掩码 (每组 4 列一个掩码)
      - B 为稠密: k x n

    Args:
        a_sparse: A 的压缩非零值 (每行 k/2 个值)
        b_dense: B 的稠密值 (k * n)
        a_masks: A 的掩码 (每行 k/4 个掩码)
        m: A 的行数
        k: A 的列数 / B 的行数
        n: B 的列数

    Returns:
        C 的稠密值 (m * n)
    """
    result = [0.0] * (m * n)
    val_idx = 0

    for i in range(m):
        # Decompress A row i from sparse format
        a_row = [0.0] * k
        mask_idx = i * (k // 4)

        for g in range(k // 4):
            if mask_idx + g < len(a_masks):
                mask = a_masks[mask_idx + g]
                cols = SparsityMask.decode(mask)
                for col in cols:
                    if col < k and val_idx < len(a_sparse):
                        a_row[col] = a_sparse[val_idx]
                        val_idx += 1

        # Compute C[i][j] = sum(A[i][p] * B[p][j])
        for j in range(n):
            total = 0.0
            for p in range(k):
                b_val = b_dense[p * n + j] if p * n + j < len(b_dense) else 0.0
                total += a_row[p] * b_val
            result[i * n + j] = total

    return result


class SparsityStats:
    """稀疏性统计 — 跟踪压缩率等信息。"""

    def __init__(self):
        self.original_elements = 0
        self.sparse_elements = 0
        self.num_groups = 0

    def record(self, original_count: int, sparse_count: int):
        """记录一次转换的统计信息。"""
        self.original_elements += original_count
        self.sparse_elements += sparse_count
        self.num_groups += original_count // 4

    @property
    def compression_ratio(self) -> float:
        """压缩率 (非零 / 原始元素)。"""
        if self.original_elements == 0:
            return 1.0
        return self.sparse_elements / self.original_elements

    def report(self) -> str:
        return (f"SparsityStats: original={self.original_elements}, "
                f"sparse={self.sparse_elements}, "
                f"ratio={self.compression_ratio:.3f}, "
                f"groups={self.num_groups}")
