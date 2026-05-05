"""
VectorALU — 向量算术逻辑单元
=============================
对标 GPGPU-Sim 中 simd_function_unit 的多 lane 并行执行。

VLEN 个 ALU lane 同时执行同一条运算指令。

GPGPU-Sim 的 simd_function_unit 在 execute() 阶段对 warp 中
所有活跃线程执行运算。Phase 1 简化为: 所有 lane 始终活跃。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 simd_function_unit::cycle()
"""

from typing import List
from alu import ALU


class VectorALU:
    """VLEN 路并行 ALU

    每条 lane 运行一个独立的 ALU 实例。
    GPGPU-Sim 对应: 32 个 SP（Streaming Processor）并行执行。
    """

    def __init__(self, vlen: int = 8):
        self.vlen = vlen

    def vadd(self, a: List[int], b: List[int]) -> List[int]:
        """向量加法: 每 lane 独立加"""
        return [ALU.add(a[i], b[i]) for i in range(self.vlen)]

    def vsub(self, a: List[int], b: List[int]) -> List[int]:
        """向量减法"""
        return [ALU.sub(a[i], b[i]) for i in range(self.vlen)]

    def vmul(self, a: List[int], b: List[int]) -> List[int]:
        """向量乘法"""
        return [ALU.mul(a[i], b[i]) for i in range(self.vlen)]

    def vdiv(self, a: List[int], b: List[int]) -> List[int]:
        """向量除法"""
        return [ALU.div(a[i], b[i]) for i in range(self.vlen)]
