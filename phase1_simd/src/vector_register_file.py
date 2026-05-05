"""
VectorRegisterFile — 向量寄存器堆模块
======================================
对标 GPGPU-Sim 中 warp 内 32 个线程的寄存器堆。

每个向量寄存器 = VLEN 个 32-bit lane。
8 个向量寄存器 (v0-v7)。

GPGPU-Sim 中每个 warp 有 N 个线程，每个线程有独立的标量寄存器。
Phase 1 的向量寄存器 = 把 warp 的所有线程寄存器"并排"成一个宽寄存器。

参考: GPGPU-Sim shader.h 中 shd_warp_t 的寄存器管理
       RISC-V V extension 的向量寄存器模型
"""

from typing import List


class VectorRegisterFile:
    """VLEN × 8 向量寄存器堆

    每个向量寄存器包含 VLEN 个 32-bit lane。

    Attributes:
        vlen: 向量长度 (lane 数, 默认 8)
        num_regs: 向量寄存器数 (默认 8, v0-v7)
        regs: regs[reg_id][lane] → 32-bit value
    """

    def __init__(self, vlen: int = 8, num_regs: int = 8):
        self.vlen = vlen
        self.num_regs = num_regs
        # 每个寄存器是一个 VLEN 长度的 list
        self.regs = [[0] * vlen for _ in range(num_regs)]

    def read(self, reg_id: int) -> List[int]:
        """读向量寄存器（返回 VLEN 个 lane 值）"""
        if reg_id < 0 or reg_id >= self.num_regs:
            raise IndexError(f"Vector register index out of range: v{reg_id}")
        return list(self.regs[reg_id])

    def read_lane(self, reg_id: int, lane: int) -> int:
        """读单个 lane"""
        if reg_id < 0 or reg_id >= self.num_regs:
            raise IndexError(f"Vector register index out of range: v{reg_id}")
        if lane < 0 or lane >= self.vlen:
            raise IndexError(f"Lane index out of range: {lane}")
        return self.regs[reg_id][lane]

    def write(self, reg_id: int, values: List[int]):
        """写向量寄存器（自动截断到 32-bit）"""
        if reg_id < 0 or reg_id >= self.num_regs:
            raise IndexError(f"Vector register index out of range: v{reg_id}")
        for i in range(min(self.vlen, len(values))):
            self.regs[reg_id][i] = values[i] & 0xFFFFFFFF

    def write_lane(self, reg_id: int, lane: int, value: int):
        """写单个 lane"""
        if reg_id < 0 or reg_id >= self.num_regs:
            raise IndexError(f"Vector register index out of range: v{reg_id}")
        if lane < 0 or lane >= self.vlen:
            raise IndexError(f"Lane index out of range: {lane}")
        self.regs[reg_id][lane] = value & 0xFFFFFFFF

    def broadcast(self, reg_id: int, value: int):
        """将同一个值广播到所有 lane"""
        self.write(reg_id, [value] * self.vlen)

    def dump(self) -> str:
        """打印向量寄存器状态"""
        lines = []
        for i in range(self.num_regs):
            vals = ", ".join(
                f"0x{self.regs[i][j]:08X}" for j in range(min(4, self.vlen))
            )
            tail = "..." if self.vlen > 4 else ""
            lines.append(f"  v{i}: [{vals}{tail}]")
        return "\n".join(lines)
