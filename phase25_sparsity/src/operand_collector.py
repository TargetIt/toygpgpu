"""
Operand Collector — 操作数收集器 (Banked Register File)
==========================================================
对标 GPGPU-Sim 中 opndcoll_rfu_t 的 bank 仲裁逻辑。

寄存器堆分为 4 个 bank (reg_id % 4)，每 bank 每周期 1 个读端口。
同一 bank 的多个读请求需要串行化（bank conflict）。

GPGPU-Sim 的 opndcoll_rfu_t: 每个执行单元管线(SP/SFU/MEM)有独立的
collector unit pool, 负责仲裁操作数读取请求。

Phase 7 简化: 统一 collector, 每个 warp 的指令操作数经过它读取。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 opndcoll_rfu_t::allocate_cu()
"""

from typing import List, Tuple, Optional


class OperandCollector:
    """Banked Register File 操作数收集器

    寄存器 r0-r15 按 reg_id % 4 分配到 4 个 bank:
      Bank 0: r0, r4, r8, r12
      Bank 1: r1, r5, r9, r13
      Bank 2: r2, r6, r10, r14
      Bank 3: r3, r7, r11, r15

    r0 硬连线到 0，读 r0 不消耗 bank 端口。

    Attributes:
        num_banks: bank 数量
        bank_busy: 每个 bank 当周期是否被占用
        conflict_count: 累计 bank conflict 次数
        total_reqs: 累计操作数读取请求数
    """

    def __init__(self, num_banks: int = 4):
        self.num_banks = num_banks
        self.bank_busy = [False] * num_banks
        self.conflict_count = 0
        self.total_reqs = 0
        self.pending_operands: dict = {}  # warp_id → pending reg reads

    @staticmethod
    def bank_of(reg_id: int) -> int:
        """返回寄存器所属 bank"""
        return reg_id % 4 if reg_id != 0 else -1  # r0 无 bank

    def can_read_operands(self, rs1: int, rs2: int) -> Tuple[bool, Optional[str]]:
        """检查能否在本周期读取操作数

        Returns:
            (can_read, reason_if_not)
        """
        banks_needed = set()
        if rs1 != 0:
            banks_needed.add(self.bank_of(rs1))
        if rs2 != 0:
            banks_needed.add(self.bank_of(rs2))

        busy = [b for b in banks_needed if self.bank_busy[b]]
        if busy:
            return False, f"bank conflict: banks {busy} busy"

        return True, None

    def reserve_banks(self, rs1: int, rs2: int):
        """占用 bank 端口 (操作数读取期间)"""
        if rs1 != 0:
            self.bank_busy[self.bank_of(rs1)] = True
        if rs2 != 0:
            b2 = self.bank_of(rs2)
            if rs1 != 0 and self.bank_of(rs1) == b2:
                self.conflict_count += 1  # 同 bank 访问
            self.bank_busy[b2] = True
        self.total_reqs += 1

    def release_banks(self):
        """释放所有 bank 端口 (下一周期)"""
        self.bank_busy = [False] * self.num_banks

    def bank_conflict_rate(self) -> float:
        return self.conflict_count / self.total_reqs if self.total_reqs > 0 else 0.0

    def stats(self) -> str:
        return (f"OpCollector: {self.total_reqs} reads, "
                f"{self.conflict_count} bank conflicts "
                f"({self.bank_conflict_rate():.1%})")
