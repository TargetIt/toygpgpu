"""
Scoreboard — 记分板 / 寄存器冒险检测
=======================================
对标 GPGPU-Sim 中 scoreboard 类 (gpgpu-sim/scoreboard.h)。

跟踪每个寄存器的 pending write 状态，检测 RAW 和 WAW 数据冒险。
检测到冒险时，warp 被 stall，等待 pending write 完成。

GPGPU-Sim 的 scoreboard:
  - check_collision(): 检测 WAW 和 RAW 冲突
  - reserve_reg(): 标记寄存器有 pending write
  - release_reg(): 写回后清除

Phase 3 简化为每个周期的自动延迟递减。

参考: GPGPU-Sim gpgpu-sim/scoreboard.h
"""


class Scoreboard:
    """寄存器记分板

    每个 warp 一个 scoreboard 实例。
    跟踪哪些寄存器有正在进行的写操作（pending write）。

    Attributes:
        reserved: {reg_id: remaining_cycles}
    """

    def __init__(self):
        self.reserved: dict[int, int] = {}

    def advance(self):
        """每周期推进一个周期

        所有 pending write 的剩余周期减 1，
        到期的自动清除（模拟硬件写回完成）。
        """
        expired = []
        for reg_id in self.reserved:
            self.reserved[reg_id] -= 1
            if self.reserved[reg_id] <= 0:
                expired.append(reg_id)
        for reg_id in expired:
            del self.reserved[reg_id]

    def check_raw(self, rs1: int, rs2: int = 0) -> bool:
        """检查 RAW (Read After Write) 冒险

        如果源寄存器有 pending write，说明前面还有指令
        正在写这个寄存器，当前指令不能读。

        Args:
            rs1: 源寄存器 1
            rs2: 源寄存器 2 (0 表示不使用)

        Returns:
            True = 有 RAW 冒险，需要 stall
        """
        if rs1 != 0 and rs1 in self.reserved:
            return True
        if rs2 != 0 and rs2 in self.reserved:
            return True
        return False

    def check_waw(self, rd: int) -> bool:
        """检查 WAW (Write After Write) 冒险

        如果目的寄存器有 pending write，说明前面还有指令
        正在写这个寄存器。当前指令必须等它完成。

        Args:
            rd: 目的寄存器

        Returns:
            True = 有 WAW 冒险，需要 stall
        """
        return rd != 0 and rd in self.reserved

    def reserve(self, rd: int, latency: int):
        """标记寄存器有 pending write

        Args:
            rd: 目的寄存器 (r0 忽略)
            latency: 写回需要的周期数
        """
        if rd != 0:
            self.reserved[rd] = latency

    @property
    def stalled(self) -> bool:
        """是否有任何 pending write (warp 被 stall?)"""
        return len(self.reserved) > 0

    def __repr__(self):
        if not self.reserved:
            return "Scoreboard(clean)"
        items = ", ".join(f"r{r}={c}" for r, c in self.reserved.items())
        return f"Scoreboard({items})"


# ============================================================
# 流水线延迟配置
# ============================================================
# 对标 GPGPU-Sim 中不同执行单元的流水线深度。
PIPELINE_LATENCY = {
    # ALU: 1 cycle (execute + writeback 在同一周期)
    'alu': 1,
    # LD/ST: 4 cycles (address calc + cache access + writeback)
    'mem': 4,
    # 其余: 1 cycle
    'default': 1,
}
