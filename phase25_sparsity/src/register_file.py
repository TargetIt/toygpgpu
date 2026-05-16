"""
RegisterFile — 寄存器堆模块
============================
对标 GPGPU-Sim 中 shader_core_ctx 的寄存器堆。

16 个 32-bit 通用寄存器 (r0-r15)。
r0 硬连线为 0 — 写忽略，读恒为 0。
这一设计参考 RISC-V 的 x0 寄存器。

参考: GPGPU-Sim shader.h 中的寄存器管理
      TinyGPU 的 per-thread register file
"""


class RegisterFile:
    """16 × 32-bit 寄存器堆

    每个线程 / 每个 warp 拥有独立的寄存器堆。
    Phase 0 中只有一个隐含线程，因此只有一个 RegisterFile 实例。

    Attributes:
        regs: 16 个 32-bit 寄存器的列表 (regs[0] 始终为 0)
    """

    def __init__(self, num_regs: int = 16):
        self._num_regs = num_regs
        self.regs = [0] * num_regs

    def read(self, reg_id: int) -> int:
        """读寄存器

        对应硬件中的读端口。r0 硬连线为 0。

        Args:
            reg_id: 寄存器编号 (0-15)

        Returns:
            32-bit 寄存器值

        Raises:
            IndexError: 寄存器编号越界
        """
        if reg_id < 0 or reg_id >= self._num_regs:
            raise IndexError(f"Register index out of range: r{reg_id} (0-{self._num_regs-1})")
        return self.regs[reg_id]  # regs[0] 永远是 0

    def write(self, reg_id: int, value: int):
        """写寄存器

        r0 写忽略（硬件连线到 GND）。

        Args:
            reg_id: 寄存器编号 (0-15)
            value: 32-bit 写入值（自动截断到 32-bit）
        """
        if reg_id < 0 or reg_id >= self._num_regs:
            raise IndexError(f"Register index out of range: r{reg_id}")
        if reg_id == 0:
            return  # r0 写忽略

        # 截断到 32-bit（模拟硬件位宽）
        self.regs[reg_id] = value & 0xFFFFFFFF

    def dump(self) -> str:
        """打印寄存器状态（调试用）"""
        lines = []
        for i in range(self._num_regs):
            val = self.regs[i]
            marker = " (r0=0)" if i == 0 else ""
            lines.append(f"  r{i:2d}: 0x{val:08X} ({val}){marker}")
        return "\n".join(lines)
