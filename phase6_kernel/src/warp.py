"""
Warp & Thread — SIMT 执行单元
===============================
对标 GPGPU-Sim 中 shd_warp_t 和其管理的线程组。

GPGPU-Sim 的 shd_warp_t 包含:
  - 线程状态（寄存器、PC）
  - SIMT Stack（分支发散/重汇聚，Phase 3 加入）
  - active mask（当前活跃线程掩码）
  - barrier 状态

Phase 2 实现: 线程独立寄存器 + 共享 PC + active mask + barrier。

参考: GPGPU-Sim gpgpu-sim/shader.h 中 shd_warp_t 定义
"""

from typing import List
from register_file import RegisterFile


class Thread:
    """SIMT 线程

    对标 GPGPU-Sim 中一个 warp 内的 thread。
    当前仅包含寄存器堆，后续 Phase 可加入局部内存等。

    Attributes:
        thread_id: 线程 ID (0..WARP_SIZE-1)
        reg_file: 独立的 16×32-bit 寄存器堆
    """

    def __init__(self, thread_id: int, num_regs: int = 16):
        self.thread_id = thread_id
        self.reg_file = RegisterFile(num_regs)

    def read_reg(self, reg_id: int) -> int:
        return self.reg_file.read(reg_id)

    def write_reg(self, reg_id: int, value: int):
        self.reg_file.write(reg_id, value)

    def dump_regs(self) -> str:
        """单线程寄存器快照（仅非零值）"""
        non_zero = []
        for i in range(16):
            v = self.reg_file.read(i)
            if v != 0:
                non_zero.append(f"r{i}={v}")
        return f"T{self.thread_id}: " + (", ".join(non_zero) if non_zero else "all zero")


class Warp:
    """SIMT Warp

    对标 GPGPU-Sim 中 shd_warp_t。
    一组线程共享 PC，执行同一条指令。

    GPGPU-Sim 的 warp 最多 32 线程，本项目默认 WARP_SIZE=8。

    Attributes:
        warp_id: warp 编号
        threads: 线程列表
        pc: 程序计数器（warp 内所有线程共享）
        active_mask: bitmask of active threads
        at_barrier: 是否在等待 barrier
        barrier_count: 已到达 barrier 的线程数
        done: warp 是否已完成（遇到 HALT）
    """

    def __init__(self, warp_id: int, warp_size: int = 8):
        self.warp_id = warp_id
        self.threads = [Thread(tid) for tid in range(warp_size)]
        self.pc = 0
        self.active_mask = (1 << warp_size) - 1  # 初始全活跃
        self.at_barrier = False
        self.barrier_count = 0
        self.done = False
        from simt_stack import SIMTStack
        self.simt_stack = SIMTStack()
        from scoreboard import Scoreboard
        self.scoreboard = Scoreboard()
        self.scoreboard_stalled = False  # warp 被 scoreboard stall

    @property
    def warp_size(self) -> int:
        return len(self.threads)

    def active_threads(self) -> List[Thread]:
        """返回所有活跃线程"""
        return [t for i, t in enumerate(self.threads)
                if (self.active_mask >> i) & 1]

    def is_active(self, thread_id: int) -> bool:
        return bool((self.active_mask >> thread_id) & 1)

    def reset_barrier(self):
        """重置 barrier 状态"""
        self.at_barrier = False
        self.barrier_count = 0

    def dump(self) -> str:
        lines = [f"Warp {self.warp_id}: PC={self.pc}, "
                 f"active=0b{self.active_mask:0{self.warp_size}b}, "
                 f"barrier={self.at_barrier}, done={self.done}"]
        for t in self.threads:
            lines.append(f"  {t.dump_regs()}")
        return "\n".join(lines)
