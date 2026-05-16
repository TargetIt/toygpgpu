"""
Independent Thread Scheduling — Per-Thread PC Model (Phase 23)
==================================================================
对标 NVIDIA Volta+ 独立线程调度架构。

核心创新:
  - PerThreadPC: 每个线程有自己的 PC, 支持真正的独立线程执行
  - ReconvergenceEngine: 硬件管理的重汇聚, 无需 SIMT 栈
  - SIMTStack 保留用于向后兼容, 但新路径使用 per-thread PC
  - Thread divergence tracking without stack push/pop

Volta 的独立线程调度:
  - 每个线程有自己的 PC (不再是 warp 统一 PC)
  - 硬件自动管理重汇聚点
  - 线程在重汇聚点自动同步
  - 无需编译器插入的重汇聚代码

参考:
  - NVIDIA Volta Architecture Whitepaper
  - "Independent Thread Scheduling" in CUDA Programming Guide
  - GPGPU-Sim's divergence tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class PerThreadPC:
    """每个线程的 PC 状态。

    Attributes:
        thread_id: 线程在 warp 内的索引 (0-31)
        pc: 当前程序计数器
        active: 线程是否活跃
        next_pc: 下一条要执行的指令
    """

    thread_id: int
    pc: int = 0
    active: bool = True
    next_pc: int = 0

    def branch(self, target_pc: int, condition: bool):
        """线程独立分支: 如果条件满足则跳转到 target_pc, 否则 PC+1。

        Args:
            target_pc: 分支目标地址
            condition: 是否执行跳转
        """
        if condition:
            self.next_pc = target_pc
        else:
            self.next_pc = self.pc + 1

    def step(self):
        """推进到下一条指令。"""
        self.pc = self.next_pc
        self.next_pc = self.pc + 1

    def __repr__(self):
        return (f"PerThreadPC(tid={self.thread_id}, pc={self.pc}, "
                f"active={self.active})")


class ReconvergenceEngine:
    """硬件管理的重汇聚 — 检测线程何时在重汇聚点汇合。

    Volta 的重汇聚机制:
      - 每个 warp 有一个 reconvergence PC (RPC)
      - 所有线程独立执行, 直到所有线程都到达 RPC
      - 当所有线程到达 RPC, warp 继续统一执行
      - Reconvergence point 记录每个 diverging branch 的返回地址

    Attributes:
        warp_id: warp 标识
        num_threads: warp 中的线程数 (通常 32)
        threads: PerThreadPC 字典 {tid: PerThreadPC}
        reconv_points: 待处理的重汇聚点 {pc: set_of_thread_ids}
    """

    def __init__(self, warp_id: int = 0, num_threads: int = 32):
        self.warp_id = warp_id
        self.num_threads = num_threads
        self.threads: Dict[int, PerThreadPC] = {
            tid: PerThreadPC(thread_id=tid) for tid in range(num_threads)
        }
        self.reconv_points: Dict[int, Set[int]] = {}
        self.active_mask: int = (1 << num_threads) - 1  # all active

    def set_pc_all(self, pc: int):
        """将所有线程的 PC 设置为同一值。"""
        for t in self.threads.values():
            t.pc = pc
            t.next_pc = pc + 1
            t.active = True

    def branch_thread(self, tid: int, target_pc: int,
                      condition: bool, reconv_pc: int):
        """线程独立分支。

        如果线程的 `tid` 满足条件, 跳转到 target_pc, 否则执行下一指令。
        如果发生真正的发散, 记录重汇聚点。

        Args:
            tid: 线程 ID
            target_pc: 分支目标 PC
            condition: 是否满足分支条件
            reconv_pc: 重汇聚点 PC (分支后的地址)
        """
        thread = self.threads[tid]
        if not thread.active:
            return

        thread.branch(target_pc, condition)

        # Track reconvergence point if this thread is taking the branch
        # while other threads may not
        if condition:
            if reconv_pc not in self.reconv_points:
                self.reconv_points[reconv_pc] = set()
            self.reconv_points[reconv_pc].add(tid)

    def execute_instruction(self, tid: int, instr_pc: int) -> bool:
        """执行单线程单指令。

        Args:
            tid: 线程 ID
            instr_pc: 指令 PC

        Returns:
            是否成功执行
        """
        thread = self.threads[tid]
        if not thread.active:
            return False
        if thread.pc != instr_pc:
            return False

        thread.step()
        return True

    def check_reconvergence(self, pc: int) -> List[int]:
        """检查给定 PC 处是否有线程应重汇聚。

        当所有发散线程到达重汇聚点时, 返回所有重汇聚的线程 ID。

        Args:
            pc: 当前 PC

        Returns:
            重汇聚的线程 ID 列表 (在重汇聚点汇合的线程)
        """
        if pc not in self.reconv_points:
            return []

        waiting = self.reconv_points[pc]
        arrived = set()

        # Check which threads in the reconv set have reached this PC
        for tid in waiting:
            thread = self.threads[tid]
            if thread.pc == pc:
                arrived.add(tid)
            elif not thread.active:
                arrived.add(tid)

        # If all threads have arrived, reconvergence is complete
        if arrived == waiting:
            del self.reconv_points[pc]
            return list(arrived)

        return []

    def get_active_pcs(self) -> Dict[int, int]:
        """获取所有活跃线程的当前 PC。"""
        return {tid: t.pc for tid, t in self.threads.items() if t.active}

    def get_divergent_mask(self, pc: int) -> int:
        """获取给定 PC 处的发散掩码 — 哪些线程在指定 PC 处。

        Returns:
            位掩码: 位 n = 1 表示线程 n 的当前 PC 等于参数 pc
        """
        mask = 0
        for tid, thread in self.threads.items():
            if thread.active and thread.pc == pc:
                mask |= (1 << tid)
        return mask

    def step_all(self):
        """所有活跃线程推进到下一条指令。"""
        for thread in self.threads.values():
            if thread.active:
                thread.step()

    def get_pc_diversity(self) -> int:
        """返回 warp 中不同的 PC 数量 (发散度)。"""
        pcs = set()
        for t in self.threads.values():
            if t.active:
                pcs.add(t.pc)
        return len(pcs)

    def reset(self):
        """重置引擎。"""
        for tid, thread in self.threads.items():
            thread.pc = 0
            thread.next_pc = 0
            thread.active = True
        self.reconv_points.clear()
        self.active_mask = (1 << self.num_threads) - 1

    def __repr__(self):
        return (f"ReconvergenceEngine(warp={self.warp_id}, "
                f"divergent_pcs={self.get_pc_diversity()}, "
                f"reconv_points={len(self.reconv_points)})")
