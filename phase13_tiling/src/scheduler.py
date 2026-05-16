"""
WarpScheduler — Warp 调度器 (Phase 6: + GTO policy)
======================================================
对标 GPGPU-Sim 中 scheduler_unit。

RR (Round-Robin): 轮询所有 warp
GTO (Greedy Then Oldest): 优先选最久未执行的 warp (GPGPU-Sim 默认策略)

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 scheduler_unit 实现
"""

from typing import List, Optional
from warp import Warp


class WarpScheduler:
    """Warp 调度器

    Attributes:
        warps: 所有 warp 的列表
        current_idx: RR 策略的当前 warp 索引
        policy: "rr" (Round-Robin) 或 "gto" (Greedy Then Oldest)
    """

    def __init__(self, warps: List[Warp], policy: str = "rr"):
        self.warps = warps
        self.current_idx = 0
        self.policy = policy
        self.cycle_count = 0
        self._warp_last_issue = [0] * len(warps)

    def select_warp(self) -> Optional[Warp]:
        if self.policy == "gto":
            return self._select_gto()
        return self._select_rr()

    def _select_rr(self):
        num = len(self.warps)
        for _ in range(num):
            warp = self.warps[self.current_idx]
            self.current_idx = (self.current_idx + 1) % num
            if not warp.done and not warp.at_barrier and not warp.scoreboard_stalled:
                self.cycle_count += 1
                self._warp_last_issue[warp.warp_id] = self.cycle_count
                return warp
        return None

    def _select_gto(self):
        """GTO: 选 oldest pending warp"""
        candidates = [w for w in self.warps
                      if not w.done and not w.at_barrier and not w.scoreboard_stalled]
        if not candidates:
            return None
        warp = min(candidates, key=lambda w: self._warp_last_issue[w.warp_id])
        self.cycle_count += 1
        self._warp_last_issue[warp.warp_id] = self.cycle_count
        return warp

    def has_active_warps(self) -> bool:
        return any(not w.done for w in self.warps)
