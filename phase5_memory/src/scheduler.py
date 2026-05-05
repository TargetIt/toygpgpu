"""
WarpScheduler — Warp 调度器
============================
对标 GPGPU-Sim 中 scheduler_unit，负责从多个 warp 中选择下一个执行。

Phase 2 实现 Round-Robin (LRR / Loose Round Robin)。
后续 Phase 可加入 GTO (Greedy Then Oldest) 等策略。

GPGPU-Sim 的 scheduler_unit 通过 issue() 方法从 I-Buffer 中
选 warp 发射指令。Phase 2 简化为从 warp 列表直接轮询。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 scheduler_unit 实现
"""

from typing import List, Optional
from warp import Warp


class WarpScheduler:
    """Warp 调度器

    Attributes:
        warps: 所有 warp 的列表
        current_idx: 当前被选中的 warp 索引
        policy: 调度策略 "rr" (round-robin)
    """

    def __init__(self, warps: List[Warp], policy: str = "rr"):
        self.warps = warps
        self.current_idx = 0
        self.policy = policy
        self.cycle_count = 0

    def select_warp(self) -> Optional[Warp]:
        """选择下一个可执行的 warp

        对应 GPGPU-Sim 中 scheduler_unit::cycle() 的 issue 阶段。
        Round-Robin: 轮询所有 warp，跳过已完成的。

        Returns:
            可执行的 Warp，或 None（所有 warp 完成）
        """
        num_warps = len(self.warps)
        # 尝试轮询一轮
        for _ in range(num_warps):
            warp = self.warps[self.current_idx]
            self.current_idx = (self.current_idx + 1) % num_warps

            if not warp.done and not warp.at_barrier and not warp.scoreboard_stalled:
                self.cycle_count += 1
                return warp

        # 所有 warp 都完成（只有 barrier wait 也算半完成）
        active = [w for w in self.warps if not w.done]
        if not active:
            return None

        # 如果还有 warp 在 barrier wait 但未完成...应该处理
        # Phase 2 简化：返回 None（理论上不应该到这里）
        return None

    def has_active_warps(self) -> bool:
        """是否有未完成的 warp"""
        return any(not w.done for w in self.warps)
