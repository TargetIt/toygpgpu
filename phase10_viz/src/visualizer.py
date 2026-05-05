"""
Visualizer — GPU Simulator Trace Visualization
================================================
Phase 10: Warp Timeline, Stall Analysis, Memory Heatmap, JSON Export.

对标 GPGPU-Sim 的 AerialVision 可视化 + stat-tool 统计模块。

输出:
  - ASCII Warp Timeline (PC over cycles)
  - Stall Cause Analysis (bar chart)
  - Memory Access Heatmap (address usage density)
  - JSON Trace Export (for external tools)

参考: GPGPU-Sim gpgpu-sim/visualizer.cc
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    """单条执行追踪事件"""
    cycle: int
    warp_id: int
    pc: int
    opcode: int
    opcode_name: str
    active_threads: int
    stall_reason: str = ""     # "" = no stall, "sb", "barrier", "ibuffer"
    mem_addr: int = -1         # memory access address (if LD/ST/SHLD/SHST)


class TraceCollector:
    """追踪数据收集器

    在 SIMTCore 执行过程中收集每个周期的执行事件。
    """

    def __init__(self):
        self.events: List[TraceEvent] = []
        self.mem_accesses: List[tuple] = []  # (cycle, warp_id, addr, is_write)
        self.total_cycles = 0
        self.stall_cycles = 0

    def record_exec(self, cycle: int, warp_id: int, pc: int,
                    opcode: int, opcode_name: str, active: int,
                    mem_addr: int = -1):
        self.events.append(TraceEvent(
            cycle=cycle, warp_id=warp_id, pc=pc,
            opcode=opcode, opcode_name=opcode_name,
            active_threads=active, mem_addr=mem_addr
        ))
        if mem_addr >= 0:
            self.mem_accesses.append((cycle, warp_id, mem_addr,
                                      opcode_name == 'ST'))

    def record_stall(self, cycle: int, warp_id: int, reason: str):
        self.events.append(TraceEvent(
            cycle=cycle, warp_id=warp_id, pc=-1,
            opcode=0, opcode_name='STALL',
            active_threads=0, stall_reason=reason
        ))
        self.stall_cycles += 1

    def export_json(self, filepath: str):
        """导出为 JSON 格式（兼容 Chrome Tracing / Perfetto）"""
        trace = []
        for e in self.events:
            trace.append({
                'cycle': e.cycle,
                'warp': e.warp_id,
                'pc': e.pc,
                'op': e.opcode_name,
                'active': e.active_threads,
                'stall': e.stall_reason or None,
                'mem_addr': e.mem_addr if e.mem_addr >= 0 else None,
            })
        with open(filepath, 'w') as f:
            json.dump({'events': trace, 'total_cycles': self.total_cycles,
                       'stall_rate': self.stall_cycles / max(1, self.total_cycles)},
                      f, indent=2)


# ============================================================
# ASCII 可视化生成器
# ============================================================

def warp_timeline(events: List[TraceEvent], num_warps: int,
                  max_cycles: int = 80, width: int = 60) -> str:
    """生成 Warp PC 时间线 (ASCII)

    每行 = 1 个 cycle，每列显示 warp 的 PC。

    Returns:
        多行字符串，适合终端打印
    """
    # 按 cycle 分组
    cycles: Dict[int, Dict[int, str]] = {}
    for e in events:
        if e.cycle not in cycles:
            cycles[e.cycle] = {}
        marker = f'{e.opcode_name[:4]:>4}' if e.pc >= 0 else '----'
        cycles[e.cycle][e.warp_id] = marker

    lines = []
    lines.append(f"Warp Timeline (first {min(max_cycles, len(cycles))} cycles)")
    lines.append(f"{'Cyc':>4} |" + ''.join(f' W{w:2d} ' for w in range(num_warps)))
    lines.append('-' * (6 + num_warps * 6))

    for cycle in sorted(cycles.keys())[:max_cycles]:
        parts = []
        for w in range(num_warps):
            parts.append(f'[{cycles[cycle].get(w, "    ")}]')
        lines.append(f'{cycle:4d} |' + ' '.join(parts))

    return '\n'.join(lines)


def stall_analysis(events: List[TraceEvent]) -> str:
    """生成 Stall 原因分析 (文本柱状图)

    Returns:
        含柱状图的格式字符串
    """
    reasons = {}
    total_stalls = 0
    for e in events:
        if e.stall_reason:
            reasons[e.stall_reason] = reasons.get(e.stall_reason, 0) + 1
            total_stalls += 1

    total = len(events)
    lines = []
    lines.append("Stall Analysis")
    lines.append(f"Total events: {total}, Stall events: {total_stalls} ({total_stalls/max(1,total)*100:.1f}%)")
    lines.append("")
    max_bar = 40

    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        bar_len = int(count / max(1, total_stalls) * max_bar)
        bar = '█' * bar_len
        lines.append(f"  {reason:<12} {count:>4d} |{bar}")

    return '\n'.join(lines)


def memory_heatmap(mem_accesses: List[tuple], mem_size: int = 256,
                   width: int = 32) -> str:
    """生成内存访问热力图 (ASCII)

    Args:
        mem_accesses: list of (cycle, warp_id, addr, is_write)
        mem_size: 内存大小 (words)
        width: 热力图宽度 (每行多少个地址)

    Returns:
        ASCII 热力图字符串
    """
    # 统计每个地址的访问次数
    access_count = [0] * mem_size
    for _, _, addr, _ in mem_accesses:
        if 0 <= addr < mem_size:
            access_count[addr] += 1

    max_count = max(access_count) if max(access_count) > 0 else 1

    # ASCII 密度字符: 从低到高
    density_chars = ' .:-=+*#%@'

    lines = []
    lines.append(f"Memory Access Heatmap (0-{mem_size-1}, width={width})")
    lines.append(f"Max accesses per word: {max_count}")
    lines.append('')

    for row_start in range(0, mem_size, width):
        row_chars = []
        for addr in range(row_start, min(row_start + width, mem_size)):
            density = int(access_count[addr] / max_count * (len(density_chars) - 1))
            row_chars.append(density_chars[density])
        lines.append(f'{row_start:4d} |' + ''.join(row_chars) + '|')

    return '\n'.join(lines)


def full_report(collector: TraceCollector, num_warps: int,
                mem_size: int = 256) -> str:
    """生成完整可视化报告"""
    parts = []
    parts.append("=" * 60)
    parts.append("  toygpgpu Execution Report")
    parts.append("=" * 60)
    parts.append("")
    parts.append(warp_timeline(collector.events, num_warps))
    parts.append("")
    parts.append(stall_analysis(collector.events))
    if collector.mem_accesses:
        parts.append("")
        parts.append(memory_heatmap(collector.mem_accesses, mem_size))
    parts.append("")
    parts.append("=" * 60)
    return '\n'.join(parts)
