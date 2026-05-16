"""
Console Display — 学习控制台渲染模块
=========================================
每周期渲染完整的 GPGPU 流水线状态：Fetch/Decode/Issue/Exec/WB
+ Scoreboard + I-Buffer + SIMT Stack + 寄存器变化。

初学者可以逐周期观察 GPU 内部的工作方式。
"""

from isa import OPCODE_NAMES
from operand_collector import OperandCollector


# ─── 颜色代码 (ANSI) ───
CLR = {
    'reset': '\033[0m', 'bold': '\033[1m',
    'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
    'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m',
    'gray': '\033[90m',
}
def c(text, color='reset'):
    return f"{CLR.get(color, '')}{text}{CLR['reset']}"


def render_cycle(cycle: int, simt, instr_info: dict, prev_regs: dict,
                 mem_changes: list, stage_info: dict) -> str:
    """渲染一个周期的完整状态

    Args:
        cycle: 当前周期数
        simt: SIMTCore 实例
        instr_info: {'op': str, 'rd': int, 'rs1': int, 'rs2': int, 'pc': int} or None
        prev_regs: {warp_id: {thread_id: [reg_values]}} 上周期的寄存器值
        mem_changes: [(addr, old_val, new_val)] 内存变化
        stage_info: {'fetch': str, 'decode': str, 'issue': str, 'exec': str, 'wb': str}
    """
    lines = []
    lines.append(c(f"╔══ Cycle {cycle:4d} " + "═" * 49 + "╗", 'cyan'))

    # ─── Warp 状态行 ───
    warp_parts = []
    for w in simt.warps:
        active_str = f"0b{w.active_mask:0{simt.warp_size}b}"
        status = 'DONE' if w.done else ('BAR' if w.at_barrier else 'ACT')
        color = 'gray' if w.done else ('yellow' if w.at_barrier else 'green')
        warp_parts.append(f"W{w.warp_id}:PC={w.pc:2d} mask={active_str} [{c(status, color)}]")
    lines.append("  " + " │ ".join(warp_parts))

    # ─── 流水线五阶段 ───
    if instr_info:
        op = instr_info.get('op', '?')
        lines.append("  ┌─ Pipeline ──────────────────────────────────────┐")
        stages = [
            ('FETCH ', 'gray',   stage_info.get('fetch', f"PC={instr_info.get('pc', '?')} → IBuffer")),
            ('DECODE', 'blue',   stage_info.get('decode', f"{op} decoded, scoreboard check")),
            ('ISSUE ', 'yellow', stage_info.get('issue', 'operand bank check')),
            ('EXEC  ', 'green',  stage_info.get('exec', f"{op} executing on {instr_info.get('active', '?')} threads")),
            ('WB    ', 'magenta',stage_info.get('wb', 'result → rd register')),
        ]
        for name, color, detail in stages:
            lines.append(f"  │ {c(name, color)} │ {detail:<45s} │")
        lines.append("  └────────────────────────────────────────────────┘")

    # ─── 寄存器变化 ───
    reg_lines = _render_reg_changes(simt, prev_regs)
    if reg_lines:
        lines.append("")
        lines.append("  ┌─ Register Changes ─────────────────────────────┐")
        for rl in reg_lines:
            lines.append(f"  │ {rl:<49s} │")
        lines.append("  └────────────────────────────────────────────────┘")

    # ─── Scoreboard ───
    sb_lines = _render_scoreboard(simt)
    lines.append("")
    lines.append("  ┌─ Scoreboard ───────────────────────────────────┐")
    if sb_lines:
        for sl in sb_lines:
            lines.append(f"  │ {sl:<49s} │")
    else:
        lines.append(f"  │ {c('  (clean)', 'gray'):<49s} │")
    lines.append("  └────────────────────────────────────────────────┘")

    # ─── I-Buffer ───
    ib_lines = _render_ibuffer(simt)
    lines.append("")
    lines.append("  ┌─ I-Buffer ─────────────────────────────────────┐")
    if ib_lines:
        for il in ib_lines:
            lines.append(f"  │ {il:<49s} │")
    else:
        lines.append(f"  │ {c('  (empty)', 'gray'):<49s} │")
    lines.append("  └────────────────────────────────────────────────┘")

    # ─── SIMT Stack ───
    stack_lines = _render_simt_stack(simt)
    if stack_lines:
        lines.append("")
        lines.append("  ┌─ SIMT Stack ───────────────────────────────────┐")
        for sl in stack_lines:
            lines.append(f"  │ {sl:<49s} │")
        lines.append("  └────────────────────────────────────────────────┘")

    # ─── 内存变化 ───
    if mem_changes:
        lines.append("")
        lines.append("  ┌─ Memory Changes ───────────────────────────────┐")
        for addr, old, new in mem_changes[:6]:
            lines.append(f"  │ mem[{addr:3d}]: 0x{old:08X} → {c(f'0x{new:08X}', 'yellow'):<30s} │")
        lines.append("  └────────────────────────────────────────────────┘")

    # ─── OpCollector Stats ───
    lines.append("")
    lines.append(f"  {simt.op_collector.stats()}  |  {simt.l1_cache.stats()}")

    lines.append(c("╚" + "═" * 58 + "╝", 'cyan'))
    return '\n'.join(lines)


def _render_reg_changes(simt, prev_regs):
    """渲染寄存器变化"""
    changes = []
    for w in simt.warps:
        for t in w.active_threads():
            tid = t.thread_id
            prev = prev_regs.get(w.warp_id, {}).get(tid, [0]*16)
            curr = [t.read_reg(i) for i in range(16)]
            for ri in range(1, 16):  # skip r0
                if prev[ri] != curr[ri]:
                    changes.append(
                        f"W{w.warp_id}T{tid} r{ri:2d}: "
                        f"0x{prev[ri]:08X} → {c(f'0x{curr[ri]:08X}', 'yellow')} "
                        f"({c(f'{_s32(curr[ri])}', 'green')})"
                    )
    return changes[:8]  # 最多显示 8 条


def _render_scoreboard(simt):
    """渲染 scoreboard 状态"""
    lines = []
    for w in simt.warps:
        sb = w.scoreboard
        if sb.reserved:
            items = []
            for reg, cycles in sb.reserved.items():
                color = 'red' if cycles > 0 else 'green'
                items.append(f"r{reg}:{c(str(cycles), color)}")
            lines.append(f"W{w.warp_id}: " + ' '.join(items))
    return lines


def _render_ibuffer(simt):
    """渲染 I-Buffer 状态"""
    lines = []
    for w in simt.warps:
        parts = []
        for i, e in enumerate(w.ibuffer.entries):
            if e.valid:
                opname = OPCODE_NAMES.get((e.instruction_word >> 24) & 0xFF, '?')
                ready = c('✓', 'green') if e.ready else c('✗', 'red')
                parts.append(f"[{opname:5s} PC={e.pc:2d} {ready}]")
            else:
                parts.append(f"[{c('empty', 'gray'):>14s}]")
        lines.append(f"W{w.warp_id}: " + ' '.join(parts))
    return lines


def _render_simt_stack(simt):
    """渲染 SIMT Stack 状态"""
    lines = []
    for w in simt.warps:
        stack = w.simt_stack
        if not stack.empty:
            for i, entry in enumerate(stack.entries):
                mask_bits = f"{entry.orig_mask:0{8}b}"
                taken_bits = f"{entry.taken_mask:0{8}b}"
                lines.append(
                    f"W{w.warp_id}[{i}]: reconv={entry.reconv_pc} "
                    f"orig={mask_bits} taken={taken_bits}"
                )
    return lines


def snapshot_regs(simt):
    """记录当前所有寄存器的快照"""
    regs = {}
    for w in simt.warps:
        regs[w.warp_id] = {}
        for t in w.threads:
            regs[w.warp_id][t.thread_id] = [t.read_reg(i) for i in range(16)]
    return regs


def snapshot_mem(simt):
    """记录当前内存快照（前 256 words）"""
    return [simt.memory.read_word(i) for i in range(min(256, simt.memory.size_words))]


def mem_diff(old_mem, new_mem):
    """计算内存变化"""
    changes = []
    for i in range(len(old_mem)):
        if old_mem[i] != new_mem[i]:
            changes.append((i, old_mem[i], new_mem[i]))
    return changes


def _s32(val):
    """32-bit unsigned → signed"""
    return val - 0x100000000 if val & 0x80000000 else val
