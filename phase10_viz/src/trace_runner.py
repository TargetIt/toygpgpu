"""
TraceRunner — 带追踪的模拟器运行器
=====================================
包装 SIMTCore，在每次 step() 时记录事件。
"""
from visualizer import TraceCollector, full_report
from isa import OPCODE_NAMES, OP_LD, OP_ST, OP_SHLD, OP_SHST


def run_with_trace(simt, max_cycles=500):
    """运行 SIMTCore 并收集追踪数据"""
    collector = TraceCollector()
    MEM_OPS = {OP_LD, OP_ST, OP_SHLD, OP_SHST}

    for cycle in range(max_cycles):
        collector.total_cycles = cycle + 1

        warp = simt.scheduler.select_warp()
        if warp is None:
            if not simt.scheduler.has_active_warps():
                break
            # All warps stalled
            for w in simt.warps:
                if not w.done:
                    reason = 'sb' if w.scoreboard_stalled else 'barrier'
                    collector.record_stall(cycle, w.warp_id, reason)
            # Advance scoreboards
            for w in simt.warps:
                w.scoreboard.advance()
                if not w.scoreboard.stalled:
                    w.scoreboard_stalled = False
                w._fetched_this_cycle = False
            simt.op_collector.release_banks()
            simt._fetch_decode()
            continue

        # Normal execution cycle
        old_active = warp.active_mask

        # Try to execute
        if not warp.done and warp.pc < len(simt.program):
            # Check if we can issue
            entry = warp.ibuffer.consume()
            if entry is not None:
                instr = type('Instr', (), {})()
                from isa import decode
                instr = decode(entry.instruction_word)

                sb = warp.scoreboard
                if sb.check_waw(instr.rd) or sb.check_raw(instr.rs1, instr.rs2):
                    warp.scoreboard_stalled = True
                    warp.ibuffer.write(entry.instruction_word, entry.pc)
                    warp.ibuffer.set_ready(entry.pc)
                else:
                    simt._execute_warp(warp, instr, entry.pc)
                    mem_addr = instr.imm if instr.opcode in MEM_OPS else -1
                    collector.record_exec(
                        cycle, warp.warp_id, entry.pc,
                        instr.opcode, OPCODE_NAMES.get(instr.opcode, '?'),
                        bin(warp.active_mask).count('1'), mem_addr
                    )
                    latency = simt._get_latency(instr.opcode)
                    sb.reserve(instr.rd, latency)

        # Advance scoreboards + fetch
        for w in simt.warps:
            w.scoreboard.advance()
            if not w.scoreboard.stalled:
                w.scoreboard_stalled = False
            w._fetched_this_cycle = False
        simt.op_collector.release_banks()
        simt._fetch_decode()

    return collector


def trace_and_report(simt, num_warps=1, mem_size=256, json_path=None):
    """运行并生成完整可视化报告"""
    collector = run_with_trace(simt)
    report = full_report(collector, num_warps, mem_size)
    if json_path:
        collector.export_json(json_path)
    return report, collector
