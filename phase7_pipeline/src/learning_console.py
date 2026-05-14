#!/usr/bin/env python3
"""
Learning Console -- Interactive GPU Pipeline Learning Console (Phase 7)
=======================================================================
Allows step-by-step execution to observe GPU internal state each cycle.

Usage:
  python3 learning_console.py <program.asm> [options]

Options:
  --warp-size N      Warp size (default 4)
  --num-warps N      Number of warps (default 1)
  --max-cycles N     Maximum cycles (default 500)

Commands:
  Enter / s   Step one cycle
  r           Run until all warps complete
  r N         Run N cycles
  i           Print current full state
  m           Print non-zero memory
  reg         Print all registers
  sb          Print scoreboard
  ib          Print I-Buffer
  oc          Print operand collector bank state
  stack       Print SIMT stack
  q           Quit
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simt_core import SIMTCore
from assembler import assemble
from isa import OPCODE_NAMES, decode


# ---- ANSI color helpers ----
CLR = {
    'reset': '\033[0m', 'bold': '\033[1m',
    'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
    'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m',
    'gray': '\033[90m',
}


def c(text, color='reset'):
    return CLR.get(color, '') + text + CLR['reset']


# ---- Snapshot helpers (no external console_display) ----

def snapshot_regs(simt):
    """Record snapshot of all registers"""
    regs = {}
    for w in simt.warps:
        regs[w.warp_id] = {}
        for t in w.threads:
            regs[w.warp_id][t.thread_id] = [t.read_reg(i) for i in range(16)]
    return regs


def snapshot_mem(simt):
    """Record memory snapshot (first 256 words)"""
    return [simt.memory.read_word(i) for i in range(min(256, simt.memory.size_words))]


def mem_diff(old_mem, new_mem):
    """Calculate memory changes"""
    changes = []
    for i in range(min(len(old_mem), len(new_mem))):
        if old_mem[i] != new_mem[i]:
            changes.append((i, old_mem[i], new_mem[i]))
    return changes


def _signed32(val):
    """32-bit unsigned to signed"""
    return val - 0x100000000 if val & 0x80000000 else val


# ---- Render functions ----

def render_cycle(cycle, simt, instr_info, prev_regs, mem_changes, stage_info):
    """Render full cycle state as ASCII string"""
    lines = []
    lines.append(c("+--- Cycle %d " % cycle + "-" * 45 + "+", 'cyan'))

    # Warp status line
    warp_parts = []
    for w in simt.warps:
        active_str = "0b{0:0{1}b}".format(w.active_mask, simt.warp_size)
        if w.done:
            status, col = 'DONE', 'gray'
        elif w.at_barrier:
            status, col = 'BAR', 'yellow'
        else:
            status, col = 'ACT', 'green'
        s = c("[%s]" % status, col)
        warp_parts.append("W%d:PC=%d mask=%s %s" % (w.warp_id, w.pc, active_str, s))
    lines.append("  " + " | ".join(warp_parts))

    # Pipeline stages
    if instr_info:
        op = instr_info.get('op', '?')
        lines.append("  +- Pipeline ------------------------------------------+")
        stages = [
            ('FETCH ', 'gray',   stage_info.get('fetch', '')),
            ('DECODE', 'blue',   stage_info.get('decode', '')),
            ('ISSUE ', 'yellow', stage_info.get('issue', '')),
            ('EXEC  ', 'green',  stage_info.get('exec', '')),
            ('WB    ', 'magenta',stage_info.get('wb', '')),
        ]
        for name, color, detail in stages:
            lines.append("  | %s | %s" % (c(name, color), detail))
        lines.append("  +-----------------------------------------------------+")
    else:
        lines.append("  [no instruction issued this cycle]")

    # Register changes
    reg_lines = _render_reg_changes(simt, prev_regs)
    if reg_lines:
        lines.append("")
        lines.append("  +- Register Changes ------------------------------------+")
        for rl in reg_lines:
            lines.append("  | %s" % rl)
        lines.append("  +-------------------------------------------------------+")

    # Scoreboard
    sb_lines = _render_scoreboard(simt)
    lines.append("")
    lines.append("  +- Scoreboard -------------------------------------------+")
    if sb_lines:
        for sl in sb_lines:
            lines.append("  | %s" % sl)
    else:
        lines.append("  |  (clean)")
    lines.append("  +---------------------------------------------------------+")

    # I-Buffer
    ib_lines = _render_ibuffer(simt)
    lines.append("")
    lines.append("  +- I-Buffer ---------------------------------------------+")
    if ib_lines:
        for il in ib_lines:
            lines.append("  | %s" % il)
    else:
        lines.append("  |  (empty)")
    lines.append("  +---------------------------------------------------------+")

    # SIMT Stack
    stack_lines = _render_simt_stack(simt)
    if stack_lines:
        lines.append("")
        lines.append("  +- SIMT Stack -------------------------------------------+")
        for sl in stack_lines:
            lines.append("  | %s" % sl)
        lines.append("  +---------------------------------------------------------+")

    # Memory changes
    if mem_changes:
        lines.append("")
        lines.append("  +- Memory Changes -------------------------------------+")
        for addr, old, new_val in mem_changes[:6]:
            lines.append("  | mem[%3d]: 0x%08X -> 0x%08X" % (addr, old, new_val))
        lines.append("  +-------------------------------------------------------+")

    # Stats
    lines.append("")
    lines.append("  %s  |  %s" % (simt.op_collector.stats(), simt.l1_cache.stats()))

    lines.append(c("+" + "-" * 58 + "+", 'cyan'))
    return '\n'.join(lines)


def _render_reg_changes(simt, prev_regs):
    """Render register change lines"""
    changes = []
    for w in simt.warps:
        for t in w.active_threads():
            tid = t.thread_id
            prev = prev_regs.get(w.warp_id, {}).get(tid, [0] * 16)
            curr = [t.read_reg(i) for i in range(16)]
            for ri in range(1, 16):
                if prev[ri] != curr[ri]:
                    changes.append(
                        "W%d T%d r%2d: 0x%08X -> 0x%08X (%d)" % (
                            w.warp_id, tid, ri, prev[ri], curr[ri],
                            _signed32(curr[ri]))
                    )
    return changes[:8]


def _render_scoreboard(simt):
    """Render scoreboard status lines"""
    lines = []
    for w in simt.warps:
        sb = w.scoreboard
        if sb.reserved:
            items = []
            for reg, cycles in sb.reserved.items():
                col = 'red' if cycles > 0 else 'green'
                items.append("r%d:%s" % (reg, c(str(cycles), col)))
            lines.append("W%d: %s" % (w.warp_id, ' '.join(items)))
    return lines


def _render_ibuffer(simt):
    """Render I-Buffer status lines"""
    lines = []
    for w in simt.warps:
        parts = []
        for i, e in enumerate(w.ibuffer.entries):
            if e.valid:
                opname = OPCODE_NAMES.get((e.instruction_word >> 24) & 0x7F, '?')
                ready = c('R', 'green') if e.ready else c('W', 'red')
                parts.append("[%s PC=%d %s]" % (opname, e.pc, ready))
            else:
                parts.append("[%s]" % c('empty', 'gray'))
        lines.append("W%d: %s" % (w.warp_id, ' '.join(parts)))
    return lines


def _render_simt_stack(simt):
    """Render SIMT stack status lines"""
    lines = []
    for w in simt.warps:
        stack = w.simt_stack
        if not stack.empty:
            for i, entry in enumerate(stack.entries):
                orig_bits = "{0:0{1}b}".format(entry.orig_mask, 8)
                taken_bits = "{0:0{1}b}".format(entry.taken_mask, 8)
                lines.append("W%d[%d]: reconv=%d orig=%s taken=%s" % (
                    w.warp_id, i, entry.reconv_pc, orig_bits, taken_bits))
    return lines


# ---- Print helpers for interactive commands ----

def print_state(simt):
    """Print current state summary"""
    print(c("--- Current State ---", 'bold'))
    for w in simt.warps:
        status = 'DONE' if w.done else 'ACTIVE'
        mask_str = "0b{0:0{1}b}".format(w.active_mask, simt.warp_size)
        print("W%d: PC=%d, mask=%s, %s" % (w.warp_id, w.pc, mask_str, status))
        print("  Scoreboard: %s" % w.scoreboard)
        print("  I-Buffer: %s" % w.ibuffer)
        if not w.simt_stack.empty:
            print("  SIMT Stack depth: %d" % len(w.simt_stack))


def print_memory(simt):
    """Print non-zero memory"""
    non_zero = []
    for i in range(min(256, simt.memory.size_words)):
        v = simt.memory.read_word(i)
        if v != 0:
            non_zero.append("mem[%3d]=0x%08X(%d)" % (i, v, v))
    if non_zero:
        print('  ' + '\n  '.join(non_zero[:20]))
    else:
        print('  (all zero)')


def print_registers(simt):
    """Print all non-zero registers"""
    for w in simt.warps:
        print("Warp %d:" % w.warp_id)
        for t in w.active_threads():
            vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
            if vals:
                reg_str = ' '.join("r%d=%d" % (i, v) for i, v in vals)
                print("  T%d: %s" % (t.thread_id, reg_str))


def print_scoreboard(simt):
    for w in simt.warps:
        print("W%d Scoreboard: %s" % (w.warp_id, w.scoreboard))


def print_ibuffer(simt):
    for w in simt.warps:
        print("W%d I-Buffer: %s" % (w.warp_id, w.ibuffer))


def print_warp_regs(simt):
    """Print warp-level uniform registers"""
    from isa import WREG_NAMES
    rev = {v: k for k, v in WREG_NAMES.items()}
    print(c("--- Warp Registers ---", 'bold'))
    for w in simt.warps:
        wregs = []
        for idx, val in sorted(w.warp_regs.items()):
            name = rev.get(idx, 'wreg%d' % idx)
            wregs.append("%s=%d" % (name, val))
        print("  Warp %d: %s" % (w.warp_id, ', '.join(wregs)))


def print_simt_stack(simt):
    for w in simt.warps:
        print("W%d SIMT Stack (%d):" % (w.warp_id, len(w.simt_stack)))
        for e in w.simt_stack.entries:
            orig_bits = "{0:0{1}b}".format(e.orig_mask, 8)
            taken_bits = "{0:0{1}b}".format(e.taken_mask, 8)
            print("  reconv=%d orig=%s taken=%s" % (
                e.reconv_pc, orig_bits, taken_bits))


def print_operand_collector(simt):
    """Print operand collector bank status"""
    oc = simt.op_collector
    print(c("--- Operand Collector ---", 'bold'))
    print("  Banks: %d" % oc.num_banks)
    for i in range(oc.num_banks):
        label = "BUSY" if oc.bank_busy[i] else "free"
        print("  Bank %d: %s" % (i, label))
    print("  Total reads: %d" % oc.total_reqs)
    print("  Conflicts: %d" % oc.conflict_count)
    print("  Conflict rate: %.1f%%" % (oc.bank_conflict_rate() * 100))


# ---- Console runner ----

def run_console(simt, program_text, args):
    """Start interactive console"""
    warp_size = args.get('warp_size', 4)
    num_warps = args.get('num_warps', 1)
    max_cycles = args.get('max_cycles', 500)
    breakpoints = set()

    # Load program
    prog = assemble(program_text)
    simt.load_program(prog)

    print(c("+-" + "=" * 50 + "-+", 'cyan'))
    print(c("|  toygpgpu Learning Console -- Interactive Debugger (Phase 7)  |", 'cyan'))
    print(c("+-" + "=" * 50 + "-+", 'cyan'))
    print("|  Program: %d instructions" % len(prog))
    print("|  Config:  %d warp(s) x %d threads/warp" % (num_warps, warp_size))
    print("|  Commands: Enter=step, r=run, i=info, q=quit")
    print(c("+-" + "=" * 50 + "-+", 'cyan'))

    # Print program listing
    print()
    print(c("--- Program ---", 'bold'))
    for pc, word in enumerate(prog):
        instr = decode(word)
        opname = OPCODE_NAMES.get(instr.opcode, '?')
        print("  PC %2d: %-6s rd=r%d rs1=r%d rs2=r%d imm=%d" % (
            pc, opname, instr.rd, instr.rs1, instr.rs2, instr.imm))
    print()

    cycle = 0
    prev_regs = snapshot_regs(simt)
    prev_mem = snapshot_mem(simt)
    auto_step = False

    while cycle < max_cycles:
        if auto_step:
            pass
        else:
            try:
                cmd = input(c("[%d] > " % cycle, 'green')).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if cmd == '' or cmd.lower() == 's':
                pass  # single step
            elif cmd.lower() == 'q':
                break
            elif cmd.lower() == 'r':
                auto_step = True
            elif cmd.lower().startswith('r '):
                n = int(cmd.split()[1])
                for _ in range(n):
                    if not _do_step(simt, cycle, prev_regs, prev_mem, breakpoints):
                        break
                    cycle += 1
                prev_regs = snapshot_regs(simt)
                prev_mem = snapshot_mem(simt)
                continue
            elif cmd.lower() == 'i':
                print_state(simt)
                continue
            elif cmd.lower() == 'm':
                print_memory(simt)
                continue
            elif cmd.lower() == 'reg':
                print_registers(simt)
                continue
            elif cmd.lower() == 'wreg':
                print_warp_regs(simt)
                continue
            elif cmd.lower() == 'sb':
                print_scoreboard(simt)
                continue
            elif cmd.lower() == 'ib':
                print_ibuffer(simt)
                continue
            elif cmd.lower() == 'oc':
                print_operand_collector(simt)
                continue
            elif cmd.lower() == 'stack':
                print_simt_stack(simt)
                continue
            elif cmd.lower().startswith('b '):
                sub = cmd[2:].strip()
                if sub == 'list':
                    print("Breakpoints: %s" % (sorted(breakpoints) if breakpoints else 'none'))
                elif sub == 'clear':
                    breakpoints.clear()
                    print("Breakpoints cleared.")
                else:
                    try:
                        breakpoints.add(int(sub))
                        print("Breakpoint set at PC=%s" % sub)
                    except ValueError:
                        print("Invalid PC: %s" % sub)
                continue
            else:
                print("Unknown command: %s" % cmd)
                continue

        # Execute one cycle
        bp_hit = _do_step(simt, cycle, prev_regs, prev_mem, breakpoints)

        prev_regs = snapshot_regs(simt)
        prev_mem = snapshot_mem(simt)
        cycle += 1

        if bp_hit:
            auto_step = False
            print(c("  * Breakpoint hit", 'red'))

        # Check if done
        if not simt.scheduler.has_active_warps():
            print(c("  All warps completed at cycle %d" % cycle, 'green'))
            break

    # Final state
    print()
    print(c("--- Final State ---", 'bold'))
    print("Cycles executed: %d" % cycle)
    print("Memory (non-zero):")
    print_memory(simt)
    print()
    print(c("L1Cache", 'cyan') + ": %s" % simt.l1_cache.stats())
    print(c("OpCollector", 'cyan') + ": %s" % simt.op_collector.stats())


def _do_step(simt, cycle, prev_regs, prev_mem, breakpoints):
    """Execute one cycle and render display

    Returns:
        True if breakpoint hit
    """
    # Record state before execution
    old_regs = snapshot_regs(simt)
    old_mem = snapshot_mem(simt)

    # Collect pipeline info before step
    warp = None
    instr = None
    stage_info = {'fetch': '', 'decode': '', 'issue': '', 'exec': '', 'wb': ''}

    # Try to find instruction about to be issued
    for w in simt.warps:
        if not w.done and not w.scoreboard_stalled and not w.at_barrier:
            for e in w.ibuffer.entries:
                if e.valid and e.ready:
                    instr = decode(e.instruction_word)
                    warp = w
                    opname = OPCODE_NAMES.get(instr.opcode, '?')
                    stage_info['fetch'] = "PC=%d -> IBuffer" % e.pc
                    stage_info['decode'] = "%s decoded" % opname
                    stage_info['issue'] = "Scoreboard check, bank check"
                    stage_info['exec'] = "%s r%d=r%d op r%d" % (
                        opname, instr.rd, instr.rs1, instr.rs2)
                    stage_info['wb'] = "r%d <- result (latency)" % instr.rd
                    break
            if instr:
                break

    # Execute one cycle
    simt.step()

    # Calculate changes
    new_regs = snapshot_regs(simt)
    new_mem = snapshot_mem(simt)
    changes = mem_diff(old_mem, new_mem)

    # Format instruction info
    instr_info = None
    if instr:
        instr_info = {
            'op': OPCODE_NAMES.get(instr.opcode, '?'),
            'rd': instr.rd, 'rs1': instr.rs1, 'rs2': instr.rs2,
            'pc': warp.pc - 1 if warp else -1,
            'active': bin(warp.active_mask).count('1') if warp else 0,
            'imm': instr.imm,
        }

    # Render
    print(render_cycle(cycle, simt, instr_info, old_regs, changes, stage_info))

    # Check breakpoints
    if warp and warp.pc - 1 in breakpoints:
        return True
    return False


# ---- Main ----

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options: --warp-size N --num-warps N --max-cycles N")
        sys.exit(1)

    asm_file = sys.argv[1]
    args = {
        'warp_size': 4, 'num_warps': 1,
        'max_cycles': 500,
        'auto': False,
    }

    # Parse options
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--warp-size':
            args['warp_size'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--num-warps':
            args['num_warps'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--max-cycles':
            args['max_cycles'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--auto':
            args['auto'] = True
            i += 1
        elif sys.argv[i] == '--trace':
            args['auto'] = True
            i += 1
        else:
            i += 1

    with open(asm_file, encoding='utf-8') as f:
        program_text = f.read()

    simt = SIMTCore(
        warp_size=args['warp_size'],
        num_warps=args['num_warps'],
        memory_size=1024
    )

    if args.get('auto'):
        prog = assemble(program_text)
        simt.load_program(prog)
        simt.run(trace=True)
        return

    run_console(simt, program_text, args)


if __name__ == '__main__':
    main()
