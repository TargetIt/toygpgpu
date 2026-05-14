#!/usr/bin/env python3
"""
Learning Console -- Interactive SIMT learning console for Phase 2.
=================================================================
Step through SIMT core execution cycle by cycle, observing warp and thread state.

Usage:
  python3 learning_console.py <program.asm> [options]

Options:
  --warp-size N     Threads per warp (default 8)
  --num-warps N     Number of warps (default 2)
  --max-cycles N    Maximum cycles to execute (default 500)

Interactive commands:
  Enter / s   Single-step one cycle
  r           Run to HALT
  r N         Run N cycles
  i           Print current state (all warps + mem)
  m           Print non-zero memory
  reg         Print all thread registers (non-zero)
  w <id>      Show details for a specific warp
  q           Quit
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simt_core import SIMTCore
from assembler import assemble
from isa import OPCODE_NAMES, decode
from warp import Warp


def print_banner(simt, program):
    """Print the startup banner with program listing."""
    sep = "+" + "-" * 58 + "+"
    print(sep)
    print("|     toygpgpu Learning Console -- Phase 2 (SIMT)             |")
    print(sep)
    print(f"|  Program: {len(program)} instructions                           |")
    print(f"|  Config:  {len(simt.warps)} warp(s) x {simt.warp_size} threads/warp           |")
    print(f"|  Commands: Enter=step, r=run, i=info, w<id>, q=quit        |")
    print(sep)
    print()
    print("--- Program ---")
    for pc, word in enumerate(program):
        instr = decode(word)
        opname = OPCODE_NAMES.get(instr.opcode, "?")
        parts = []
        parts.append(f"{opname:6s}")
        parts.append(f"rd=r{instr.rd}")
        parts.append(f"rs1=r{instr.rs1}")
        parts.append(f"rs2=r{instr.rs2}")
        parts.append(f"imm={instr.imm}")
        print(f"  PC {pc:2d}: {' '.join(parts)}")
    print()


def snapshot_warp_regs(simt):
    """Return a snapshot of all warp thread registers: {wid: {tid: {reg: val}}}."""
    snap = {}
    for w in simt.warps:
        wsnap = {}
        for t in w.threads:
            tregs = {}
            for i in range(16):
                tregs[i] = t.read_reg(i)
            wsnap[t.thread_id] = tregs
        snap[w.warp_id] = wsnap
    return snap


def snapshot_mem(simt):
    """Return a snapshot dict of non-zero memory words."""
    mem = {}
    for i in range(min(256, simt.memory.size_words)):
        v = simt.memory.read_word(i)
        if v != 0:
            mem[i] = v
    return mem


def mem_diff(old_mem, new_mem):
    """Return memory changes between two snapshots."""
    changes = {}
    all_keys = set(old_mem.keys()) | set(new_mem.keys())
    for k in all_keys:
        old_v = old_mem.get(k, 0)
        new_v = new_mem.get(k, 0)
        if old_v != new_v:
            changes[k] = (old_v, new_v)
    return changes


def print_state(simt):
    """Print current state summary for all warps."""
    print("--- Current State ---")
    for w in simt.warps:
        status = "DONE" if w.done else ("BARRIER" if w.at_barrier else "ACTIVE")
        active_count = bin(w.active_mask).count("1")
        print(f"W{w.warp_id}: PC={w.pc}, mask=0b{w.active_mask:0{w.warp_size}b}, "
              f"active={active_count}/{w.warp_size}, {status}")
    print(f"Total instructions: {simt.instr_count}")


def print_memory(simt):
    """Print non-zero memory words."""
    non_zero = []
    for i in range(min(256, simt.memory.size_words)):
        v = simt.memory.read_word(i)
        if v != 0:
            non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
    if non_zero:
        for line in non_zero[:30]:
            print("  " + line)
    else:
        print("  (all zero)")


def print_registers(simt):
    """Print all thread registers (non-zero) for all warps."""
    for w in simt.warps:
        print(f"Warp {w.warp_id}:")
        for t in w.threads:
            vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
            if vals:
                reg_str = " ".join(f"r{i}={v}" for i, v in vals)
                print(f"  T{t.thread_id}: {reg_str}")
            else:
                print(f"  T{t.thread_id}: (all zero)")


def print_warp_detail(simt, warp_id):
    """Print detailed state for a specific warp."""
    if warp_id < 0 or warp_id >= len(simt.warps):
        print(f"Invalid warp ID: {warp_id}. Valid: 0-{len(simt.warps) - 1}")
        return
    w = simt.warps[warp_id]
    status = "DONE" if w.done else ("BARRIER" if w.at_barrier else "ACTIVE")
    active_count = bin(w.active_mask).count("1")
    print(f"Warp {w.warp_id}: PC={w.pc}, mask=0b{w.active_mask:0{w.warp_size}b}, "
          f"active={active_count}/{w.warp_size}, {status}")
    print(f"  Threads:")
    for t in w.threads:
        active = "A" if w.is_active(t.thread_id) else "I"
        vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
        if vals:
            reg_str = " ".join(f"r{i}={v}" for i, v in vals)
            print(f"    [{active}] T{t.thread_id}: {reg_str}")
        else:
            print(f"    [{active}] T{t.thread_id}: (all zero)")


def do_step(simt, cycle):
    """Execute one cycle and print the step info.

    Returns True if the core is still running, False if all warps done.
    """
    old_regs = snapshot_warp_regs(simt)
    old_mem = snapshot_mem(simt)

    # Collect info about which warp and instruction will execute next
    selected_warp = simt.scheduler.select_warp()
    # The scheduler already advanced current_idx, so we need to identify
    # which warp was selected by checking the last advance.
    # Actually, select_warp() returns the warp and advances. Let's track it.
    # We need to determine the warp that would have been selected.
    # Since select_warp() returns the warp or None, we can use that.
    # But we already consumed it! Let's re-find it without consuming.
    # Actually, let's just look at simt.scheduler state.
    # The scheduler advanced its current_idx. The last returned warp
    # is the one at (current_idx - 1) % n if one was returned.
    # Let's just find the warp by inspecting scheduler state before stepping.

    # Simpler approach: find the warp that will execute by looking at
    # what the scheduler would select. We need to not consume it.
    # Let's peek without modifying state.
    warp_to_execute = None
    instr_to_execute = None
    num_warps = len(simt.warps)
    start_idx = simt.scheduler.current_idx
    for off in range(num_warps):
        idx = (start_idx + off) % num_warps
        w = simt.warps[idx]
        if not w.done and not w.at_barrier:
            if 0 <= w.pc < len(simt.program):
                warp_to_execute = w
                raw_word = simt.program[w.pc]
                instr_to_execute = decode(raw_word)
            break

    # Execute one cycle
    running = simt.step()

    new_regs = snapshot_warp_regs(simt)
    new_mem = snapshot_mem(simt)
    mem_changes = mem_diff(old_mem, new_mem)

    if warp_to_execute is not None and instr_to_execute is not None:
        w = warp_to_execute
        instr = instr_to_execute
        opname = OPCODE_NAMES.get(instr.opcode, "?")
        active_count = bin(w.active_mask).count("1")

        # Determine per-thread register changes
        reg_change_strs = []
        for t in w.threads:
            if not w.is_active(t.thread_id):
                continue
            tid = t.thread_id
            changes = []
            for ri in range(16):
                old_v = old_regs[w.warp_id][tid][ri]
                new_v = new_regs[w.warp_id][tid][ri]
                if old_v != new_v:
                    changes.append(f"r{ri}:{old_v}->{new_v}")
            if changes:
                reg_change_strs.append(f"T{tid}: " + " ".join(changes))

        mem_str = ""
        if mem_changes:
            parts = []
            for addr, (old_v, new_v) in sorted(mem_changes.items()):
                parts.append(f"mem[{addr}]:{old_v}->{new_v}")
            mem_str = " | " + ", ".join(parts)

        reg_summary = "; ".join(reg_change_strs[:4])  # Limit verbosity
        if len(reg_change_strs) > 4:
            reg_summary += f" ... (+{len(reg_change_strs) - 4} threads)"
        if not reg_change_strs:
            reg_summary = "no reg change on active threads"

        print(f"Cycle {cycle}: PC={w.pc - 1} {opname} "
              f"rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} "
              f"imm={instr.imm} "
              f"| active={active_count} threads | [{','.join(reg_change_strs[:2])}]{mem_str}")
    else:
        print(f"Cycle {cycle}: (no warp selected / all warps done)")

    return running


def run_console(simt, program_text, args):
    """Launch the interactive console."""
    max_cycles = args.get("max_cycles", 500)

    prog = assemble(program_text)
    simt.load_program(prog)

    print_banner(simt, prog)

    cycle = 0
    auto_step = False

    while cycle < max_cycles:
        if auto_step:
            pass
        else:
            try:
                cmd = input(f"[{cycle}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd == "" or cmd.lower() == "s":
                pass
            elif cmd.lower() == "q":
                break
            elif cmd.lower() == "r":
                auto_step = True
            elif cmd.lower().startswith("r "):
                n = int(cmd.split()[1])
                for _ in range(n):
                    if not do_step(simt, cycle):
                        break
                    cycle += 1
                continue
            elif cmd.lower() == "i":
                print_state(simt)
                print_memory(simt)
                continue
            elif cmd.lower() == "m":
                print_memory(simt)
                continue
            elif cmd.lower() == "reg":
                print_registers(simt)
                continue
            elif cmd.lower().startswith("w"):
                parts = cmd.split()
                if len(parts) > 1:
                    try:
                        wid = int(parts[1])
                        print_warp_detail(simt, wid)
                    except ValueError:
                        print(f"Invalid warp id: {parts[1]}")
                else:
                    # Show summary of all warps
                    print_state(simt)
                continue
            else:
                print(f"Unknown command: {cmd}")
                print("Commands: Enter/s=step, r=run, r N=run N, i=info, "
                      "m=memory, reg=registers, w <id>=warp detail, q=quit")
                continue

        # Execute one cycle
        running = do_step(simt, cycle)
        cycle += 1

        if not running:
            print(f"\n  [OK] All warps completed at cycle {cycle}")
            break

        if auto_step:
            if cycle % 10 == 0:
                time.sleep(0.05)

    # Final state
    print(f"\n--- Final State ---")
    print(f"Cycles executed: {cycle}")
    print(f"Total instructions: {simt.instr_count}")
    print(f"Memory (non-zero):")
    print_memory(simt)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options: --warp-size N --num-warps N --max-cycles N")
        sys.exit(1)

    asm_file = sys.argv[1]
    args = {
        "warp_size": 8,
        "num_warps": 2,
        "max_cycles": 500,
        'auto': False,
    }

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--warp-size":
            args["warp_size"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--num-warps":
            args["num_warps"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max-cycles":
            args["max_cycles"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--auto':
            args['auto'] = True
            i += 1
        elif sys.argv[i] == '--trace':
            args['auto'] = True
            i += 1
        else:
            i += 1

    with open(asm_file, encoding="utf-8") as f:
        program_text = f.read()

    simt = SIMTCore(
        warp_size=args["warp_size"],
        num_warps=args["num_warps"],
        memory_size=1024,
    )

    # Auto mode (--auto or --trace)
    if args.get('auto'):
        prog = assemble(program_text)
        simt.load_program(prog)
        cycle = 0
        cycle_limit = args.get('max_cycles', 500)
        simt._t_regs = simt._snapshot_regs()
        simt._t_mem = simt._snapshot_mem()
        simt._t_pcs = {w.warp_id: w.pc for w in simt.warps}
        simt._t_masks = {w.warp_id: w.active_mask for w in simt.warps}
        print(f"Auto-run with trace (max {cycle_limit} cycles)")
        print(f"Program: {len(prog)} instructions, "
              f"Config: {args.get('num_warps', 1)} warp(s) x {args.get('warp_size', 4)} threads/warp")
        print()
        while cycle < cycle_limit:
            running = simt.step()
            if not running:
                break
            simt._trace_step(cycle)
            cycle += 1
        print(f"\n[Summary] {cycle} cycles, {simt.instr_count} instructions")
        if not simt.scheduler.has_active_warps():
            print("[All warps completed]")
        print()
        for w in simt.warps:
            print(f"Warp {w.warp_id}: PC={w.pc}, mask=0b{w.active_mask:0{simt.warp_size}b}, done={w.done}")
        non_zero = []
        for i in range(min(256, simt.memory.size_words)):
            v = simt.memory.read_word(i)
            if v != 0:
                non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
        if non_zero:
            print("Memory (non-zero):")
            for line in non_zero[:20]:
                print(f"  {line}")
        return

    run_console(simt, program_text, args)


if __name__ == "__main__":
    main()
