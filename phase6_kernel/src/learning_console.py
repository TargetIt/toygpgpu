#!/usr/bin/env python3
"""
Learning Console - Interactive GPU Pipeline Debugger (Phase 6: Kernel)
========================================================================
Step-by-step inspection with multi-block kernel launch and performance
counters.

Usage:
  python3 learning_console.py <program.asm> [options]

Options:
  --warp-size N      Warp size in threads (default: 4)
  --num-warps N      Warps per block (default: 1)
  --grid-dim N       Grid dimension (number of blocks, default: 1)
  --block-dim N      Block dimension (threads per block, default: 4)
  --max-cycles N     Maximum cycles before forced halt (default: 500)

Interactive Commands:
  Enter / s    Step one cycle
  r            Run to completion
  i            Print current full state
  m            Print non-zero global memory
  reg          Print all registers per thread
  stack        Print SIMT stack entries per warp
  sb           Print scoreboard status per warp
  cache        Print L1 cache statistics and contents
  smem         Print non-zero shared memory
  perf         Print performance counters
  q            Quit
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_sim import GPUSim
from assembler import assemble
from isa import decode, OPCODE_NAMES, OP_JMP, OP_BEQ, OP_BNE, OP_SETP


def c(text, style=''):
    """No-op color wrapper for ASCII-safe output."""
    return text


def snapshot_regs(core):
    """Capture all non-zero register values for one core.

    Returns list of (warp_id, thread_id, reg_id, value).
    """
    snap = []
    for w in core.warps:
        for t in w.threads:
            for i in range(16):
                v = t.read_reg(i)
                if v != 0:
                    snap.append((w.warp_id, t.thread_id, i, v))
    return snap


def snapshot_mem(core):
    """Capture all non-zero memory values as {addr: value} for one core."""
    snap = {}
    for i in range(core.memory.size_words):
        v = core.memory.read_word(i)
        if v != 0:
            snap[i] = v
    return snap


def mem_diff(old, new):
    """Return {addr: (old_val, new_val)} for changed memory entries."""
    diff = {}
    all_addrs = set(old.keys()) | set(new.keys())
    for a in all_addrs:
        old_v = old.get(a, 0)
        new_v = new.get(a, 0)
        if old_v != new_v:
            diff[a] = (old_v, new_v)
    return diff


def regs_diff(old, new):
    """Return list of changed registers as (w, t, r, old_v, new_v)."""
    old_set = set((w, t, r) for (w, t, r, v) in old)
    new_set = set((w, t, r) for (w, t, r, v) in new)
    old_map = {(w, t, r): v for (w, t, r, v) in old}
    new_map = {(w, t, r): v for (w, t, r, v) in new}
    diffs = []
    for key in old_set | new_set:
        ov = old_map.get(key, 0)
        nv = new_map.get(key, 0)
        if ov != nv:
            diffs.append((key[0], key[1], key[2], ov, nv))
    return diffs


def predict_next_per_core(core):
    """Predict which warp will execute next on this core.

    Returns (warp_or_None, instr_info_or_None, is_stalled).
    """
    sched = core.scheduler
    num_warps = len(core.warps)
    idx = sched.current_idx

    for _ in range(num_warps):
        w = core.warps[idx]
        idx = (idx + 1) % num_warps
        stalled = getattr(w, 'scoreboard_stalled', False)
        if not w.done and not w.at_barrier and not stalled:
            if w.pc < len(core.program):
                raw_word = core.program[w.pc]
                instr = decode(raw_word)
                opname = OPCODE_NAMES.get(instr.opcode, '?')
                info = {
                    'warp_id': w.warp_id,
                    'op': opname,
                    'rd': instr.rd,
                    'rs1': instr.rs1,
                    'rs2': instr.rs2,
                    'imm': instr.imm,
                    'pc': w.pc,
                    'active_mask': w.active_mask,
                    'asm': '',
                }
                # Build asm text
                if instr.opcode == OP_JMP:
                    target = w.pc + 1 + instr.imm
                    info['asm'] = f"-> PC{target}"
                elif instr.opcode in (OP_BEQ, OP_BNE):
                    target = w.pc + 1 + instr.imm
                    info['asm'] = f"r{instr.rs1},r{instr.rs2} -> PC{target}"
                elif instr.opcode == OP_SETP:
                    mode = "EQ" if (instr.imm & 1) == 0 else "NE"
                    info['asm'] = f"r{instr.rs1},r{instr.rs2} ({mode})"
                elif instr.opcode == 0x31:
                    info['asm'] = f"r{instr.rd}, [{instr.imm & 0xFF}]"
                elif instr.opcode == 0x32:
                    info['asm'] = f"r{instr.rs1}, [{instr.imm & 0xFF}]"
                return w, info, False

    for _ in range(num_warps):
        w = core.warps[idx]
        idx = (idx + 1) % num_warps
        stalled = getattr(w, 'scoreboard_stalled', False)
        if not w.done and not w.at_barrier and stalled:
            return w, None, True

    return None, None, False


def step_all_cores(gpu):
    """Advance all cores by one cycle. Returns True if any warp is active."""
    any_active = False
    for core in gpu.cores:
        core.step()
        if core.scheduler.has_active_warps():
            any_active = True
    return any_active


def print_state(gpu):
    """Print current state summary for all blocks/cores."""
    print(c("=== Current State ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}:")
        for w in core.warps:
            status = 'DONE' if w.done else ('BARRIER' if w.at_barrier else 'ACTIVE')
            stall_tag = ' STALLED' if getattr(w, 'scoreboard_stalled', False) else ''
            mask_str = bin(w.active_mask)[2:].zfill(core.warp_size)
            print(f"    Warp {w.warp_id}: PC={w.pc}, mask=0b{mask_str}, "
                  f"{status}{stall_tag}")
            print(f"      Scoreboard: {w.scoreboard}")
            if not w.simt_stack.empty:
                print(f"      SIMT Stack depth: {len(w.simt_stack)}")
        print(f"    L1 Cache: {core.l1_cache.stats()}")
        if core.total_mem_reqs > 0:
            eff = core.coalesce_count / core.total_mem_reqs * 100
            print(f"    Coalescing: {core.coalesce_count}/{core.total_mem_reqs} "
                  f"({eff:.0f}%)")
    print()


def print_memory(gpu):
    """Print non-zero memory values (from block 0, or all if different)."""
    print(c("=== Global Memory (non-zero) ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}:")
        non_zero = []
        for i in range(min(256, core.memory.size_words)):
            v = core.memory.read_word(i)
            if v != 0:
                non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
        if non_zero:
            for line in non_zero[:15]:
                print(f"    {line}")
            if len(non_zero) > 15:
                print(f"    ... and {len(non_zero) - 15} more entries")
        else:
            print("    (all zero)")
    print()


def print_registers(gpu):
    """Print per-thread non-zero register values for all blocks."""
    print(c("=== Registers ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}:")
        for w in core.warps:
            for t in w.threads:
                vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
                if vals:
                    reg_str = ', '.join(f"r{i}={v}" for i, v in vals)
                    print(f"    W{w.warp_id} T{t.thread_id}: {reg_str}")
    print()


def print_warp_regs(gpu):
    """Print warp-level uniform registers for all blocks."""
    from isa import WREG_NAMES
    rev = {v: k for k, v in WREG_NAMES.items()}
    print(c("=== Warp Registers ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        for w in core.warps:
            print(f"  Block {bid} Warp {w.warp_id}:")
            for idx, val in sorted(w.warp_regs.items()):
                name = rev.get(idx, f"wreg[{idx}]")
                print(f"    {name} = {val}")
    print()


def print_simt_stack(gpu):
    """Print SIMT stack entries for all blocks."""
    print(c("=== SIMT Stack ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        for w in core.warps:
            print(f"  Block {bid} Warp {w.warp_id} (depth={len(w.simt_stack)}):")
            if w.simt_stack.empty:
                print("    (empty)")
            else:
                ws = core.warp_size
                for i, e in enumerate(w.simt_stack.entries):
                    orig = bin(e.orig_mask)[2:].zfill(ws)
                    taken = bin(e.taken_mask)[2:].zfill(ws)
                    print(f"    [{i}] reconv=PC{e.reconv_pc} "
                          f"orig=0b{orig} taken=0b{taken} "
                          f"fallthrough=PC{e.fallthrough_pc}")
    print()


def print_scoreboard(gpu):
    """Print scoreboard status for all blocks."""
    print(c("=== Scoreboard ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        for w in core.warps:
            stalled = getattr(w, 'scoreboard_stalled', False)
            print(f"  Block {bid} Warp {w.warp_id}: {w.scoreboard}")
            if stalled:
                print(f"    -> STALLED (scoreboard hazard)")
            if w.scoreboard.reserved:
                for reg_id, rem in w.scoreboard.reserved.items():
                    print(f"    r{reg_id}: {rem} cycles remaining")
    print()


def print_cache_stats(gpu):
    """Print L1 cache statistics for all blocks."""
    print(c("=== L1 Cache ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}: {core.l1_cache.stats()}")
        if core.total_mem_reqs > 0:
            eff = core.coalesce_count / core.total_mem_reqs * 100
            print(f"    Coalescing: {core.coalesce_count}/{core.total_mem_reqs} "
                  f"({eff:.0f}%)")
        found = False
        for i, line in enumerate(core.l1_cache.lines):
            if line.valid:
                found = True
                data_str = ' '.join(f"0x{v:08X}" for v in line.data)
                print(f"    Line {i:2d}: tag={line.tag} dirty={line.dirty} "
                      f"[{data_str}]")
        if not found:
            print(f"    (all invalid)")
    print()


def print_shared_memory(gpu):
    """Print non-zero shared memory values for all blocks."""
    print(c("=== Shared Memory ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}:")
        smem = core.thread_block.shared_memory
        non_zero = []
        for i in range(smem.size_words):
            v = smem.read_word(i)
            if v != 0:
                non_zero.append(f"smem[{i:3d}]=0x{v:08X}({v})")
        if non_zero:
            for line in non_zero[:15]:
                print(f"    {line}")
            if len(non_zero) > 15:
                print(f"    ... and {len(non_zero) - 15} more entries")
        else:
            print("    (all zero)")
    print()


def print_perf(gpu, cycle, total_instrs, stall_cycles):
    """Print performance counters."""
    print(c("=== Performance Counters ===", 'bold'))
    ipc = total_instrs / cycle if cycle > 0 else 0.0
    active_pct = (cycle - stall_cycles) / cycle * 100 if cycle > 0 else 0.0
    print(f"  Total cycles:       {cycle}")
    print(f"  Total instructions: {total_instrs}")
    print(f"  IPC:                {ipc:.3f}")
    print(f"  Active cycles:      {cycle - stall_cycles} ({active_pct:.1f}%)")
    print(f"  Stall cycles:       {stall_cycles}")

    # Per-block cache stats
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}: {core.l1_cache.stats()}")
        if core.total_mem_reqs > 0:
            eff = core.coalesce_count / core.total_mem_reqs * 100
            print(f"    Coalescing: {core.coalesce_count}/{core.total_mem_reqs} "
                  f"({eff:.0f}%)")
    print()


def run_console(gpu, program_text, args):
    """Start interactive console session."""
    warp_size = args.get('warp_size', 4)
    num_warps = args.get('num_warps', 1)
    max_cycles = args.get('max_cycles', 500)
    grid_dim = args.get('grid_dim', (1,))
    block_dim = args.get('block_dim', (warp_size,))

    prog = assemble(program_text)

    # Launch kernel
    gpu.launch_kernel(prog, grid_dim, block_dim)

    total_blocks = 1
    for d in grid_dim:
        total_blocks *= d

    # Header
    print("==============================================")
    print("  toygpgpu Learning Console (Phase 6)")
    print("  Multi-Block Kernel / Performance Monitor")
    print("==============================================")
    print(f"  Program: {len(prog)} instructions")
    print(f"  Grid:    {total_blocks} block(s)")
    print(f"  Block:   {block_dim} thread(s), {num_warps} warp(s) x "
          f"{warp_size} threads/warp")
    print(f"  Total:   {total_blocks * num_warps * warp_size} threads")
    print(f"  Commands: Enter=step, r=run, i=info, q=quit, perf")
    print("==============================================")
    print()

    # Print program
    print(c("=== Program ===", 'bold'))
    for pc, word in enumerate(prog):
        instr = decode(word)
        opname = OPCODE_NAMES.get(instr.opcode, '?')
        rd = instr.rd
        rs1 = instr.rs1
        rs2 = instr.rs2
        imm = instr.imm
        if instr.opcode == OP_JMP:
            target = pc + 1 + imm
            print(f"  PC {pc:2d}: {opname:6s} -> PC{target}")
        elif instr.opcode in (OP_BEQ, OP_BNE):
            target = pc + 1 + imm
            print(f"  PC {pc:2d}: {opname:6s} r{rs1},r{rs2} -> PC{target}")
        elif instr.opcode == OP_SETP:
            mode = "EQ" if (imm & 1) == 0 else "NE"
            print(f"  PC {pc:2d}: {opname:6s} r{rs1},{rs2} ({mode})")
        elif instr.opcode in (0x21, 0x22):
            print(f"  PC {pc:2d}: {opname:6s} r{rd}")
        else:
            print(f"  PC {pc:2d}: {opname:6s} rd=r{rd} rs1=r{rs1} "
                  f"rs2=r{rs2} imm={imm}")
    print()

    cycle = 0
    total_instrs = 0
    stall_cycles = 0
    prev_total_instrs = 0
    auto_run = False

    # Track per-core pre-step state
    old_regs_per_core = {}
    old_mem_per_core = {}
    old_cache_hits_per_core = {}
    old_cache_misses_per_core = {}

    for core in gpu.cores:
        old_regs_per_core[id(core)] = snapshot_regs(core)
        old_mem_per_core[id(core)] = snapshot_mem(core)
        old_cache_hits_per_core[id(core)] = core.l1_cache.hits
        old_cache_misses_per_core[id(core)] = core.l1_cache.misses

    while cycle < max_cycles:
        # Check if all blocks finished
        all_done = all(
            not core.scheduler.has_active_warps()
            for core in gpu.cores
        )
        if all_done:
            print(f"\n[DONE] All blocks completed at cycle {cycle}")
            break

        if auto_run:
            pass
        else:
            try:
                cmd = input(f"[{cycle}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd == '' or cmd.lower() == 's':
                pass
            elif cmd.lower() == 'q':
                break
            elif cmd.lower() == 'r':
                auto_run = True
            elif cmd.lower() == 'i':
                print_state(gpu)
                continue
            elif cmd.lower() == 'm':
                print_memory(gpu)
                continue
            elif cmd.lower() == 'reg':
                print_registers(gpu)
                continue
            elif cmd.lower() == 'wreg':
                print_warp_regs(gpu)
                continue
            elif cmd.lower() == 'stack':
                print_simt_stack(gpu)
                continue
            elif cmd.lower() == 'sb':
                print_scoreboard(gpu)
                continue
            elif cmd.lower() == 'cache':
                print_cache_stats(gpu)
                continue
            elif cmd.lower() == 'smem':
                print_shared_memory(gpu)
                continue
            elif cmd.lower() == 'perf':
                print_perf(gpu, cycle, total_instrs, stall_cycles)
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue

        # --- Execute one cycle across all cores ---
        cycle += 1
        this_cycle_instrs = 0
        this_cycle_stalled = 0

        for core in gpu.cores:
            # Skip if this core is done
            if not core.scheduler.has_active_warps():
                continue

            # Pre-decode for display
            warp, info, stalled = predict_next_per_core(core)

            # Execute
            had_stall = False
            if stalled:
                had_stall = True
            else:
                # Save pre-state
                core_old_regs = old_regs_per_core.get(id(core), [])
                core_old_mem = old_mem_per_core.get(id(core), {})
                core_old_ch = old_cache_hits_per_core.get(id(core), 0)
                core_old_cm = old_cache_misses_per_core.get(id(core), 0)

                core.step()

                # Get post-state
                core_new_regs = snapshot_regs(core)
                core_new_mem = snapshot_mem(core)
                core_new_ch = core.l1_cache.hits
                core_new_cm = core.l1_cache.misses

                instr_count_before = total_instrs - this_cycle_instrs

                # Display
                block_id = gpu.cores.index(core)
                if info:
                    wid = info['warp_id']
                    op = info['op']
                    pc = info['pc']
                    mask = info['active_mask']
                    mask_str = bin(mask)[2:].zfill(core.warp_size)
                    aname = info.get('asm', '')
                    rd = info['rd']
                    rs1 = info['rs1']
                    rs2 = info['rs2']

                    if aname:
                        print(f"[{cycle}] Block {block_id} W{wid}: "
                              f"{op} {aname}   | PC={pc} active=0b{mask_str}")
                    else:
                        print(f"[{cycle}] Block {block_id} W{wid}: "
                              f"{op} rd=r{rd} rs1=r{rs1} rs2=r{rs2}   "
                              f"| PC={pc} active=0b{mask_str}")

                    # Register changes
                    reg_diffs = regs_diff(core_old_regs, core_new_regs)
                    if reg_diffs:
                        for w, t, r, ov, nv in reg_diffs[:5]:
                            print(f"           Reg W{w} T{t}: r{r} {ov} -> {nv}")
                        if len(reg_diffs) > 5:
                            print(f"           ... and {len(reg_diffs) - 5} more reg changes")

                    # Memory changes
                    mem_changes = mem_diff(core_old_mem, core_new_mem)
                    if mem_changes:
                        for addr, (ov, nv) in sorted(mem_changes.items())[:5]:
                            print(f"           Mem mem[{addr}] = {nv}")
                        if len(mem_changes) > 5:
                            print(f"           ... and {len(mem_changes) - 5} more mem changes")

                    # Cache delta
                    if core_new_ch != core_old_ch or core_new_cm != core_old_cm:
                        dh = core_new_ch - core_old_ch
                        dm = core_new_cm - core_old_cm
                        parts = []
                        if dh > 0:
                            parts.append(f"+{dh} hit")
                        if dm > 0:
                            parts.append(f"+{dm} miss")
                        if parts:
                            print(f"           Cache: {', '.join(parts)}")
                else:
                    print(f"[{cycle}] Block {gpu.cores.index(core)}: "
                          f"(no active warp)")

                # Update old state
                old_regs_per_core[id(core)] = core_new_regs
                old_mem_per_core[id(core)] = core_new_mem
                old_cache_hits_per_core[id(core)] = core_new_ch
                old_cache_misses_per_core[id(core)] = core_new_cm

                this_cycle_instrs += 1

            if had_stall:
                this_cycle_stalled += 1
        # End per-core loop

        total_instrs += this_cycle_instrs
        if this_cycle_stalled > 0 or this_cycle_instrs == 0:
            stall_cycles += 1

        # Check if all done
        all_done = all(
            not core.scheduler.has_active_warps()
            for core in gpu.cores
        )
        if all_done:
            print(f"\n[DONE] All blocks completed at cycle {cycle}")
            break

        if auto_run and all_done:
            break

    # --- Final summary ---
    print(f"\n{'='*46}")
    print(f"  Final State (after {cycle} cycles)")
    print(f"{'='*46}")
    print(f"Total instructions: {total_instrs}")

    print_state(gpu)
    print(c("=== Global Memory ===", 'bold'))
    for bid, core in enumerate(gpu.cores):
        print(f"  Block {bid}:")
        non_zero = []
        for i in range(min(256, core.memory.size_words)):
            v = core.memory.read_word(i)
            if v != 0:
                non_zero.append(f"mem[{i}]: {v}")
        if non_zero:
            for line in non_zero[:15]:
                print(f"    {line}")
        else:
            print("    (all zero)")
    print()

    print_registers(gpu)
    print_perf(gpu, cycle, total_instrs, stall_cycles)

    print(f"{'='*46}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options:")
        print("  --warp-size N   Warp size in threads (default: 4)")
        print("  --num-warps N   Warps per block (default: 1)")
        print("  --grid-dim N    Grid dimension / number of blocks (default: 1)")
        print("  --block-dim N   Threads per block (default: warp_size)")
        print("  --max-cycles N  Max cycles before halt (default: 500)")
        sys.exit(1)

    asm_file = sys.argv[1]
    args = {
        'warp_size': 4,
        'num_warps': 1,
        'grid_dim': (1,),
        'block_dim': None,
        'max_cycles': 500,
        'auto': False,
    }

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--warp-size':
            args['warp_size'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--num-warps':
            args['num_warps'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--grid-dim':
            args['grid_dim'] = (int(sys.argv[i + 1]),)
            i += 2
        elif sys.argv[i] == '--block-dim':
            args['block_dim'] = (int(sys.argv[i + 1]),)
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

    if args['block_dim'] is None:
        args['block_dim'] = (args['warp_size'],)

    with open(asm_file, encoding='utf-8') as f:
        program_text = f.read()

    gpu = GPUSim(
        num_sms=1,
        warp_size=args['warp_size'],
        memory_size=1024,
    )

    # Auto mode (--auto or --trace)
    if args.get('auto'):
        prog = assemble(program_text)
        gpu.launch_kernel(prog, args['grid_dim'], args['block_dim'])
        gpu.run()
        print("Auto-run completed.")
        gpu.report()
        return

    # Batch mode via env var
    if os.environ.get('LEARN_CONSOLE_BATCH'):
        prog = assemble(program_text)
        gpu.launch_kernel(prog, args['grid_dim'], args['block_dim'])
        gpu.run()
        gpu.report()
        print(f"Memory (non-zero):")
        for core in gpu.cores:
            for i in range(core.memory.size_words):
                v = core.memory.read_word(i)
                if v != 0:
                    print(f"  mem[{i}]: {v}")
        return

    run_console(gpu, program_text, args)


if __name__ == '__main__':
    main()
