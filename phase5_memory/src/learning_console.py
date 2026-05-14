#!/usr/bin/env python3
"""
Learning Console - Interactive GPU Pipeline Debugger (Phase 5: Memory)
========================================================================
Step-by-step inspection with L1 cache, shared memory, and coalescing.

Usage:
  python3 learning_console.py <program.asm> [options]

Options:
  --warp-size N      Warp size in threads (default: 4)
  --num-warps N      Number of warps (default: 1)
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
  q            Quit
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simt_core import SIMTCore
from assembler import assemble
from isa import (decode, OPCODE_NAMES,
                 OP_JMP, OP_BEQ, OP_BNE, OP_SETP,
                 OP_SHLD, OP_SHST)
from simt_stack import SIMTStackEntry
from scoreboard import Scoreboard
from cache import L1Cache
from shared_memory import SharedMemory
from thread_block import ThreadBlock


def c(text, style=''):
    """No-op color wrapper for ASCII-safe output."""
    return text


def snapshot_regs(simt):
    """Capture all non-zero register values.

    Returns list of (warp_id, thread_id, reg_id, value).
    """
    snap = []
    for w in simt.warps:
        for t in w.threads:
            for i in range(16):
                v = t.read_reg(i)
                if v != 0:
                    snap.append((w.warp_id, t.thread_id, i, v))
    return snap


def snapshot_mem(simt):
    """Capture all non-zero global memory values as {addr: value}."""
    snap = {}
    for i in range(simt.memory.size_words):
        v = simt.memory.read_word(i)
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


def print_state(simt):
    """Print current state summary for all warps."""
    print(c("=== Current State ===", 'bold'))
    for w in simt.warps:
        status = 'DONE' if w.done else ('BARRIER' if w.at_barrier else 'ACTIVE')
        stall_tag = ' STALLED' if getattr(w, 'scoreboard_stalled', False) else ''
        mask_str = bin(w.active_mask)[2:].zfill(simt.warp_size)
        print(f"  Warp {w.warp_id}: PC={w.pc}, mask=0b{mask_str}, "
              f"{status}{stall_tag}")
        print(f"    Scoreboard: {w.scoreboard}")
        if not w.simt_stack.empty:
            print(f"    SIMT Stack depth: {len(w.simt_stack)}")
    # Cache stats inline
    print(f"  L1 Cache: {simt.l1_cache.stats()}")
    if simt.total_mem_reqs > 0:
        eff = simt.coalesce_count / simt.total_mem_reqs * 100
        print(f"  Coalescing: {simt.coalesce_count}/{simt.total_mem_reqs} "
              f"({eff:.0f}%)")
    print()


def print_memory(simt):
    """Print non-zero global memory values."""
    print(c("=== Global Memory (non-zero) ===", 'bold'))
    non_zero = []
    for i in range(min(256, simt.memory.size_words)):
        v = simt.memory.read_word(i)
        if v != 0:
            non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
    if non_zero:
        for line in non_zero[:20]:
            print(f"  {line}")
        if len(non_zero) > 20:
            print(f"  ... and {len(non_zero) - 20} more entries")
    else:
        print("  (all zero)")
    print()


def print_registers(simt):
    """Print per-thread non-zero register values."""
    print(c("=== Registers ===", 'bold'))
    for w in simt.warps:
        print(f"  Warp {w.warp_id}:")
        for t in w.threads:
            vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
            if vals:
                reg_str = ', '.join(f"r{i}={v}" for i, v in vals)
                print(f"    T{t.thread_id}: {reg_str}")
            else:
                print(f"    T{t.thread_id}: (all zero)")
    print()


def print_warp_regs(simt):
    """Print warp-level uniform registers."""
    from isa import WREG_NAMES
    rev = {v: k for k, v in WREG_NAMES.items()}
    print(c("=== Warp Registers ===", 'bold'))
    for w in simt.warps:
        print(f"  Warp {w.warp_id}:")
        for idx, val in sorted(w.warp_regs.items()):
            name = rev.get(idx, f"wreg[{idx}]")
            print(f"    {name} = {val}")
    print()


def print_simt_stack(simt):
    """Print SIMT stack entries for all warps."""
    print(c("=== SIMT Stack ===", 'bold'))
    for w in simt.warps:
        print(f"  Warp {w.warp_id} (depth={len(w.simt_stack)}):")
        if w.simt_stack.empty:
            print("    (empty)")
        else:
            for i, e in enumerate(w.simt_stack.entries):
                orig = bin(e.orig_mask)[2:].zfill(simt.warp_size)
                taken = bin(e.taken_mask)[2:].zfill(simt.warp_size)
                print(f"    [{i}] reconv=PC{e.reconv_pc} "
                      f"orig=0b{orig} taken=0b{taken} "
                      f"fallthrough=PC{e.fallthrough_pc}")
    print()


def print_scoreboard(simt):
    """Print scoreboard status for all warps."""
    print(c("=== Scoreboard ===", 'bold'))
    for w in simt.warps:
        stalled = getattr(w, 'scoreboard_stalled', False)
        sb_str = str(w.scoreboard)
        print(f"  Warp {w.warp_id}: {sb_str}")
        if stalled:
            print(f"    -> STALLED (scoreboard hazard)")
        if w.scoreboard.reserved:
            for reg_id, rem in w.scoreboard.reserved.items():
                print(f"    r{reg_id}: {rem} cycles remaining")
    print()


def print_cache_stats(simt):
    """Print L1 cache statistics and contents."""
    print(c("=== L1 Cache ===", 'bold'))
    print(f"  {simt.l1_cache.stats()}")

    # Show coalescing info
    if simt.total_mem_reqs > 0:
        eff = simt.coalesce_count / simt.total_mem_reqs * 100
        print(f"  Coalescing: {simt.coalesce_count}/{simt.total_mem_reqs} "
              f"transactions ({eff:.0f}%)")
    else:
        print(f"  Coalescing: no memory transactions yet")

    # Show cache lines that are valid
    print(f"  Cache lines (valid only):")
    found = False
    for i, line in enumerate(simt.l1_cache.lines):
        if line.valid:
            found = True
            data_str = ' '.join(f"0x{v:08X}" for v in line.data)
            print(f"    Line {i:2d}: tag={line.tag} dirty={line.dirty} "
                  f"[{data_str}]")
    if not found:
        print(f"    (all invalid)")
    print()


def print_shared_memory(simt):
    """Print non-zero shared memory values."""
    print(c("=== Shared Memory ===", 'bold'))
    smem = simt.thread_block.shared_memory
    non_zero = []
    for i in range(smem.size_words):
        v = smem.read_word(i)
        if v != 0:
            non_zero.append(f"smem[{i:3d}]=0x{v:08X}({v})")
    if non_zero:
        for line in non_zero[:20]:
            print(f"  {line}")
        if len(non_zero) > 20:
            print(f"  ... and {len(non_zero) - 20} more entries")
    else:
        print("  (all zero)")
    print()


def print_program(simt):
    """Print program disassembly."""
    print(c("=== Program ===", 'bold'))
    for pc, word in enumerate(simt.program):
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
        elif instr.opcode in (OP_SHLD, OP_SHST):
            tag = "SHLD" if instr.opcode == OP_SHLD else "SHST"
            if instr.opcode == OP_SHLD:
                print(f"  PC {pc:2d}: {tag:6s} r{rd}, [{imm & 0xFF}]")
            else:
                print(f"  PC {pc:2d}: {tag:6s} r{rs1}, [{imm & 0xFF}]")
        else:
            print(f"  PC {pc:2d}: {opname:6s} rd=r{rd} rs1=r{rs1} "
                  f"rs2=r{rs2} imm={imm}")
    print()


def predict_next_warp(simt):
    """Predict which warp will execute next, matching scheduler logic.

    Returns (warp_or_None, is_stalled).
    """
    sched = simt.scheduler
    num_warps = len(simt.warps)
    idx = sched.current_idx
    for _ in range(num_warps):
        w = simt.warps[idx]
        idx = (idx + 1) % num_warps
        stalled = getattr(w, 'scoreboard_stalled', False)
        if not w.done and not w.at_barrier and not stalled:
            return w, False
    for _ in range(num_warps):
        w = simt.warps[idx]
        idx = (idx + 1) % num_warps
        stalled = getattr(w, 'scoreboard_stalled', False)
        if not w.done and not w.at_barrier and stalled:
            return w, True
    return None, False


def print_changes(simt, cycle, instr_info, warp_info, stalled,
                  old_regs, new_regs, old_mem, new_mem,
                  old_stack_entries, new_stack_entries,
                  old_cache_hits, old_cache_misses):
    """Print one cycle's state changes in a readable format."""
    if stalled:
        wid = warp_info.warp_id if warp_info else 0
        print(f"[{cycle}] Warp {wid}: SCOREBOARD STALLED  "
              f"| SB={warp_info.scoreboard}")
        return

    if instr_info is None and warp_info is None:
        print(f"[{cycle}] (no warp active)")
        return

    # --- Instruction line ---
    if instr_info:
        wid = instr_info.get('warp_id', 0)
        op = instr_info.get('op', '?')
        rd = instr_info.get('rd', 0)
        rs1 = instr_info.get('rs1', 0)
        rs2 = instr_info.get('rs2', 0)
        imm = instr_info.get('imm', 0)
        pc = instr_info.get('pc', 0)
        mask = instr_info.get('active_mask', 0)
        mask_str = bin(mask)[2:].zfill(simt.warp_size)
        aname = instr_info.get('asm', '')

        if aname:
            print(f"[{cycle}] Warp {wid}: {op} {aname}  "
                  f"| PC={pc} active=0b{mask_str}")
        else:
            print(f"[{cycle}] Warp {wid}: {op}  "
                  f"rd=r{rd} rs1=r{rs1} rs2=r{rs2} imm={imm}  "
                  f"| PC={pc} active=0b{mask_str}")

        # Show divergence info
        if instr_info.get('diverged'):
            taken = instr_info.get('taken_mask', 0)
            nt = instr_info.get('not_taken_mask', 0)
            t_str = bin(taken)[2:].zfill(simt.warp_size)
            nt_str = bin(nt)[2:].zfill(simt.warp_size)
            print(f"         [DIVERGENCE] taken=0b{t_str} not_taken=0b{nt_str}")

        # Show LD/ST with memory address info
        if instr_info.get('is_mem'):
            base = imm & 0x3FF
            tids = [t for t in range(simt.warp_size) if (mask >> t) & 1]
            addrs = [(base + t) & 0x3FF for t in tids]
            contig = all(addrs[i] == addrs[i-1] + 1 for i in range(1, len(addrs))) if addrs else True
            print(f"         [MEM] base={base} threads={len(tids)} "
                  f"contiguous={contig}")

        # Show shared memory access info
        if instr_info.get('is_smem'):
            base = imm & 0xFF
            print(f"         [SMEM] base={base} "
                  f"threads={bin(mask).count('1')}")
    else:
        print(f"[{cycle}] (no instruction)")

    # --- Register changes ---
    reg_diffs = regs_diff(old_regs, new_regs)
    if reg_diffs:
        for wid, tid, rid, ov, nv in reg_diffs:
            print(f"         Reg W{wid} T{tid}: r{rid} {ov} -> {nv}")

    # --- Memory changes ---
    mem_changes = mem_diff(old_mem, new_mem)
    if mem_changes:
        for addr, (ov, nv) in sorted(mem_changes.items()):
            print(f"         Mem mem[{addr}] = {nv}")

    # --- SIMT stack changes ---
    if old_stack_entries != new_stack_entries:
        if len(new_stack_entries) > len(old_stack_entries):
            # pushed = (warp_id, reconv_pc, orig_mask, taken_mask, fallthrough_pc)
            pushed = new_stack_entries[-1]
            orig_s = bin(pushed[2])[2:].zfill(simt.warp_size)
            taken_s = bin(pushed[3])[2:].zfill(simt.warp_size)
            print(f"         SIMT: PUSH reconv=PC{pushed[1]} "
                  f"orig=0b{orig_s} taken=0b{taken_s}")
        elif len(new_stack_entries) < len(old_stack_entries):
            print(f"         SIMT: POP (reconvergence)")
            if new_stack_entries:
                top = new_stack_entries[-1]
                mask_s = bin(top[2])[2:].zfill(simt.warp_size)
                print(f"         SIMT: now top entry mask=0b{mask_s}")
            else:
                print(f"         SIMT: stack now empty (all paths done)")

    # --- Scoreboard status ---
    any_sb = False
    for w in simt.warps:
        if w.scoreboard.reserved:
            any_sb = True
            sb_items = ', '.join(f"r{r}={c}" for r, c in w.scoreboard.reserved.items())
            print(f"         SB W{w.warp_id}: pending {sb_items}")
    if not any_sb:
        print(f"         SB: (clean)")

    # --- Cache stats delta ---
    new_hits = simt.l1_cache.hits
    new_misses = simt.l1_cache.misses
    if new_hits != old_cache_hits or new_misses != old_cache_misses:
        dh = new_hits - old_cache_hits
        dm = new_misses - old_cache_misses
        if dh > 0:
            print(f"         Cache: +{dh} hit(s)")
        if dm > 0:
            print(f"         Cache: +{dm} miss(es)")
        new_total = new_hits + new_misses
        if new_total > 0:
            print(f"         Cache: total {new_hits}h/{new_misses}m "
                  f"({new_hits/new_total*100:.0f}%)")


def run_console(simt, program_text, args):
    """Start interactive console session."""
    warp_size = args.get('warp_size', 4)
    num_warps = args.get('num_warps', 1)
    max_cycles = args.get('max_cycles', 500)

    prog = assemble(program_text)
    simt.load_program(prog)

    # Header
    print("==============================================")
    print("  toygpgpu Learning Console (Phase 5)")
    print("  Memory Hierarchy / Cache / Shared Memory")
    print("==============================================")
    print(f"  Program: {len(prog)} instructions")
    print(f"  Config:  {num_warps} warp(s) x {warp_size} threads/warp")
    print(f"  Commands: Enter=step, r=run, i=info, q=quit")
    print("==============================================")
    print()

    print_program(simt)

    cycle = 0
    old_regs = snapshot_regs(simt)
    old_mem = snapshot_mem(simt)
    old_cache_hits = simt.l1_cache.hits
    old_cache_misses = simt.l1_cache.misses
    auto_run = False

    while cycle < max_cycles:
        # Check if all warps completed
        if not simt.scheduler.has_active_warps():
            print(f"\n[DONE] All warps completed at cycle {cycle}")
            break

        # Handle auto-run
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
            elif cmd.lower() == 'stack':
                print_simt_stack(simt)
                continue
            elif cmd.lower() == 'sb':
                print_scoreboard(simt)
                continue
            elif cmd.lower() == 'cache':
                print_cache_stats(simt)
                continue
            elif cmd.lower() == 'smem':
                print_shared_memory(simt)
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue

        # --- Execute one cycle ---
        old_cache_hits = simt.l1_cache.hits
        old_cache_misses = simt.l1_cache.misses

        # Record pre-step state for detecting which warp executed
        pre_pc = {w.warp_id: w.pc for w in simt.warps}
        pre_mask = {w.warp_id: w.active_mask for w in simt.warps}
        pre_reconv = {}
        for w in simt.warps:
            pre_reconv[w.warp_id] = w.simt_stack.at_reconvergence(w.pc)

        old_stack_entries = []
        for w in simt.warps:
            for e in w.simt_stack.entries:
                old_stack_entries.append(
                    (w.warp_id, e.reconv_pc, e.orig_mask,
                     e.taken_mask, e.fallthrough_pc)
                )

        simt.step()

        # Determine which warp executed and what happened
        exec_warp = None
        exec_instr_info = None
        reconverged = False
        for w in simt.warps:
            wid = w.warp_id
            if w.pc != pre_pc.get(wid, -1) or w.active_mask != pre_mask.get(wid, 0):
                exec_warp = w
                if pre_reconv.get(wid, False):
                    reconverged = True
                else:
                    old_pc = pre_pc[wid]
                    if old_pc < len(simt.program) and old_pc >= 0:
                        raw_word = simt.program[old_pc]
                        instr = decode(raw_word)
                        opname = OPCODE_NAMES.get(instr.opcode, '?')
                        exec_instr_info = {
                            'warp_id': wid, 'op': opname,
                            'rd': instr.rd, 'rs1': instr.rs1, 'rs2': instr.rs2,
                            'imm': instr.imm, 'pc': old_pc,
                            'active_mask': pre_mask.get(wid, 0), 'asm': '',
                            'is_mem': instr.opcode in (0x05, 0x06),
                            'is_smem': instr.opcode in (OP_SHLD, OP_SHST),
                        }
                        if instr.opcode == OP_JMP:
                            target = old_pc + 1 + instr.imm
                            exec_instr_info['asm'] = f"-> PC{target}"
                        elif instr.opcode in (OP_BEQ, OP_BNE):
                            target = old_pc + 1 + instr.imm
                            exec_instr_info['asm'] = f"r{instr.rs1},r{instr.rs2} -> PC{target}"
                        elif instr.opcode == OP_SETP:
                            mode = "EQ" if (instr.imm & 1) == 0 else "NE"
                            exec_instr_info['asm'] = f"r{instr.rs1},r{instr.rs2} ({mode})"
                        elif instr.opcode == OP_SHLD:
                            exec_instr_info['asm'] = f"r{instr.rd}, [{instr.imm & 0xFF}]"
                        elif instr.opcode == OP_SHST:
                            exec_instr_info['asm'] = f"r{instr.rs1}, [{instr.imm & 0xFF}]"
                break

        new_stack_entries = []
        for w in simt.warps:
            for e in w.simt_stack.entries:
                new_stack_entries.append(
                    (w.warp_id, e.reconv_pc, e.orig_mask,
                     e.taken_mask, e.fallthrough_pc)
                )

        new_regs = snapshot_regs(simt)
        new_mem = snapshot_mem(simt)

        if reconverged:
            stalled_flag = exec_warp is not None and getattr(exec_warp, 'scoreboard_stalled', False)
            print_changes(simt, cycle, None, exec_warp, stalled_flag,
                          old_regs, new_regs, old_mem, new_mem,
                          old_stack_entries, new_stack_entries,
                          old_cache_hits, old_cache_misses)
            if exec_warp is not None and exec_warp.pc > 0 and exec_warp.pc - 1 < len(simt.program):
                actual_instr = decode(simt.program[exec_warp.pc - 1])
                opname = OPCODE_NAMES.get(actual_instr.opcode, '?')
                mask_str = bin(exec_warp.active_mask)[2:].zfill(simt.warp_size)
                print(f"         [RECONV: {opname} at PC={exec_warp.pc - 1} "
                      f"active=0b{mask_str}]")
        else:
            stalled_flag = exec_warp is not None and getattr(exec_warp, 'scoreboard_stalled', False)
            print_changes(simt, cycle, exec_instr_info, exec_warp, stalled_flag,
                          old_regs, new_regs, old_mem, new_mem,
                          old_stack_entries, new_stack_entries,
                          old_cache_hits, old_cache_misses)

        old_regs = new_regs
        old_mem = new_mem
        cycle += 1

        if auto_run and not simt.scheduler.has_active_warps():
            print(f"\n[DONE] All warps completed at cycle {cycle}")
            break

    # --- Final summary ---
    print(f"\n{'='*46}")
    print(f"  Final State (after {cycle} cycles)")
    print(f"{'='*46}")
    print(f"Instructions executed: {simt.instr_count}")

    print_state(simt)
    print_memory(simt)
    print_registers(simt)
    print_cache_stats(simt)
    print_shared_memory(simt)

    print(f"{'='*46}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options: --warp-size N --num-warps N --max-cycles N")
        sys.exit(1)

    asm_file = sys.argv[1]
    args = {
        'warp_size': 4,
        'num_warps': 1,
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
        memory_size=1024,
    )

    # Auto mode (--auto or --trace)
    if args.get('auto'):
        prog = assemble(program_text)
        simt.load_program(prog)
        simt.run(trace=True)
        return

    # Batch mode via env var
    if os.environ.get('LEARN_CONSOLE_BATCH'):
        prog = assemble(program_text)
        simt.load_program(prog)
        simt.run()
        print(f"Cycles: {simt.instr_count}")
        print(f"Memory (non-zero):")
        for i in range(simt.memory.size_words):
            v = simt.memory.read_word(i)
            if v != 0:
                print(f"  mem[{i}]: {v}")
        print(f"Cache: {simt.l1_cache.stats()}")
        if simt.total_mem_reqs > 0:
            eff = simt.coalesce_count / simt.total_mem_reqs * 100
            print(f"Coalescing: {simt.coalesce_count}/{simt.total_mem_reqs} ({eff:.0f}%)")
        return

    run_console(simt, program_text, args)


if __name__ == '__main__':
    main()
