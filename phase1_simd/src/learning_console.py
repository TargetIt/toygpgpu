#!/usr/bin/env python3
"""
Learning Console -- Interactive SIMD vector learning console for Phase 1.
=========================================================================
Step through SIMD CPU execution cycle by cycle, observing scalar and vector state.

Usage:
  python3 learning_console.py <program.asm> [options]

Options:
  --max-cycles N    Maximum cycles to execute (default 500)

Interactive commands:
  Enter / s   Single-step one cycle
  r           Run to HALT
  r N         Run N cycles
  i           Print current state (PC + scalar regs + vector regs + mem)
  m           Print non-zero memory
  reg         Print scalar registers (non-zero, r0-r15)
  vreg        Print vector registers (non-zero lanes)
  q           Quit
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpu import CPU
from assembler import assemble
from isa import OPCODE_NAMES, decode, is_vector


def print_banner(cpu, program):
    """Print the startup banner with program listing."""
    sep = "+" + "-" * 58 + "+"
    print(sep)
    print("|     toygpgpu Learning Console -- Phase 1 (SIMD Vector)       |")
    print(sep)
    print(f"|  Program: {len(program)} instructions                           |")
    print(f"|  VLEN:    {cpu.vlen} lanes                                  |")
    print(f"|  Commands: Enter=step, r=run, i=info, q=quit              |")
    print(sep)
    print()
    print("--- Program ---")
    for pc, word in enumerate(program):
        instr = decode(word)
        opname = OPCODE_NAMES.get(instr.opcode, "?")
        vec = "V" if is_vector(instr.opcode) else " "
        parts = []
        parts.append(f"{opname:6s}")
        if is_vector(instr.opcode):
            parts.append(f"vd=v{instr.rd}")
            parts.append(f"vs1=v{instr.rs1}")
            parts.append(f"vs2=v{instr.rs2}")
            parts.append(f"imm={instr.imm}")
        else:
            parts.append(f"rd=r{instr.rd}")
            parts.append(f"rs1=r{instr.rs1}")
            parts.append(f"rs2=r{instr.rs2}")
            parts.append(f"imm={instr.imm}")
        print(f"  PC {pc:2d}: {vec} {' '.join(parts)}")
    print()


def snapshot_scalar_regs(cpu):
    """Return a snapshot dict of all scalar registers."""
    regs = {}
    for i in range(16):
        regs[i] = cpu.reg_file.read(i)
    return regs


def snapshot_vector_regs(cpu):
    """Return a snapshot dict of all vector register lanes."""
    vregs = {}
    for ri in range(8):
        for lane in range(cpu.vlen):
            v = cpu.vec_reg_file.read_lane(ri, lane)
            if v != 0:
                vregs[(ri, lane)] = v
    return vregs


def snapshot_mem(cpu):
    """Return a snapshot dict of non-zero memory words."""
    mem = {}
    for i in range(min(256, cpu.memory.size_words)):
        v = cpu.memory.read_word(i)
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


def print_state(cpu):
    """Print current CPU state summary."""
    print("--- Current State ---")
    status = "HALTED" if cpu.halted else "RUNNING"
    print(f"PC={cpu.pc}, {status}, Instructions={cpu.instr_count}, VLEN={cpu.vlen}")


def print_memory(cpu):
    """Print non-zero memory words."""
    non_zero = []
    for i in range(min(256, cpu.memory.size_words)):
        v = cpu.memory.read_word(i)
        if v != 0:
            non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
    if non_zero:
        print("  " + "\n  ".join(non_zero[:30]))
    else:
        print("  (all zero)")


def print_scalar_registers(cpu):
    """Print all non-zero scalar registers (r0-r15)."""
    vals = []
    for i in range(16):
        v = cpu.reg_file.read(i)
        if v != 0 or i == 0:
            vals.append(f"r{i}={v}")
    print("  " + ", ".join(vals))


def print_vector_registers(cpu):
    """Print all vector registers showing non-zero lanes."""
    for ri in range(8):
        lanes = []
        for lane in range(cpu.vlen):
            v = cpu.vec_reg_file.read_lane(ri, lane)
            if v != 0:
                lanes.append(f"[{lane}]={v}")
        if lanes:
            print(f"  v{ri}: " + ", ".join(lanes))
        else:
            # Check if entire register is zero by looking at lane 0
            v0 = cpu.vec_reg_file.read_lane(ri, 0)
            if ri == 0 and v0 == 0:
                # Only show v0 if it's the only one all-zero, otherwise skip
                pass


def print_all_vector_registers(cpu):
    """Print all vector registers (including zero lanes)."""
    for ri in range(8):
        lane_vals = []
        for lane in range(cpu.vlen):
            v = cpu.vec_reg_file.read_lane(ri, lane)
            lane_vals.append(str(v))
        print(f"  v{ri}: [" + ", ".join(lane_vals) + "]")


def do_step(cpu, cycle):
    """Execute one cycle and print the step info.

    Returns True if the CPU is still running, False if halted.
    """
    old_sregs = snapshot_scalar_regs(cpu)
    old_vregs = snapshot_vector_regs(cpu)
    old_mem = snapshot_mem(cpu)

    # Decode the current instruction before stepping
    instr = None
    opname = "???"
    vec = False
    if not cpu.halted and 0 <= cpu.pc < len(cpu.program):
        raw_word = cpu.program[cpu.pc]
        instr = decode(raw_word)
        opname = OPCODE_NAMES.get(instr.opcode, "?")
        vec = is_vector(instr.opcode)

    running = cpu.step()

    new_sregs = snapshot_scalar_regs(cpu)
    new_vregs = snapshot_vector_regs(cpu)
    new_mem = snapshot_mem(cpu)
    mem_changes = mem_diff(old_mem, new_mem)

    last_pc = cpu.pc - 1 if not cpu.halted and cpu.pc > 0 else cpu.pc

    if instr is not None:
        if vec:
            # Vector instruction: show vector register changes
            vreg_changes = []
            for ri in range(8):
                for lane in range(cpu.vlen):
                    old_v = old_vregs.get((ri, lane), 0)
                    new_v = new_vregs.get((ri, lane), 0)
                    if old_v != new_v:
                        vreg_changes.append(f"v{ri}[{lane}]:{old_v}->{new_v}")
            vreg_str = " ".join(vreg_changes) if vreg_changes else "no change"

            mem_str = ""
            if mem_changes:
                parts = []
                for addr, (old_v, new_v) in sorted(mem_changes.items()):
                    parts.append(f"mem[{addr}]:{old_v}->{new_v}")
                mem_str = " | " + ", ".join(parts)

            print(f"Cycle {cycle}: PC={last_pc} {opname} vd=v{instr.rd} "
                  f"vs1=v{instr.rs1} vs2=v{instr.rs2} imm={instr.imm} "
                  f"| {vreg_str}{mem_str}")
        else:
            # Scalar instruction
            reg_changes = []
            for i in range(16):
                if old_sregs[i] != new_sregs[i]:
                    reg_changes.append(f"r{i}:{old_sregs[i]}->{new_sregs[i]}")
            reg_str = " ".join(reg_changes) if reg_changes else "no reg change"

            mem_str = ""
            if mem_changes:
                parts = []
                for addr, (old_v, new_v) in sorted(mem_changes.items()):
                    parts.append(f"mem[{addr}]:{old_v}->{new_v}")
                mem_str = " | " + ", ".join(parts)

            print(f"Cycle {cycle}: PC={last_pc} {opname} rd=r{instr.rd} "
                  f"rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm} "
                  f"| {reg_str}{mem_str}")
    else:
        print(f"Cycle {cycle}: PC={last_pc} (no instruction / halted)")

    return running


def run_console(cpu, program_text, args):
    """Launch the interactive console."""
    max_cycles = args.get("max_cycles", 500)

    prog = assemble(program_text)
    cpu.load_program(prog)

    print_banner(cpu, prog)

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
                    if not do_step(cpu, cycle):
                        break
                    cycle += 1
                continue
            elif cmd.lower() == "i":
                print_state(cpu)
                print("  Scalar registers:")
                print_scalar_registers(cpu)
                print("  Vector registers:")
                print_all_vector_registers(cpu)
                print("  Memory:")
                print_memory(cpu)
                continue
            elif cmd.lower() == "m":
                print_memory(cpu)
                continue
            elif cmd.lower() == "reg":
                print_scalar_registers(cpu)
                continue
            elif cmd.lower() == "vreg":
                print_all_vector_registers(cpu)
                continue
            else:
                print(f"Unknown command: {cmd}")
                print("Commands: Enter/s=step, r=run, r N=run N, i=info, "
                      "m=memory, reg=registers, vreg=vector regs, q=quit")
                continue

        # Execute one cycle
        running = do_step(cpu, cycle)
        cycle += 1

        if not running:
            print(f"\n  [HALT] CPU halted at cycle {cycle}")
            break

        if auto_step:
            if cycle % 10 == 0:
                time.sleep(0.05)

    # Final state
    print(f"\n--- Final State ---")
    print(f"Cycles executed: {cycle}")
    print(f"Instructions executed: {cpu.instr_count}")
    print(f"Scalar registers:")
    print_scalar_registers(cpu)
    print(f"Vector registers:")
    print_all_vector_registers(cpu)
    print(f"Memory (non-zero):")
    print_memory(cpu)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options: --max-cycles N")
        sys.exit(1)

    asm_file = sys.argv[1]
    args = {
        "max_cycles": 500,
        'auto': False,
    }

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--max-cycles":
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

    cpu = CPU(vlen=8, memory_size=1024)

    if args.get('auto'):
        prog = assemble(program_text)
        cpu.load_program(prog)
        cpu.run(trace=True)
        return

    run_console(cpu, program_text, args)


if __name__ == "__main__":
    main()
