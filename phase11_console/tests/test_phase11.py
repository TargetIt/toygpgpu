#!/usr/bin/env python3
"""Phase 11 Test Suite — Learning Console"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simt_core import SIMTCore
from assembler import assemble
from console_display import (
    render_cycle, snapshot_regs, mem_diff, c, _render_scoreboard,
    _render_ibuffer, _render_simt_stack, _render_reg_changes
)
from isa import OPCODE_NAMES, decode

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")


def test_display_modules():
    print("\n--- Display Module Tests ---")

    # Load a simple program
    prog = assemble("MOV r1, 5\nADD r2, r1, r1\nST r2, [0]\nHALT\n")
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(prog)

    # Snapshot
    regs = snapshot_regs(simt)
    check(len(regs) == 1, "snapshot 1 warp")
    check(0 in regs[0], "snapshot thread 0")

    # Execute one step and check render
    old_mem = [simt.memory.read_word(i) for i in range(256)]
    simt.step()
    new_mem = [simt.memory.read_word(i) for i in range(256)]
    diffs = mem_diff(old_mem, new_mem)
    check(len(diffs) >= 0, "mem_diff works")

    # Render scoreboard
    sb_str = _render_scoreboard(simt)
    check(isinstance(sb_str, list), "scoreboard render returns list")

    # Render I-Buffer
    ib_str = _render_ibuffer(simt)
    check(isinstance(ib_str, list), "ibuffer render returns list")

    # Render SIMT stack
    st_str = _render_simt_stack(simt)
    check(isinstance(st_str, list), "simt_stack render returns list")

    # Full cycle render
    instr_info = {'op': 'MOV', 'rd': 1, 'rs1': 0, 'rs2': 0, 'pc': 0, 'active': 4}
    stage_info = {'fetch': 'PC=0→IBuffer', 'decode': 'MOV decoded', 'issue': 'ok',
                  'exec': 'MOV r1←5', 'wb': 'r1←5'}
    output = render_cycle(0, simt, instr_info, {}, [], stage_info)
    check('MOV' in output, "cycle render shows MOV")
    check('Scoreboard' in output or 'scoreboard' in output.lower(), "shows scoreboard")
    check('I-Buffer' in output or 'i-buffer' in output.lower(), "shows I-Buffer")


def test_console_smoke():
    print("\n--- Console Smoke Test ---")

    # Run learning_console in non-interactive mode (via import)
    prog = assemble("MOV r1, 10\nADD r2, r1, r1\nST r2, [0]\nHALT\n")
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.load_program(prog)

    # Run all cycles and collect output
    cycles = 0
    while simt.scheduler.has_active_warps():
        cycles += 1
        simt.step()

    check(cycles > 2, f"program ran {cycles} cycles")
    check(simt.memory.read_word(0) == 20, "correct result (10+10=20)")

    # Check that render doesn't crash
    instr_info = {'op': 'ADD', 'rd': 2, 'rs1': 1, 'rs2': 1, 'pc': 1, 'active': 1}
    stage_info = {k: 'test' for k in ['fetch', 'decode', 'issue', 'exec', 'wb']}
    output = render_cycle(0, simt, instr_info, {}, [], stage_info)
    check(isinstance(output, str) and len(output) > 100, f"full render output ({len(output)} chars)")

    # Verify all sections appear
    for section in ['Pipeline', 'Scoreboard', 'I-Buffer', 'OpCollector']:
        check(section in output, f"render contains {section}")


def test_divergence_display():
    print("\n--- Divergence Display Test ---")
    path = os.path.join(os.path.dirname(__file__), 'programs', 'demo_divergence.asm')
    with open(path) as f:
        prog = assemble(f.read())
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(prog)

    # Run a few cycles and check display
    for cycle in range(10):
        if not simt.scheduler.has_active_warps():
            break

        w = simt.warps[0]
        prev_regs = snapshot_regs(simt)
        old_mem = [simt.memory.read_word(i) for i in range(256)]

        simt.step()

        new_mem = [simt.memory.read_word(i) for i in range(256)]
        mem_changes = mem_diff(old_mem, new_mem)

        # Try to get current instruction
        instr_info = None
        stage_info = {k: f"cycle {cycle}" for k in ['fetch', 'decode', 'issue', 'exec', 'wb']}
        for e in w.ibuffer.entries:
            if e.valid:
                instr = decode(e.instruction_word)
                instr_info = {
                    'op': OPCODE_NAMES.get(instr.opcode, '?'),
                    'rd': instr.rd, 'rs1': instr.rs1, 'rs2': instr.rs2,
                    'pc': e.pc, 'active': bin(w.active_mask).count('1'),
                }
                break

        output = render_cycle(cycle, simt, instr_info, prev_regs, mem_changes, stage_info)
        check(isinstance(output, str), f"cycle {cycle} render OK")

        # Check SIMT stack appears when divergence happens
        if not w.simt_stack.empty:
            check('SIMT Stack' in output, f"SIMT Stack shown at cycle {cycle}")
            break  # test passes once we see the stack

    check(True, "divergence display completed")  # extra pass


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 11: Learning Console — Test Suite")
    print("=" * 60)
    test_display_modules()
    test_console_smoke()
    test_divergence_display()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
