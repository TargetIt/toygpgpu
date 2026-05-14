#!/usr/bin/env python3
"""Phase 7 Test Suite — I-Buffer + Operand Collector"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ibuffer import IBuffer, IBufferEntry
from operand_collector import OperandCollector
from simt_core import SIMTCore
from assembler import assemble

passed = 0; failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")

def run_prog(f, checks):
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path, encoding='utf-8') as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=8, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    for addr, expected in checks.items():
        actual = simt.memory.read_word(addr)
        if actual == (expected & 0xFFFFFFFF):
            passed += 1; print(f"  ✅ {f}: mem[{addr}] = {expected}")
        else:
            failed += 1; print(f"  ❌ {f}: mem[{addr}] = {actual}, expected {expected}")
    return simt


def test_ibuffer():
    print("\n--- I-Buffer Tests ---")
    ib = IBuffer(2)
    check(ib.has_free(), "initially free")
    check(not ib.has_ready(), "no ready entries initially")
    ib.write(0x01234000, 0)
    check(ib.has_free(), "1/2 free")
    ib.set_ready(0)
    check(ib.has_ready(), "ready after set_ready")
    e = ib.consume()
    check(e is not None and e.pc == 0, "consume returns correct PC")
    check(not ib.has_ready(), "no ready after consume")
    ib.flush()
    check(ib.has_free() and not ib.has_ready(), "flush clears all")


def test_op_collector():
    print("\n--- Operand Collector Tests ---")
    oc = OperandCollector(4)
    check(oc.bank_of(1) == 1, "r1 → bank 1")
    check(oc.bank_of(5) == 1, "r5 → bank 1 (same as r1)")
    check(oc.bank_of(2) == 2, "r2 → bank 2")
    check(oc.bank_of(0) == -1, "r0 → no bank")

    # Same bank → should NOT be a conflict at first (banks free)
    ok, _ = oc.can_read_operands(1, 5)
    check(ok, "r1+r5 no initial conflict (banks free)")

    oc.reserve_banks(1, 5)
    check(oc.bank_busy[1], "bank 1 busy after reserve")
    oc.release_banks()
    check(not oc.bank_busy[1], "bank 1 free after release")

    # Different banks
    ok, _ = oc.can_read_operands(2, 3)
    check(ok, "r2+r3 different banks → no conflict")


def test_programs():
    global passed, failed
    print("\n--- Assembly Program Tests ---")
    simt = run_prog("01_ibuffer_basic.asm", {0: 13})
    print(f"     {simt.op_collector.stats()}")

    simt2 = run_prog("02_bank_conflict.asm", {0: 30, 1: 8})
    print(f"     {simt2.op_collector.stats()}")
    run_prog("03_branch_ibuffer.asm", {0: 1, 10: 42})

    run_prog("04_backward_compat.asm", {0: 20, 10: 0, 11: 1, 12: 2, 13: 3})


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 7: Pipeline Decouple — Test Suite")
    print("=" * 60)
    test_ibuffer()
    test_op_collector()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
