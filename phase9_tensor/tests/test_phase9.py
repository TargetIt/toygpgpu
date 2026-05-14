#!/usr/bin/env python3
"""Phase 9 Test Suite — Tensor Core MMA"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from simt_core import SIMTCore
from assembler import assemble

passed = 0; failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")

def pack(lo, hi):
    """Pack two 16-bit signed values into 32-bit."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)

def run_mma_test(f, preload, checks):
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path, encoding='utf-8') as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=1024)
    for addr, val in preload.items():
        simt.memory.write_word(addr, val)
    simt.load_program(prog)
    simt.run()
    for addr, expected in checks.items():
        actual = simt.memory.read_word(addr)
        if actual == (expected & 0xFFFFFFFF):
            passed += 1; print(f"  ✅ {f}: mem[{addr}] = {expected}")
        else:
            failed += 1; print(f"  ❌ {f}: mem[{addr}] = {actual}, expected {expected}")


def test_programs():
    print("\n--- MMA Tests ---")

    # Test 01: Dot product
    run_mma_test("01_mma_dot.asm",
                 {0: pack(2, 3), 1: pack(4, 5)},
                 {10: 33})  # 2*4+3*5+10

    # Test 02: 2x2 Matrix Multiply
    # A = [[2,3],[1,4]], B = [[4,5],[6,7]], C = 1
    run_mma_test("02_matmul_2x2.asm",
                 {0: pack(2, 3), 1: pack(1, 4),
                  2: pack(4, 6), 3: pack(5, 7)},
                 {10: 27, 11: 32, 12: 29, 13: 34})

    # Test 03: Different values
    run_mma_test("03_negative_mma.asm",
                 {0: pack(1, 7), 1: pack(3, 2)},
                 {10: 27})  # 1*3+7*2+10


def test_backward_compat():
    global passed, failed
    print("\n--- Backward Compat ---")
    from assembler import assemble
    src = "MOV r1, 10\nADD r2, r1, r1\nST r2, [0]\nHALT\n"
    prog = assemble(src)
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.load_program(prog)
    simt.run()
    check(simt.memory.read_word(0) == 20, "scalar ADD still works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 9: Tensor Core MMA — Test Suite")
    print("=" * 60)
    test_programs()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
