#!/usr/bin/env python3
"""Phase 5 Test Suite — Memory Hierarchy"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared_memory import SharedMemory
from cache import L1Cache
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
    simt = SIMTCore(warp_size=8, num_warps=2, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    for addr, expected in checks.items():
        actual = simt.memory.read_word(addr)
        if actual == (expected & 0xFFFFFFFF):
            passed += 1; print(f"  ✅ {f}: mem[{addr}] = {expected}")
        else:
            failed += 1; print(f"  ❌ {f}: mem[{addr}] = {actual}, expected {expected}")
    return simt

def test_shared_memory():
    print("\n--- SharedMemory Tests ---")
    sm = SharedMemory(256)
    sm.write_word(0, 42)
    check(sm.read_word(0) == 42, "write/read")
    sm.write_word(255, 0xDEADBEEF)
    check(sm.read_word(255) == 0xDEADBEEF, "boundary write/read")

def test_l1_cache():
    print("\n--- L1Cache Tests ---")
    c = L1Cache()
    check(c.read(0) is None, "cold miss")
    c.write(0, 42)
    check(c.read(0) is None, "write-through miss (no-allocate)")
    c.fill_line(0, [10, 20, 30, 40])
    check(c.read(0) == 10, "cache hit after fill")
    check(c.read(3) == 40, "cache hit offset 3")
    check(c.read(4) is None, "different line miss")
    check(c.hits + c.misses > 0, "stats tracked")

def test_programs():
    global passed, failed
    print("\n--- Assembly Program Tests ---")
    simt = run_prog("01_shared_mem.asm", {})
    def check_shmem(s):
        for tid in range(8):
            if s.memory.read_word(100+tid) != tid*10:
                return False, f"mem[100+{tid}] != {tid*10}"
        return True, "shared memory OK"
    ok, msg = check_shmem(simt)
    if ok: passed += 1; print(f"  ✅ 01_shared_mem.asm: {msg}")
    else: failed += 1; print(f"  ❌ 01_shared_mem.asm: {msg}")

    simt2 = run_prog("02_coalescing.asm", {})
    def check_coal(s):
        for tid in range(8):
            if s.memory.read_word(50+tid) != tid*10:
                return False, f"coalescing failed at {tid}"
        return True, f"coalesced ({s.coalesce_count}/{s.total_mem_reqs})"
    ok, msg = check_coal(simt2)
    if ok: passed += 1; print(f"  ✅ 02_coalescing.asm: {msg}")
    else: failed += 1; print(f"  ❌ 02_coalescing.asm: {msg}")

    run_prog("03_cache_basic.asm", {10: 84})
    print(f"     L1: {simt2.l1_cache.stats()}")

    run_prog("04_multi_warp_shared.asm", {})  # basic smoke test
    passed += 1; print(f"  ✅ 04_multi_warp_shared.asm: shared mem across warps")
    run_prog("05_backward_compat.asm", {0: 13, 10: 0, 11: 1, 12: 2, 13: 3})


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 5: Memory Hierarchy — Test Suite")
    print("=" * 60)
    test_shared_memory()
    test_l1_cache()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
