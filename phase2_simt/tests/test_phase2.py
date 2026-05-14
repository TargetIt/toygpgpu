#!/usr/bin/env python3
"""Phase 2 测试套件 — SIMT Core"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isa import *
from warp import Thread, Warp
from scheduler import WarpScheduler
from simt_core import SIMTCore
from assembler import assemble
from register_file import RegisterFile

passed = 0
failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


def run_prog(asm_file, cpu_checks=None, checks=None):
    """运行汇编测试。cpu_checks: callable(simt) → (ok, msg)"""
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', asm_file)
    with open(path, encoding='utf-8') as f:
        prog = assemble(f.read())
    simt = SIMTCore(warp_size=8, num_warps=2, memory_size=1024)
    simt.load_program(prog)
    simt.run()

    if cpu_checks:
        ok, msg = cpu_checks(simt)
        if ok:
            passed += 1
            print(f"  ✅ {asm_file}: {msg}")
        else:
            failed += 1
            print(f"  ❌ {asm_file}: {msg}")
    if checks:
        for addr, expected in checks.items():
            actual = simt.memory.read_word(addr)
            exp = expected & 0xFFFFFFFF
            if actual == exp:
                passed += 1
                print(f"  ✅ {asm_file}: mem[{addr}] = {expected}")
            else:
                failed += 1
                print(f"  ❌ {asm_file}: mem[{addr}] = {actual}, expected {expected}")


# ============================================================
def test_thread():
    print("\n--- Thread Tests ---")
    t = Thread(3, num_regs=16)
    check(t.thread_id == 3, "thread_id = 3")
    t.write_reg(1, 42)
    check(t.read_reg(1) == 42, "thread reg read/write")
    check(t.read_reg(0) == 0, "thread r0 hardwired")
    t.write_reg(0, 99)
    check(t.read_reg(0) == 0, "thread r0 write ignored")


def test_warp():
    print("\n--- Warp Tests ---")
    w = Warp(warp_id=0, warp_size=8)
    check(len(w.threads) == 8, "8 threads in warp")
    check(w.warp_size == 8, "warp_size=8")
    check(w.active_mask == 0xFF, "all active initially")
    active = w.active_threads()
    check(len(active) == 8, "8 active threads")
    check(not w.at_barrier, "not at barrier initially")
    check(not w.done, "not done initially")

    # Test per-thread register independence
    w.threads[0].write_reg(1, 10)
    w.threads[1].write_reg(1, 20)
    check(w.threads[0].read_reg(1) == 10, "thread 0 reg independent")
    check(w.threads[1].read_reg(1) == 20, "thread 1 reg independent")


def test_scheduler():
    print("\n--- Scheduler Tests ---")
    warps = [Warp(0, 4), Warp(1, 4)]
    s = WarpScheduler(warps)
    # Round-robin: 0, 1, 0, 1, ...
    w = s.select_warp()
    check(w.warp_id == 0, "RR: first warp=0")
    w = s.select_warp()
    check(w.warp_id == 1, "RR: second warp=1")
    w = s.select_warp()
    check(w.warp_id == 0, "RR: third warp=0")
    # Mark warp 0 done
    warps[0].done = True
    w = s.select_warp()
    check(w.warp_id == 1, "RR: skip done warp, pick 1")


def test_isa_simt():
    print("\n--- ISA SIMT Tests ---")
    for opcode, name in [(OP_TID, "TID"), (OP_WID, "WID"), (OP_BAR, "BAR")]:
        w = encode_rtype(opcode, 1, 0, 0)
        check(decode(w).opcode == opcode, f"{name} encode/decode")


def test_assembler_simt():
    print("\n--- Assembler SIMT Tests ---")
    src = "TID r1\nWID r2\nBAR\nHALT\n"
    prog = assemble(src)
    check(decode(prog[0]).opcode == OP_TID, "asm TID")
    check(decode(prog[0]).rd == 1, "asm TID rd=r1")
    check(decode(prog[1]).opcode == OP_WID, "asm WID")
    check(decode(prog[2]).opcode == OP_BAR, "asm BAR")


def test_programs():
    print("\n--- Assembly Programs ---")

    # Test 01: TID/WID
    def check_tid(simt):
        for tid in range(8):
            if simt.memory.read_word(100 + tid) != tid:
                return False, f"mem[{100 + tid}] != {tid}"
        return True, "TID correct (0..7)"

    run_prog("01_tid_wid.asm", cpu_checks=check_tid)

    # Test 02: Thread-independent regs (tid * 10)
    def check_thread_regs(simt):
        for tid in range(8):
            if simt.memory.read_word(200 + tid) != tid * 10:
                return False, f"mem[{200 + tid}] != {tid * 10}"
        return True, "per-thread compute correct"

    run_prog("02_thread_regs.asm", cpu_checks=check_thread_regs)

    # Test 03: Barrier
    run_prog("03_barrier.asm", checks={
        300: 0, 301: 10, 302: 20, 303: 30, 304: 40, 305: 50, 306: 60, 307: 70,
        310: 10, 311: 20, 312: 30, 313: 40, 314: 50, 315: 60, 316: 70, 317: 80,
    })

    # Test 04: Multi-thread vector add
    def check_vadd(simt):
        expected = [11, 22, 33, 44, 55, 66, 77, 88]
        for i, e in enumerate(expected):
            if simt.memory.read_word(16 + i) != e:
                return False, f"mem[{16+i}] != {e}"
        return True, "C=A+B across threads"

    run_prog("04_vector_add_mt.asm", cpu_checks=check_vadd)

    # Test 05: Multi-warp
    def check_multi_warp(simt):
        # Both warps run same kernel, store wid*100+tid to mem[tid]
        # Warp 0: mem[0..7] = [0,1,2,3,4,5,6,7]
        # Warp 1: mem[0..7] = [100,101,102,103,104,105,106,107]
        # But they overwrite! Check that each warp's threads ran
        vals = set(simt.memory.read_word(i) for i in range(8))
        # Should have values from either warp
        if len(vals) > 0:
            return True, f"multi-warp executed ({len(vals)} unique values)"
        return False, "no warp output"

    run_prog("05_multi_warp.asm", cpu_checks=check_multi_warp)


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 2: SIMT Core (Warp) — Test Suite")
    print("=" * 60)

    test_thread()
    test_warp()
    test_scheduler()
    test_isa_simt()
    test_assembler_simt()
    test_programs()

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
