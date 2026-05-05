#!/usr/bin/env python3
"""Phase 4 测试套件 — Scoreboard + Pipeline"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scoreboard import Scoreboard
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
    with open(path) as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=8, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    for addr, expected in checks.items():
        actual = simt.memory.read_word(addr)
        if actual == (expected & 0xFFFFFFFF):
            passed += 1; print(f"  ✅ {f}: mem[{addr}] = {expected}")
        else:
            failed += 1; print(f"  ❌ {f}: mem[{addr}] = {actual}, expected {expected}")


def test_scoreboard():
    print("\n--- Scoreboard Tests ---")
    sb = Scoreboard()
    check(not sb.stalled, "initially clean")
    sb.reserve(1, 3)
    check(sb.check_raw(1), "RAW: r1 has pending write")
    check(not sb.check_raw(2), "RAW: r2 clean")
    check(sb.check_waw(1), "WAW: r1 has pending write")
    check(not sb.check_waw(2), "WAW: r2 clean")
    check(sb.stalled, "stalled after reserve")
    sb.advance(); sb.advance(); sb.advance()
    check(not sb.stalled, "clean after 3 advances (latency=3)")
    check(not sb.check_raw(1), "RAW: r1 clean after release")

    # r0 always clean
    sb.reserve(0, 10)
    check(not sb.check_waw(0), "WAW: r0 never reserved")
    check(not sb.check_raw(0), "RAW: r0 always clean")


def test_programs():
    global passed, failed
    print("\n--- Assembly Program Tests ---")
    # Test 01: basic ALU (no real hazard at 1 cycle latency since advance happens before check)
    run_prog("01_raw_hazard.asm", {0: 15})
    run_prog("02_waw_hazard.asm", {0: 50})
    run_prog("03_ld_latency.asm", {0: 84})
    run_prog("04_no_hazard.asm", {0: 60})

    # Test 05: backward compat
    def check_compat(simt):
        for tid in range(8):
            if simt.memory.read_word(100+tid) != tid*2:
                return False, f"mem[100+{tid}] != {tid*2}"
            if simt.memory.read_word(200+tid) != tid*2+2:
                return False, f"mem[200+{tid}] != {tid*2+2}"
        return True, "all correct"
    path = os.path.join(os.path.dirname(__file__), 'programs', '05_backward_compat.asm')
    with open(path) as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=8, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    ok, msg = check_compat(simt)
    if ok:
        passed += 1
        print(f"  ✅ 05_backward_compat.asm: {msg}")
    else:
        failed += 1
        print(f"  ❌ 05_backward_compat.asm: {msg}")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 4: Scoreboard + Pipeline — Test Suite")
    print("=" * 60)
    test_scoreboard()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
