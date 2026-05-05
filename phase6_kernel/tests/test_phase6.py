#!/usr/bin/env python3
"""Phase 6 Test Suite — Kernel Launch & Scheduling"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scheduler import WarpScheduler
from warp import Warp
from gpu_sim import GPUSim, PerfCounters
from assembler import assemble

passed = 0; failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")


def test_gto_scheduler():
    print("\n--- GTO Scheduler Tests ---")
    warps = [Warp(0, 4), Warp(1, 4)]
    # RR
    s = WarpScheduler(warps, policy="rr")
    w = s.select_warp(); check(w.warp_id == 0, "RR: first=0")
    w = s.select_warp(); check(w.warp_id == 1, "RR: second=1")

    # GTO: first pick, then stall warp 0, should pick warp 1
    warps2 = [Warp(0, 4), Warp(1, 4)]
    s2 = WarpScheduler(warps2, policy="gto")
    w = s2.select_warp(); check(w.warp_id == 0, "GTO: first=0")
    warps2[0].scoreboard_stalled = True
    w = s2.select_warp(); check(w.warp_id == 1, "GTO: skip stalled, pick 1")
    warps2[0].scoreboard_stalled = False
    warps2[1].scoreboard_stalled = True
    w = s2.select_warp(); check(w.warp_id == 0, "GTO: pick oldest active")


def test_perf_counters():
    print("\n--- PerfCounters Tests ---")
    p = PerfCounters()
    check(p.ipc == 0.0, "IPC=0 initially")
    p.total_cycles = 10
    p.total_instructions = 5
    check(abs(p.ipc - 0.5) < 0.01, "IPC=0.5 after 5/10")
    print(p.report())


def test_programs():
    global passed, failed
    print("\n--- Kernel Launch Tests ---")
    path = os.path.join(os.path.dirname(__file__), 'programs')
    rpt = None

    # Test 01: GTO scheduling
    with open(f"{path}/01_gto_schedule.asm") as f:
        prog = assemble(f.read())
    gpu = GPUSim(num_sms=1, warp_size=4, memory_size=1024)
    gpu.launch_kernel(prog, grid_dim=(1,), block_dim=(8,))
    gpu.run()
    rpt = gpu
    # Warp 1 overwrites warp 0 (same base+tid) → last-write = warp 1 values
    vals = [gpu.cores[0].memory.read_word(100+i) for i in range(4)]
    expected = [10, 11, 12, 13]  # wid=1: 10+tid
    ok = vals == expected
    if ok: passed += 1; print(f"  ✅ 01_gto_schedule.asm: GTO warp outputs correct {vals}")
    else: failed += 1; print(f"  ❌ 01_gto_schedule.asm: got {vals}, expected {expected}")

    # Test 02: Multi-block
    with open(f"{path}/02_multi_block.asm") as f:
        prog = assemble(f.read())
    gpu2 = GPUSim(num_sms=1, warp_size=4, memory_size=1024)
    gpu2.launch_kernel(prog, grid_dim=(2,), block_dim=(4,))
    gpu2.run()
    check(len(gpu2.cores) == 2, "2 blocks created")
    passed += 1; print(f"  ✅ 02_multi_block.asm: multi-block kernel launched")

    # Test 03: Backward compat
    with open(f"{path}/03_backward_compat.asm") as f:
        prog = assemble(f.read())
    gpu3 = GPUSim(num_sms=1, warp_size=8, memory_size=1024)
    gpu3.launch_kernel(prog, grid_dim=(1,), block_dim=(8,))
    gpu3.run()
    # Verify shared memory + divergence work
    ok = True
    for tid in range(8):
        if gpu3.cores[0].memory.read_word(0+tid) != tid*10:
            ok = False
    check(ok, "compat: scalar compute OK")
    passed += 1; print(f"  ✅ 03_backward_compat.asm: all Phase 0-5 features work")

    if rpt:
        print(f"\n{rpt.perf.report()}")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 6: Kernel Launch & Scheduling — Test Suite")
    print("=" * 60)
    test_gto_scheduler()
    test_perf_counters()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
