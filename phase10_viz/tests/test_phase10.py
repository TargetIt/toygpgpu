#!/usr/bin/env python3
"""Phase 10 Test Suite — Visualization & Toolchain"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simt_core import SIMTCore
from assembler import assemble
from visualizer import warp_timeline, stall_analysis, memory_heatmap, full_report
from trace_runner import run_with_trace

passed = 0; failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")


def test_timeline():
    print("\n--- Warp Timeline ---")
    # Run a multi-warp program
    src = """
    MOV r1, 10
    ADD r2, r1, r1
    ST r2, [0]
    HALT
    """
    prog = assemble(src)
    simt = SIMTCore(warp_size=4, num_warps=2, memory_size=1024)
    simt.load_program(prog)
    collector = run_with_trace(simt, max_cycles=200)

    timeline = warp_timeline(collector.events, 2, max_cycles=30)
    print(timeline)
    check(len(collector.events) > 0, "events collected")
    check('MOV' in timeline or 'mov' in timeline.lower(), "timeline shows MOV")
    check(len(collector.events) >= 3, f"timeline has {len(collector.events)} events")


def test_stall_analysis():
    print("\n--- Stall Analysis ---")
    # LD latency causes stalls
    src = """
    MOV r1, 42
    ST r1, [10]
    LD r2, [10]
    ADD r3, r2, r2
    ST r3, [0]
    HALT
    """
    prog = assemble(src)
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    collector = run_with_trace(simt, max_cycles=200)

    analysis = stall_analysis(collector.events)
    print(analysis)
    check(collector.stall_cycles > 0, f"{collector.stall_cycles} stall cycles detected")


def test_heatmap():
    print("\n--- Memory Heatmap ---")
    # Program that hits different memory regions
    src = """
    MOV r1, 1
    ST r1, [0]
    ST r1, [10]
    ST r1, [50]
    ST r1, [100]
    LD r2, [0]
    HALT
    """
    prog = assemble(src)
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(prog)
    collector = run_with_trace(simt, max_cycles=200)

    hmap = memory_heatmap(collector.mem_accesses, mem_size=128, width=32)
    print(hmap)
    check(len(collector.mem_accesses) > 0, f"{len(collector.mem_accesses)} mem accesses")


def test_json_export():
    print("\n--- JSON Export ---")
    src = "MOV r1, 10\nADD r2, r1, r1\nST r2, [0]\nHALT\n"
    prog = assemble(src)
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.load_program(prog)
    collector = run_with_trace(simt, max_cycles=200)

    json_path = os.path.join(os.path.dirname(__file__), 'trace_output.json')
    collector.export_json(json_path)
    import json
    with open(json_path) as f:
        data = json.load(f)
    check('events' in data, "JSON has events key")
    check(len(data['events']) > 0, f"{len(data['events'])} events exported")
    os.remove(json_path)


def test_full_report():
    print("\n--- Full Report ---")
    prog = assemble("MOV r1, 5\nADD r2, r1, r1\nST r2, [0]\nHALT\n")
    simt = SIMTCore(warp_size=4, num_warps=2, memory_size=256)
    simt.load_program(prog)
    collector = run_with_trace(simt, max_cycles=200)
    report = full_report(collector, num_warps=2, mem_size=128)
    print(report[:500])
    check('Warp Timeline' in report, "report has timeline")
    check('Stall Analysis' in report, "report has stall analysis")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 10: Visualization & Toolchain — Test Suite")
    print("=" * 60)
    test_timeline()
    test_stall_analysis()
    test_heatmap()
    test_json_export()
    test_full_report()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
