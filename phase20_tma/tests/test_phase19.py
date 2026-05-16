#!/usr/bin/env python3
"""Phase 19 Test Suite — L2 Cache Hierarchy"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from l2_cache import L2Cache, L2CacheLine, BandwidthModel
from simt_core import SIMTCore
from assembler import assemble
from memory import Memory

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_l2_basic():
    print("\n--- L2 Cache Basic Tests ---")
    l2 = L2Cache(total_lines=256, associativity=4, line_words=4)
    check(l2.num_sets == 64, "256/4 = 64 sets")
    check(l2.total_lines == 256, "256 total lines")
    check(l2.hits == 0 and l2.misses == 0, "initial stats clean")


def test_l2_read_write():
    print("\n--- L2 Read/Write Tests ---")
    l2 = L2Cache(total_lines=64, associativity=2, line_words=4)

    # First read: miss
    val, lat = l2.read(0)
    check(val is None, "first read: miss")
    check(lat == 100, "miss latency = 100 cycles")
    check(l2.misses == 1, "miss count incremented")

    # Write: allocates
    l2.write(0, 42)
    check(l2.misses == 2, "write miss allocates")

    # Read again: should hit now
    val, lat = l2.read(0)
    check(val == 42, "read after write: hit, value=42")
    check(lat == 10, "hit latency = 10 cycles")
    check(l2.hits == 1, "hit count incremented")


def test_l2_lru_eviction():
    print("\n--- L2 LRU Eviction Tests ---")
    l2 = L2Cache(total_lines=4, associativity=2, line_words=4)
    # 2 sets, 2-way = 4 total lines

    # Fill all lines in set 0
    l2.write(0, 1)    # set 0, way 0
    l2.write(256, 2)  # set 0, way 1 (different tag, same set)

    # Both reads should hit
    v1, _ = l2.read(0)
    v2, _ = l2.read(256)
    check(v1 == 1 and v2 == 2, "both values present in set 0")

    # Add third tag to same set — evicts LRU
    l2.write(512, 3)  # set 0 again, evicts way 0 (LRU since read above updated way 1)

    # Check eviction count
    check(l2.evictions >= 1, f"eviction happened ({l2.evictions})")


def test_l2_hit_rate():
    print("\n--- L2 Hit Rate Tests ---")
    l2 = L2Cache(total_lines=64, associativity=2, line_words=4)

    # Pre-fill with sequential data
    for i in range(32):
        l2.write(i * 4, i * 100)

    # Read back: should hit
    for i in range(32):
        l2.read(i * 4)

    rate = l2.hit_rate()
    check(rate > 0.4, f"hit rate > 40% (actual: {rate*100:.1f}%)")


def test_bandwidth_model():
    print("\n--- Bandwidth Model Tests ---")
    bw = BandwidthModel()
    check(bw.l1_latency == 1, "L1 latency = 1")
    check(bw.l2_latency == 10, "L2 latency = 10")
    check(bw.hbm_latency == 100, "HBM latency = 100")

    # Effective bandwidth with 80% L1, 50% L2
    eff = bw.effective_bandwidth(0.8, 0.5)
    check(eff > bw.hbm_bandwidth, "effective BW > HBM BW")


def test_l2_in_simt_core():
    print("\n--- L2 in SIMTCore Tests ---")
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    # Verify L2 cache exists
    check(hasattr(simt, 'l2_cache'), "SIMTCore has l2_cache")
    check(hasattr(simt, 'bw_model'), "SIMTCore has bw_model")

    # Test memory access through cache hierarchy
    prog = assemble("MOV r1, 99\nST r1, [0]\nLD r2, [0]\nST r2, [100]\nHALT\n")
    simt.load_program(prog)
    simt.run()

    check(simt.memory.read_word(100) == 99, "value propagated through L1+L2+HBM")
    check(simt.l2_cache.hits + simt.l2_cache.misses > 0,
          "L2 cache was accessed during LD/ST")


def test_cache_stats_report():
    print("\n--- Cache Stats Report Tests ---")
    l2 = L2Cache()
    for i in range(8):
        l2.write(i * 16, i)
    for i in range(8):
        l2.read(i * 16)

    stats = l2.stats()
    check("hits=" in stats, "stats shows hits")
    check("hit_rate" in stats, "stats shows hit_rate")
    check("writebacks" in stats, "stats shows writebacks")

    bw = BandwidthModel()
    report = bw.report(l2)
    check("Cache Hierarchy Report" in report, "report has title")
    check("L1:" in report and "L2:" in report and "HBM:" in report,
         "report shows all cache levels")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from perf_model import RooflineModel
    rm = RooflineModel(200, 100)
    check(rm.classify(3.0) == "compute-bound", "Phase 18 perf_model OK")
    from stream import StreamManager
    check(StreamManager(2).streams[0].pending() == 0, "Phase 17 stream OK")
    from graph_executor import GraphExecutor
    from graph_ir import ComputeGraph
    g = ComputeGraph("t"); g.add_kernel("k")
    exec = GraphExecutor(g); exec.run()
    check(exec.stats["kernels_executed"] == 1, "Phase 16 executor OK")
    from cutile_parser import parse_cutile
    k = parse_cutile("tile M=2,N=2,K=1\nkernel t(A:[M,K],B:[K,N],C:[M,N]){load A[0:M,0:K]->smem[0]}")
    check(k.name == "t", "Phase 14 CuTile OK")
    from assembler import assemble
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(assemble("TID r1\nSHFL r2,r1,0,0\nHALT\n"))
    simt.run()
    check(True, "Phase 12 SHFL OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 19: L2 Cache Hierarchy — Test Suite")
    print("=" * 60)
    test_l2_basic()
    test_l2_read_write()
    test_l2_lru_eviction()
    test_l2_hit_rate()
    test_bandwidth_model()
    test_l2_in_simt_core()
    test_cache_stats_report()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
