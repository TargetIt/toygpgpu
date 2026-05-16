#!/usr/bin/env python3
"""Phase 18 Test Suite — Performance Model"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perf_model import RooflineModel, PerfAnalyzer

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_roofline_basic():
    print("\n--- Roofline Basic Tests ---")
    rm = RooflineModel(peak_flops=200, peak_bandwidth=100)
    check(rm.peak_flops == 200, "peak_flops=200")
    check(rm.peak_bandwidth == 100, "peak_bandwidth=100")
    check(rm.compute_bound_threshold == 2.0, "ridge point = 2.0 FLOP/Byte")


def test_roofline_classify():
    print("\n--- Roofline Classification Tests ---")
    rm = RooflineModel(peak_flops=100, peak_bandwidth=50)
    # ridge = 100/50 = 2.0
    check(rm.classify(0.5) == "memory-bound", "OI=0.5 → memory-bound")
    check(rm.classify(2.5) == "compute-bound", "OI=2.5 → compute-bound")
    check(rm.classify(2.0) == "compute-bound", "OI=2.0 → compute-bound (at ridge)")


def test_attainable_perf():
    print("\n--- Attainable Performance Tests ---")
    rm = RooflineModel(peak_flops=100, peak_bandwidth=50)
    # OI=1.0: 50*1=50 < 100 → 50
    check(abs(rm.attainable_performance(1.0) - 50.0) < 0.1,
          "OI=1.0 → 50 GFLOPS (memory-bound)")
    # OI=10.0: 100 < 50*10=500 → 100 (capped at peak)
    check(abs(rm.attainable_performance(10.0) - 100.0) < 0.1,
          "OI=10 → 100 GFLOPS (capped at peak)")


def test_roofline_chart():
    print("\n--- Roofline Chart Tests ---")
    rm = RooflineModel(peak_flops=100, peak_bandwidth=50)
    chart = rm.ascii_chart(
        kernels={"matmul": 4.0, "add": 0.5},
        width=40, height=10
    )
    check("Roofline Model Chart" in chart, "chart has title")
    check("matmul" in chart, "chart shows matmul point")
    check("memory-bound" in chart or "compute-bound" in chart,
          "chart shows classification")


def test_perf_analyzer():
    print("\n--- Perf Analyzer Tests ---")
    rm = RooflineModel(peak_flops=100, peak_bandwidth=50)
    pa = PerfAnalyzer(rm)

    # Memory-bound kernel: high bytes, low flops
    r1 = pa.analyze("vector_add", total_flops=8, bytes_read=64, bytes_written=32)
    check(r1["classification"] == "memory-bound", "vector_add is memory-bound")
    check(len(r1["suggestions"]) >= 1, "memory-bound has suggestions")

    # Compute-bound kernel: high flops, low bytes
    r2 = pa.analyze("matmul_tiled", total_flops=512, bytes_read=64, bytes_written=16)
    check(r2["classification"] == "compute-bound", "matmul is compute-bound")


def test_perf_report():
    print("\n--- Perf Report Tests ---")
    rm = RooflineModel(peak_flops=100, peak_bandwidth=50)
    pa = PerfAnalyzer(rm)
    results = [
        pa.analyze("k1", total_flops=100, bytes_read=1000, bytes_written=500),
        pa.analyze("k2", total_flops=500, bytes_read=100, bytes_written=50),
    ]
    report = pa.report(results)
    check("Performance Analysis Report" in report, "report has title")
    check("k1" in report and "k2" in report, "report shows both kernels")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from stream import Stream, StreamManager
    sm = StreamManager(2)
    check(len(sm.streams) == 2, "Phase 17 StreamManager works")
    from graph_ir import ComputeGraph
    from graph_executor import GraphExecutor
    g = ComputeGraph("t"); g.add_kernel("k")
    exec = GraphExecutor(g); exec.run()
    check(exec.stats["kernels_executed"] == 1, "Phase 16 executor works")
    from cutile_parser import parse_cutile
    k = parse_cutile("tile M=2,N=2,K=1\nkernel t(A:[M,K],B:[K,N],C:[M,N]){load A[0:M,0:K]->smem[0]}")
    check(k.name == "t", "Phase 14 CuTile works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 18: Performance Model — Test Suite")
    print("=" * 60)
    test_roofline_basic()
    test_roofline_classify()
    test_attainable_perf()
    test_roofline_chart()
    test_perf_analyzer()
    test_perf_report()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
