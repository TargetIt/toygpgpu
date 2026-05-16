#!/usr/bin/env python3
"""Phase 16 Test Suite — Graph Scheduling & Optimization"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_ir import ComputeGraph
from graph_executor import (GraphExecutor, fuse_kernels, plan_memory)

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_executor_basic():
    print("\n--- Graph Executor Basic Tests ---")
    g = ComputeGraph("exec_test")
    g.add_kernel("A"); g.add_kernel("B"); g.add_kernel("C")
    exec = GraphExecutor(g, memory_size=256)
    exec.run()
    check(exec.stats["total_ops"] == 3, "executed 3 kernel nodes")

    # Memcpy test
    g2 = ComputeGraph("copy_test")
    a = g2.add_kernel("producer")
    b = g2.add_memcpy("copy", 0, 10, 4, dependencies=[a])
    exec2 = GraphExecutor(g2)
    exec2.run()
    check(exec2.stats["memcpys_executed"] == 1, "executed memcpy")

    # Barrier test
    g3 = ComputeGraph("barrier_test")
    g3.add_kernel("A"); g3.add_barrier("sync"); g3.add_kernel("B")
    exec3 = GraphExecutor(g3)
    exec3.run()
    check(exec3.stats["barriers_hit"] == 1, "executed barrier")


def test_critical_path():
    print("\n--- Critical Path Tests ---")
    g = ComputeGraph("path_test")
    n0 = g.add_kernel("A"); n1 = g.add_kernel("B", dependencies=[n0])
    n2 = g.add_kernel("C"); n3 = g.add_kernel("D", dependencies=[n1, n2])
    exec = GraphExecutor(g)
    cp = exec.get_critical_path()
    check(len(cp) >= 3, f"critical path length = {len(cp)}")
    check(cp[0] == 0 or cp[0] == 2, "critical path starts from entry node")


def test_concurrent_groups():
    print("\n--- Concurrent Groups Tests ---")
    g = ComputeGraph("concurrent")
    n0 = g.add_kernel("A")           # level 0
    n1 = g.add_kernel("B", dependencies=[n0])  # level 1
    n2 = g.add_kernel("C", dependencies=[n0])  # level 1 (parallel with B)
    n3 = g.add_kernel("D", dependencies=[n1, n2])  # level 2
    exec = GraphExecutor(g)
    groups = exec.concurrent_groups()
    check(len(groups) == 3, f"3 concurrent groups (got {len(groups)})")
    check(len(groups[0]) == 1, "level 0: 1 node")
    check(len(groups[1]) == 2, "level 1: 2 nodes (parallel)")


def test_kernel_fusion():
    print("\n--- Kernel Fusion Tests ---")
    g = ComputeGraph("fusion_test")
    a = g.add_kernel("relu")
    b = g.add_kernel("bias_add", dependencies=[a])
    c = g.add_kernel("scale", dependencies=[b])

    fused = fuse_kernels(g)
    check(len(fused.nodes) < len(g.nodes),
          f"fusion reduced nodes: {len(g.nodes)} → {len(fused.nodes)}")


def test_memory_planning():
    print("\n--- Memory Planning Tests ---")
    g = ComputeGraph("mem_plan")
    a = g.add_kernel("A")
    b = g.add_kernel("B", dependencies=[a])
    g.add_kernel("C", dependencies=[b])

    assignments = plan_memory(g, memory_size=64)
    check(len(assignments) == 3, "all 3 nodes assigned memory")
    check(assignments[a] == 0, "first node gets addr 0")


def test_executor_report():
    print("\n--- Executor Report Tests ---")
    g = ComputeGraph("report_test")
    g.add_kernel("A"); g.add_kernel("B")
    exec = GraphExecutor(g)
    exec.run()
    rpt = exec.report()
    check("Nodes: 2" in rpt, "report shows node count")
    check("Kernels executed: 2" in rpt, "report shows kernel count")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from graph_ir import build_example_graph
    g = build_example_graph()
    ok, msg = g.validate()
    check(ok, "Phase 15 graph validation still works")
    from cutile_parser import parse_cutile
    kernel = parse_cutile("tile M=2,N=2,K=1\nkernel test(A:[M,K],B:[K,N],C:[M,N]){load A[0:M,0:K]->smem[0]}")
    check(kernel.name == "test", "Phase 14 CuTile still works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 16: Graph Scheduling — Test Suite")
    print("=" * 60)
    test_executor_basic()
    test_critical_path()
    test_concurrent_groups()
    test_kernel_fusion()
    test_memory_planning()
    test_executor_report()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
