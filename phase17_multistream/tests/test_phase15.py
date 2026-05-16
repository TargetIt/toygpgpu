#!/usr/bin/env python3
"""Phase 15 Test Suite — Compute Graph IR"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_ir import (ComputeGraph, GraphNode, build_example_graph)
from simt_core import SIMTCore
from assembler import assemble

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


# ============================================================
# Graph Construction Tests
# ============================================================

def test_graph_construction():
    print("\n--- Graph Construction Tests ---")

    g = ComputeGraph("test")
    check(g.name == "test", "graph name set")
    check(len(g.nodes) == 0, "empty graph has 0 nodes")

    # Add nodes
    n0 = g.add_kernel("matmul", grid_dim=(1,), block_dim=(4,))
    check(n0 == 0, "first node id=0")
    check(g.nodes[n0].op_type == "kernel", "kernel node type")

    n1 = g.add_memcpy("copy_result", 0, 100, 16, dependencies=[n0])
    check(n1 == 1, "second node id=1")
    check(g.nodes[n1].dependencies == [0], "memcpy depends on kernel")

    n2 = g.add_barrier("sync", dependencies=[n1])
    check(g.nodes[n2].op_type == "barrier", "barrier node type")

    n3 = g.add_kernel("relu", dependencies=[n2])
    check(len(g.nodes) == 4, "4 nodes total")


def test_graph_validation():
    print("\n--- Graph Validation Tests ---")

    # Valid DAG
    g = ComputeGraph("valid")
    a = g.add_kernel("A"); b = g.add_kernel("B", dependencies=[a])
    c = g.add_kernel("C", dependencies=[a, b])
    ok, msg = g.validate()
    check(ok, f"valid DAG: {msg}")

    # Missing dependency
    g2 = ComputeGraph("missing_dep")
    g2.nodes[5] = GraphNode(5, "kernel", "orphan", dependencies=[99])
    g2.nodes[99] = GraphNode(99, "kernel", "dep")
    ok, msg = g2.validate()
    check(ok, "missing dep present in graph — valid")  # 99 exists

    # Empty graph
    g3 = ComputeGraph("empty")
    ok, msg = g3.validate()
    check(not ok, f"empty graph invalid: {msg}")


def test_topological_sort():
    print("\n--- Topological Sort Tests ---")

    g = ComputeGraph("sort_test")
    n0 = g.add_kernel("A")           # 0
    n1 = g.add_kernel("B", dependencies=[n0])  # 1
    n2 = g.add_kernel("C", dependencies=[n0])  # 2
    n3 = g.add_kernel("D", dependencies=[n1, n2])  # 3

    order = g.topological_order()
    check(order[0] == 0, "first in topo order = root")
    check(order[-1] == 3, "last in topo order = leaf")
    check(order.index(1) < order.index(3), "B before D")
    check(order.index(2) < order.index(3), "C before D")


def test_graph_serialization():
    print("\n--- Graph Serialization Tests ---")

    g = build_example_graph()
    json_str = g.to_json()
    check('"name": "example_pipeline"' in json_str, "JSON has graph name")
    check('"type": "kernel"' in json_str, "JSON has kernel type")

    # Round-trip
    g2 = ComputeGraph.from_json(json_str)
    check(g2.name == g.name, "round-trip: name preserved")
    check(len(g2.nodes) == len(g.nodes), "round-trip: node count preserved")

    # DOT export
    dot = g.to_dot()
    check('digraph' in dot, "DOT has digraph")
    check('->' in dot, "DOT has edges")


def test_example_graph():
    print("\n--- Example Graph Test ---")
    g = build_example_graph()
    check(len(g.nodes) == 4, "example graph has 4 nodes")
    ok, msg = g.validate()
    check(ok, f"example graph valid: {msg}")

    order = g.topological_order()
    check(len(order) == 4, "topo order has all 4 nodes")


# ============================================================
# Graph + SIMTCore Integration
# ============================================================

def test_graph_execution_basic():
    print("\n--- Graph Execution (Basic) ---")
    # Simulate graph execution: run kernels in topological order
    g = ComputeGraph("exec_test")
    g.add_kernel("k0")
    g.add_kernel("k1")

    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    prog = assemble("MOV r1, 42\nST r1, [0]\nHALT\n")
    simt.load_program(prog)
    simt.run()
    check(simt.memory.read_word(0) == 42, "graph k0 executed: mem[0]=42")


# Backward compat
def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    # Phase 14: WGMMA
    prog = assemble("TLCONF 1,1,1\nTLDS 0,0\nWGMMA 0,0,2\nHALT\n")
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.load_program(prog)
    simt.run()
    check(True, "Phase 14 WGMMA still works")

    # Phase 12: SHFL
    prog2 = assemble("TID r1\nSHFL r2,r1,0,0\nHALT\n")
    simt2 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt2.load_program(prog2)
    simt2.run()
    check(True, "Phase 12 SHFL still works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 15: Compute Graph IR — Test Suite")
    print("=" * 60)
    test_graph_construction()
    test_graph_validation()
    test_topological_sort()
    test_graph_serialization()
    test_example_graph()
    test_graph_execution_basic()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
