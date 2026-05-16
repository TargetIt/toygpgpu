#!/usr/bin/env python3
"""Phase 25 Test Suite — Structured Sparsity"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparsity import *
from isa import OP_SPMMA, OP_SPACK, OP_SUNPACK, decode, encode_rtype

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_sparsity_mask():
    print("\n--- SparsityMask Tests ---")
    # Encode column pairs
    check(SparsityMask.encode((0, 1)) == 0, "cols (0,1) -> pattern 0")
    check(SparsityMask.encode((0, 2)) == 1, "cols (0,2) -> pattern 1")
    check(SparsityMask.encode((2, 3)) == 5, "cols (2,3) -> pattern 5")

    # Decode
    check(SparsityMask.decode(0) == (0, 1), "pattern 0 -> cols (0,1)")
    check(SparsityMask.decode(5) == (2, 3), "pattern 5 -> cols (2,3)")

    # Validation
    check(SparsityMask.is_valid_2to4([1, 2, 0, 0]), "2 non-zero valid")
    check(SparsityMask.is_valid_2to4([1, 2, 3, 0]) == False, "3 non-zero invalid")
    check(SparsityMask.is_valid_2to4([0, 0, 0, 0]) == False, "0 non-zero invalid")


def test_sparsity_find_pattern():
    print("\n--- SparsityMask Find Pattern Tests ---")
    p0 = SparsityMask.find_pattern([1, 2, 0, 0])
    check(p0 == 0, f"pattern [1,2,0,0] -> {p0} (expected 0)")

    p5 = SparsityMask.find_pattern([0, 0, 1, 2])
    check(p5 == 5, f"pattern [0,0,1,2] -> {p5} (expected 5)")


def test_dense_to_sparse():
    print("\n--- Dense to Sparse 2:4 Conversion Tests ---")
    # Already 2:4 structured
    dense = [1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    sparse_vals, masks = dense_to_sparse_2to4(dense)

    check(len(sparse_vals) == 4, "sparse values count = 4 (2 per group)")
    check(len(masks) == 2, "2 groups, 2 masks")
    check(sparse_vals[0] == 1.0, f"first non-zero = 1.0")
    check(sparse_vals[2] == 3.0, f"third non-zero = 3.0")

    # Force 2:4 on dense data
    dense2 = [1.0, 2.0, 3.0, 4.0]  # 4 non-zero -> force to 2
    sparse_vals2, masks2 = dense_to_sparse_2to4(dense2)
    check(len(sparse_vals2) == 2, "forced sparse: 2 values")


def test_sparse_to_dense():
    print("\n--- Sparse to Dense 2:4 Conversion Tests ---")
    sparse_vals = [1.0, 2.0, 3.0, 4.0]
    masks = [0, 5]  # group 0: cols 0,1; group 1: cols 2,3

    dense = sparse_to_dense_2to4(sparse_vals, masks)
    check(dense == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
          f"sparse->dense: {dense}")


def test_sparse_mma():
    print("\n--- Sparse MMA Tests ---")
    # A: 2x4 matrix with 2:4 sparsity
    # Row 0: [1, 2, 0, 0]  mask=0
    # Row 1: [0, 0, 3, 4]  mask=5
    # Compressed: [1, 2, 3, 4], masks=[0, 5]
    a_sparse = [1.0, 2.0, 3.0, 4.0]
    a_masks = [0, 5]

    # B: 4x2 matrix
    b_dense = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # C = A @ B (2x2)
    # Row 0: [1*1+2*3, 1*2+2*4] = [7, 10]
    # Row 1: [3*5+4*7, 3*6+4*8] = [43, 50]
    result = sparse_mma(a_sparse, b_dense, a_masks, m=2, k=4, n=2)

    check(len(result) == 4, f"result has 4 elements: {result}")
    check(abs(result[0] - 7.0) < 0.001, f"C[0,0] = {result[0]} (expected 7)")
    check(abs(result[1] - 10.0) < 0.001, f"C[0,1] = {result[1]} (expected 10)")
    check(abs(result[2] - 43.0) < 0.001, f"C[1,0] = {result[2]} (expected 43)")
    check(abs(result[3] - 50.0) < 0.001, f"C[1,1] = {result[3]} (expected 50)")


def test_sparsity_stats():
    print("\n--- SparsityStats Tests ---")
    ss = SparsityStats()
    ss.record(8, 4)
    check(ss.original_elements == 8, "8 original elements")
    check(ss.sparse_elements == 4, "4 sparse elements")
    check(abs(ss.compression_ratio - 0.5) < 0.001,
           f"compression ratio = {ss.compression_ratio} (expected 0.5)")
    rpt = ss.report()
    check("SparsityStats" in rpt, "report has title")


def test_isa_opcodes():
    print("\n--- ISA Opcode Tests ---")
    check(OP_SPMMA == 0x47, "OP_SPMMA = 0x47")
    check(OP_SPACK == 0x48, "OP_SPACK = 0x48")
    check(OP_SUNPACK == 0x49, "OP_SUNPACK = 0x49")

    inst_spmma = decode(encode_rtype(OP_SPMMA, 0, 0, 0))
    check(inst_spmma.name == "SPMMA", f"decode SPMMA: {inst_spmma.name}")

    inst_spack = decode(encode_rtype(OP_SPACK, 0, 0, 0))
    check(inst_spack.name == "SPACK", f"decode SPACK: {inst_spack.name}")

    inst_sunpack = decode(encode_rtype(OP_SUNPACK, 0, 0, 0))
    check(inst_sunpack.name == "SUNPACK", f"decode SUNPACK: {inst_sunpack.name}")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from block_cluster import DistributedSharedMemory, ThreadBlockCluster
    dsm = DistributedSharedMemory(num_blocks=2)
    dsm.dsm_write(10, 42)
    check(dsm.dsm_read(10) == 42, "Phase 24 DSM OK")

    from independent_thread import ReconvergenceEngine
    eng = ReconvergenceEngine(0, 4)
    check(len(eng.threads) == 4, "Phase 23 independent thread OK")

    from async_pipeline import AsyncTransactionBarrier
    b = AsyncTransactionBarrier()
    b.increment(1); b.decrement(1)
    check(b.wait(), "Phase 22 async barrier OK")

    from mix_precision import PrecisionStats
    ps = PrecisionStats()
    ps.record(2.0, 2.0)
    check(ps.total_conversions == 1, "Phase 21 precision stats OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 25: Structured Sparsity — Test Suite")
    print("=" * 60)
    test_sparsity_mask()
    test_sparsity_find_pattern()
    test_dense_to_sparse()
    test_sparse_to_dense()
    test_sparse_mma()
    test_sparsity_stats()
    test_isa_opcodes()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
