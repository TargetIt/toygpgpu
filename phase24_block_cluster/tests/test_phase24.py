#!/usr/bin/env python3
"""Phase 24 Test Suite — Thread Block Cluster"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from block_cluster import *
from isa import OP_DSM_LD, OP_DSM_ST, OP_CBAR, decode, encode_rtype

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_dsm_basic():
    print("\n--- DSM Basic Read/Write Tests ---")
    dsm = DistributedSharedMemory(num_blocks=2, block_size=256)
    check(dsm.num_blocks == 2, "2 blocks in DSM")
    check(dsm.dsm_size() == 512, "total DSM size = 512 words")

    dsm.dsm_write(10, 42)
    check(dsm.dsm_read(10) == 42, "write/read word at offset 10")

    dsm.dsm_write(300, 99)
    check(dsm.dsm_read(300) == 99, "write/read word at offset 300 (block 1)")


def test_dsm_block_slices():
    print("\n--- DSM Block Slice Isolation Tests ---")
    dsm = DistributedSharedMemory(num_blocks=4, block_size=128)

    # Write to each block's slice
    for bid in range(4):
        offset = bid * 128
        dsm.dsm_write(offset + 10, bid * 100)

    # Verify isolation
    for bid in range(4):
        offset = bid * 128
        val = dsm.dsm_read(offset + 10)
        check(val == bid * 100, f"block {bid} slice isolated: {val}")


def test_cluster_barrier():
    print("\n--- ClusterBarrier Tests ---")
    bar = ClusterBarrier(num_blocks=3)
    check(bar.arrived_blocks == 0, "initially no arrivals")

    # First block arrives
    result1 = bar.arrive(0)
    check(not result1, "block 0 arrives, barrier not complete")
    check(bar.arrived_blocks == 1, "1 arrived")

    # Second block arrives
    result2 = bar.arrive(1)
    check(not result2, "block 1 arrives, barrier not complete")

    # Third block arrives
    result3 = bar.arrive(2)
    check(result3, "block 2 arrives, barrier complete")
    check(bar.arrived_blocks == 0, "arrivals reset to 0 after complete")


def test_cluster_sync():
    print("\n--- ThreadBlockCluster Sync Tests ---")
    tc = ThreadBlockCluster(num_blocks=2)
    check(tc.num_blocks == 2, "2 blocks in cluster")

    # Block 0 arrives
    block0_done = tc.cluster_sync(0)
    check(not block0_done, "block 0 sync returns False (not all arrived)")

    # Block 1 arrives
    block1_done = tc.cluster_sync(1)
    check(block1_done, "block 1 sync returns True (all arrived)")


def test_cross_block_load_store():
    print("\n--- Cross-Block DSM Load/Store Tests ---")
    tc = ThreadBlockCluster(num_blocks=2)

    # Block 0 stores data to DSM at offset 0
    data0 = [10, 20, 30]
    tc.dsm_store(0, 0, data0)

    # Block 1 stores data to DSM at offset 256 (block 1's area)
    data1 = [40, 50, 60]
    tc.dsm_store(1, 256, data1)

    # Block 0 loads from DSM offset 256 (block 1's data)
    loaded = tc.dsm_load(0, 0, 256, 3)
    check(loaded == [40, 50, 60], f"block 0 loaded block 1's data: {loaded}")


def test_cross_block_reduce():
    print("\n--- Cross-Block Reduction Tests ---")
    tc = ThreadBlockCluster(num_blocks=3)

    tc.set_block_data(0, [1, 2, 3])
    tc.set_block_data(1, [4, 5, 6])
    tc.set_block_data(2, [7, 8, 9])

    results = tc.cross_block_reduce(op="sum")
    check(results == [6, 15, 24], f"sum reduction: {results}")

    # Verify DSM has results
    check(tc.dsm.dsm_read(0) == 6, "DSM has block 0 sum")
    check(tc.dsm.dsm_read(4) == 15, "DSM has block 1 sum")
    check(tc.dsm.dsm_read(8) == 24, "DSM has block 2 sum")


def test_isa_opcodes():
    print("\n--- ISA Opcode Tests ---")
    check(OP_DSM_LD == 0x44, "OP_DSM_LD = 0x44")
    check(OP_DSM_ST == 0x45, "OP_DSM_ST = 0x45")
    check(OP_CBAR == 0x46, "OP_CBAR = 0x46")

    inst_dsm_ld = decode(encode_rtype(OP_DSM_LD, 0, 0, 0))
    check(inst_dsm_ld.name == "DSM_LD", f"decode DSM_LD: {inst_dsm_ld.name}")

    inst_cbar = decode(encode_rtype(OP_CBAR, 0, 0, 0))
    check(inst_cbar.name == "CBAR", f"decode CBAR: {inst_cbar.name}")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from independent_thread import ReconvergenceEngine, PerThreadPC
    eng = ReconvergenceEngine(0, 4)
    eng.branch_thread(0, 200, True, 101)
    check(len(eng.reconv_points) > 0, "Phase 23 independent thread OK")

    from async_pipeline import AsyncTransactionBarrier
    b = AsyncTransactionBarrier()
    b.increment(2)
    b.decrement(2)
    check(b.wait(), "Phase 22 async barrier OK")

    from mix_precision import float_to_fp16
    check(float_to_fp16(0) == 0, "Phase 21 mixed precision OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 24: Thread Block Cluster — Test Suite")
    print("=" * 60)
    test_dsm_basic()
    test_dsm_block_slices()
    test_cluster_barrier()
    test_cluster_sync()
    test_cross_block_load_store()
    test_cross_block_reduce()
    test_isa_opcodes()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
