#!/usr/bin/env python3
"""Phase 20 Test Suite — TMA (Tensor Memory Accelerator)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tma import TMAEngine, TensorDescriptor
from memory import Memory
from shared_memory import SharedMemory

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_tensor_descriptor():
    print("\n--- Tensor Descriptor Tests ---")
    # 3x4 matrix
    desc = TensorDescriptor([3, 4], base_addr=0, element_size=4)
    check(desc.ndim == 2, "2D tensor")
    check(desc.strides == [4, 1], "row-major strides=[4,1]")
    check(desc.linear_addr([0, 0]) == 0, "[0,0] → 0")
    check(desc.linear_addr([1, 2]) == 6, "[1,2] → 1*4 + 2 = 6")
    check(desc.linear_addr([2, 3]) == 11, "[2,3] → 2*4 + 3 = 11")


def test_tile_bounds():
    print("\n--- Tile Bounds Tests ---")
    desc = TensorDescriptor([4, 4])
    bounds = desc.tile_bounds([0, 0], [2, 2])
    check(bounds['total_elements'] == 4, "2x2 tile in 4x4 = 4 elements")

    # Tile out of bounds
    bounds2 = desc.tile_bounds([3, 3], [2, 2])
    check(bounds2['total_elements'] == 1, "corner tile: only 1 element valid")


def test_tma_addresses():
    print("\n--- TMA Address Generation Tests ---")
    tma = TMAEngine()
    desc = TensorDescriptor([2, 2], base_addr=0)
    tma.register_descriptor(0, desc)

    addrs = tma.compute_addresses(0, [0, 0], [2, 2])
    check(len(addrs) == 4, "2x2 tile → 4 addresses")
    check(addrs == [0, 1, 2, 3], "row-major: [0,1,2,3]")

    # OOB tile
    addrs2 = tma.compute_addresses(0, [1, 1], [2, 2])
    check(-1 in addrs2, "OOB elements marked as -1")
    check(tma.stats["boundary_elements"] > 0, "boundary counted")


def test_tma_load():
    print("\n--- TMA Load Tests ---")
    mem = Memory(256)
    smem = SharedMemory(64)
    # Fill global memory with pattern
    for i in range(16):
        mem.write_word(i, i * 10)

    desc = TensorDescriptor([4, 4], base_addr=0)
    tma = TMAEngine()
    tma.register_descriptor(0, desc)

    # Load 2x2 tile from [1,1]
    count = tma.tma_load(0, [1, 1], [2, 2], 0, mem, smem)
    check(count == 4, "4 elements loaded")
    check(smem.read_word(0) == 50, "smem[0] = mem[1*4+1=5]*10 = 50")
    check(smem.read_word(1) == 60, "smem[1] = mem[6] = 60")


def test_tma_store():
    print("\n--- TMA Store Tests ---")
    mem = Memory(256)
    smem = SharedMemory(64)
    for i in range(4):
        smem.write_word(i, i + 100)

    desc = TensorDescriptor([2, 2], base_addr=10)
    tma = TMAEngine()
    tma.register_descriptor(0, desc)

    count = tma.tma_store(0, [0, 0], [2, 2], 0, mem, smem)
    check(count == 4, "4 elements stored")
    check(mem.read_word(10) == 100, "mem[10] = 100")
    check(mem.read_word(11) == 101, "mem[11] = 101")


def test_tma_boundary_handling():
    print("\n--- TMA Boundary Handling Tests ---")
    mem = Memory(256)
    smem = SharedMemory(64)
    for i in range(4):
        mem.write_word(i, i + 1)

    desc = TensorDescriptor([2, 2], base_addr=0)
    tma = TMAEngine()
    tma.register_descriptor(0, desc)

    # Load 3x3 tile on a 2x2 tensor — OOB on right/bottom
    count = tma.tma_load(0, [0, 0], [3, 3], 0, mem, smem)
    check(count == 9, "9 positions, including OOB (zero-padded)")
    check(smem.read_word(0) == 1, "valid smem[0]=1")
    check(smem.read_word(4) == 0, "OOB smem[4]=0 (zero-padded)")


def test_isa():
    print("\n--- TMA ISA Tests ---")
    from isa import OP_TMA_LOAD, OP_TMA_STORE
    check(OP_TMA_LOAD == 0x3C, "OP_TMA_LOAD = 0x3C")
    check(OP_TMA_STORE == 0x3D, "OP_TMA_STORE = 0x3D")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from l2_cache import L2Cache, BandwidthModel
    l2 = L2Cache()
    l2.write(0, 42)
    v, _ = l2.read(0)
    check(v == 42, "Phase 19 L2 cache OK")
    from simt_core import SIMTCore
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    check(hasattr(simt, 'l2_cache'), "SIMTCore has L2")
    check(hasattr(simt, 'tma'), "SIMTCore has TMA")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 20: TMA — Test Suite")
    print("=" * 60)
    test_tensor_descriptor()
    test_tile_bounds()
    test_tma_addresses()
    test_tma_load()
    test_tma_store()
    test_tma_boundary_handling()
    test_isa()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
