#!/usr/bin/env python3
"""Phase 23 Test Suite — Independent Thread Scheduling"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from independent_thread import *
from isa import OP_TBRANCH, OP_TRECONV, decode, encode_rtype

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_per_thread_pc():
    print("\n--- PerThreadPC Tests ---")
    tp = PerThreadPC(thread_id=5, pc=100)
    check(tp.thread_id == 5, "thread_id = 5")
    check(tp.pc == 100, "initial pc = 100")
    check(tp.active, "initially active")

    # Branch taken
    tp.branch(200, True)
    check(tp.next_pc == 200, "branch taken -> next_pc = 200")

    # Branch not taken
    tp.branch(300, False)
    check(tp.next_pc == tp.pc + 1, "branch not taken -> next_pc = pc+1")

    # Step
    old = tp.pc
    tp.step()
    check(tp.pc == old + 1, "step advances pc by 1")


def test_reconv_engine_init():
    print("\n--- ReconvergenceEngine Init Tests ---")
    eng = ReconvergenceEngine(warp_id=3, num_threads=32)
    check(eng.warp_id == 3, "warp_id = 3")
    check(len(eng.threads) == 32, "32 threads created")
    check(eng.get_pc_diversity() == 1, "all threads at same PC initially")


def test_reconv_engine_set_pc():
    print("\n--- ReconvergenceEngine Set PC Tests ---")
    eng = ReconvergenceEngine(0, 32)
    eng.set_pc_all(42)
    for tid, t in eng.threads.items():
        check(t.pc == 42, f"thread {tid} pc = 42")
    check(eng.get_pc_diversity() == 1, "uniform after set_pc_all")


def test_thread_independent_branch():
    print("\n--- Independent Thread Branch Tests ---")
    eng = ReconvergenceEngine(0, 4)  # 4 threads for simplicity
    eng.set_pc_all(100)

    # Thread 0 and 2 take branch to 200, others fall through
    eng.branch_thread(0, 200, True, 101)
    eng.branch_thread(1, 200, False, 101)
    eng.branch_thread(2, 200, True, 101)
    eng.branch_thread(3, 200, False, 101)

    check(eng.threads[0].next_pc == 200, "thread 0 takes branch")
    check(eng.threads[1].next_pc == 101, "thread 1 falls through")
    check(eng.threads[2].next_pc == 200, "thread 2 takes branch")
    check(eng.threads[3].next_pc == 101, "thread 3 falls through")

    # Check divergence tracking
    check(len(eng.reconv_points) == 1, "one reconv point tracked")
    check(101 in eng.reconv_points, "reconv at PC 101")
    check(eng.reconv_points[101] == {0, 2}, "threads 0,2 in divergent set")


def test_reconvergence():
    print("\n--- Reconvergence Tests ---")
    eng = ReconvergenceEngine(0, 4)
    eng.set_pc_all(100)

    # Divergent branch: threads 0,1 take branch to 200, reconv at 102
    eng.branch_thread(0, 200, True, 102)
    eng.branch_thread(1, 200, True, 102)

    # Threads 2,3 fall through to 101
    eng.threads[2].next_pc = 101
    eng.threads[3].next_pc = 101

    # Advance threads 2,3 to PC 101
    eng.threads[2].step()
    eng.threads[3].step()

    # Advance threads 0,1 to PC 200
    eng.threads[0].step()
    eng.threads[1].step()

    # No reconvergence at 101 (divergent threads are at 200)
    result_101 = eng.check_reconvergence(101)
    check(len(result_101) == 0, "no reconv at 101 (divergent at 200)")

    # Move threads 0,1 from 200 to 201, etc., finally to 102
    eng.threads[0].next_pc = 201
    eng.threads[1].next_pc = 201
    eng.threads[0].step()
    eng.threads[1].step()

    # Now threads 0,1 go to 102 (reconv point)
    eng.threads[0].next_pc = 102
    eng.threads[1].next_pc = 102

    # Step to reach 102
    eng.threads[0].step()
    eng.threads[1].step()

    # Now check reconvergence at 102
    result_102 = eng.check_reconvergence(102)
    check(len(result_102) == 2, f"reconv at 102: {result_102} got {len(result_102)} threads")


def test_divergent_mask():
    print("\n--- Divergent Mask Tests ---")
    eng = ReconvergenceEngine(0, 4)
    eng.set_pc_all(100)

    # Thread 0 branches to 200, others at 100
    eng.branch_thread(0, 200, True, 101)

    check(eng.get_divergent_mask(100) & 0b1110 == 0b1110,
          "threads 1,2,3 at PC 100")
    check(eng.get_divergent_mask(200) & 1 == 1,
          "thread 0 at PC 200")


def test_isa_opcodes():
    print("\n--- ISA Opcode Tests ---")
    check(OP_TBRANCH == 0x42, "OP_TBRANCH = 0x42")
    check(OP_TRECONV == 0x43, "OP_TRECONV = 0x43")

    inst_tbranch = decode(encode_rtype(OP_TBRANCH, 0, 0, 0))
    check(inst_tbranch.name == "TBRANCH", f"decode TBRANCH: {inst_tbranch.name}")

    inst_treconv = decode(encode_rtype(OP_TRECONV, 0, 0, 0))
    check(inst_treconv.name == "TRECONV", f"decode TRECONV: {inst_treconv.name}")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from async_pipeline import AsyncTransactionBarrier, ProducerConsumerPipeline
    b = AsyncTransactionBarrier()
    b.increment(1)
    b.decrement(1)
    check(b.wait(), "Phase 22 async barrier OK")

    from mix_precision import PrecisionStats
    ps = PrecisionStats()
    ps.record(1.0, 1.1)
    check(ps.total_conversions == 1, "Phase 21 precision stats OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 23: Independent Thread Scheduling — Test Suite")
    print("=" * 60)
    test_per_thread_pc()
    test_reconv_engine_init()
    test_reconv_engine_set_pc()
    test_thread_independent_branch()
    test_reconvergence()
    test_divergent_mask()
    test_isa_opcodes()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
