#!/usr/bin/env python3
"""Phase 22 Test Suite — Async Pipeline"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from async_pipeline import *
from isa import OP_ABAR, OP_PCOMMIT, decode, encode_rtype

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_async_barrier():
    print("\n--- AsyncTransactionBarrier Tests ---")
    b = AsyncTransactionBarrier()
    check(not b.wait(), "initially not completed")

    b.increment(3)
    check(not b.wait(), "pending transactions -> not completed")

    b.decrement(2)
    check(not b.wait(), "still 1 pending")

    b.decrement(1)
    check(b.wait(), "all done -> completed")

    b.reset()
    check(not b.wait(), "after reset not completed")
    check(b.phase == 1, "phase incremented after reset")


def test_pipeline_stage_lifecycle():
    print("\n--- PipelineStage Tests ---")
    stage = PipelineStage(PipelineStageType.LOAD, 0, 16)
    check(stage.type == PipelineStageType.LOAD, "correct stage type")

    barrier = stage.start_load()
    check(not barrier.wait(), "start_load -> pending")

    stage.complete_load()
    check(barrier.wait(), "complete_load -> done")

    stage.commit()
    check(stage.is_ready(), "committed and completed -> ready")


def test_compute_stage():
    print("\n--- PipelineStage Compute Tests ---")
    stage = PipelineStage(PipelineStageType.COMPUTE, 0, 16)
    data = [1, 2, 3, 4]
    result = stage.execute_compute(data)
    check(result == [2, 4, 6, 8], "compute stage doubles values")
    check(stage.is_ready(), "compute stage always ready")


def test_producer_consumer_basic():
    print("\n--- Producer/Consumer Basic Tests ---")
    pp = ProducerConsumerPipeline(num_stages=3)

    # Producer loads data into stage 0 (LOAD)
    data = [10, 20, 30, 40]
    barrier = pp.producer_load(0, data)
    check(barrier.wait(), "producer load completed")
    pp.producer_commit(0)

    # Consumer waits and computes on stage 1 (COMPUTE)
    check(pp.consumer_wait(1), "consumer sees completed barrier")
    result = pp.consumer_compute(1)
    check(result == [20, 40, 60, 80], "consumer compute doubles values")


def test_pipeline_advance():
    print("\n--- Pipeline Advance Tests ---")
    pp = ProducerConsumerPipeline(num_stages=2)
    check(pp.iteration == 0, "initial iteration = 0")

    pp.producer_load(0, [1, 2, 3])
    pp.producer_commit(0)
    pp.advance()
    check(pp.iteration == 1, "advanced to iteration 1")

    # After advance, stages should be reset
    for stage in pp.stages:
        check(not stage.committed, "stage not committed after advance")
        check(stage.iteration == 1, f"stage at iteration 1")


def test_invalid_stage_raises():
    print("\n--- Invalid Stage Error Tests ---")
    load_stage = PipelineStage(PipelineStageType.LOAD)
    compute_stage = PipelineStage(PipelineStageType.COMPUTE)

    try:
        compute_stage.start_load()
        check(False, "should raise on compute.start_load()")
    except RuntimeError:
        check(True, "compute.start_load() raises RuntimeError")

    try:
        load_stage.execute_compute([1, 2])
        check(False, "should raise on load.execute_compute()")
    except RuntimeError:
        check(True, "load.execute_compute() raises RuntimeError")


def test_isa_opcodes():
    print("\n--- ISA Opcode Tests ---")
    check(OP_ABAR == 0x40, "OP_ABAR = 0x40")
    check(OP_PCOMMIT == 0x41, "OP_PCOMMIT = 0x41")

    # Verify decode of encoded instructions
    inst_abar = decode(encode_rtype(OP_ABAR, 0, 0, 0))
    check(inst_abar.name == "ABAR", f"decode ABAR: {inst_abar.name}")

    inst_pcommit = decode(encode_rtype(OP_PCOMMIT, 0, 0, 0))
    check(inst_pcommit.name == "PCOMMIT", f"decode PCOMMIT: {inst_pcommit.name}")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from tma import TMAEngine, TensorDescriptor
    tma = TMAEngine()
    desc = TensorDescriptor([2, 2])
    tma.register_descriptor(0, desc)
    check(len(tma.compute_addresses(0, [0, 0], [2, 2])) == 4, "Phase 20 TMA OK")

    from mix_precision import float_to_fp16, fp16_to_float
    check(float_to_fp16(0) == 0, "Phase 21 mixed precision OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 22: Async Pipeline — Test Suite")
    print("=" * 60)
    test_async_barrier()
    test_pipeline_stage_lifecycle()
    test_compute_stage()
    test_producer_consumer_basic()
    test_pipeline_advance()
    test_invalid_stage_raises()
    test_isa_opcodes()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
