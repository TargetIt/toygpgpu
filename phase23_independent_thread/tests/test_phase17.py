#!/usr/bin/env python3
"""Phase 17 Test Suite — Multi-Stream & Async Execution"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stream import Stream, Event, StreamManager
from copy_engine import CopyEngine, AsyncCopy
from memory import Memory

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_stream_basic():
    print("\n--- Stream Basic Tests ---")
    s = Stream(0)
    check(s.stream_id == 0, "stream_id set")
    check(s.pending() == 0, "new stream has 0 pending")
    s.submit("kernel", {"name": "test"})
    check(s.pending() == 1, "1 command pending")
    cmd = s.pop()
    check(cmd is not None, "pop returns command")
    check(s.pending() == 0, "after pop: 0 pending")


def test_event():
    print("\n--- Event Tests ---")
    ev = Event()
    check(not ev.is_ready(), "new event not ready")
    ev.record(0, 5)
    check(ev.is_ready(), "after record: ready")
    check(ev.stream_id == 0, "stream_id recorded")
    check(ev.timestamp == 5, "timestamp recorded")


def test_stream_event_sync():
    print("\n--- Stream Event Sync Tests ---")
    s1 = Stream(1)
    s2 = Stream(2)
    ev = Event()

    s1.submit("kernel", {"name": "A"})
    s1.record_event(ev)
    s2.wait_event(ev)
    s2.submit("kernel", {"name": "B"})

    check(s1.pending() == 2, "stream1: 2 commands")
    check(s2.pending() == 2, "stream2: 2 commands (wait+kernel)")


def test_stream_manager():
    print("\n--- Stream Manager Tests ---")
    sm = StreamManager(num_streams=3)
    ev = sm.create_event()

    sm.streams[0].submit("kernel", {"name": "A"})
    sm.streams[0].submit("record_event", {"event": ev})
    sm.streams[1].submit("wait_event", {"event": ev})
    sm.streams[1].submit("kernel", {"name": "B"})

    sm.run_all()
    check(sm.stats["commands_executed"] >= 2, "commands executed")
    check(ev.is_ready(), "event recorded after run")


def test_copy_engine():
    print("\n--- Copy Engine Tests ---")
    mem = Memory(256)
    for i in range(8):
        mem.write_word(i, i * 10)

    ce = CopyEngine(bandwidth=2)
    ce.submit(0, 16, 4, 0)  # copy 4 words from addr 0 to 16

    cycle = 0
    while ce.has_pending() and cycle < 10:
        ce.step(mem)
        cycle += 1

    check(cycle == 2, f"4 words at bw=2 takes 2 cycles")
    for i in range(4):
        check(mem.read_word(16 + i) == i * 10, f"mem[16+{i}] = {i*10}")


def test_overlap():
    print("\n--- Compute/Copy Overlap Test ---")
    mem = Memory(256)
    for i in range(16):
        mem.write_word(i, i)

    # Simulate: copy engine copies while SM computes
    ce = CopyEngine(bandwidth=2)
    ce.submit(0, 32, 8, 0)

    sm_active = True
    cycle = 0
    while (ce.has_pending() or sm_active) and cycle < 20:
        ce.step(mem)
        if cycle >= 3:
            sm_active = False
        cycle += 1

    check(cycle <= 7, f"overlap enabled fast completion ({cycle} cycles)")
    check(ce.completed == 1, "copy completed")


def test_isa():
    print("\n--- ISA Tests ---")
    from isa import OP_CPYASYNC, OP_RECORD, OP_WAIT, OPCODE_NAMES
    check(OP_CPYASYNC == 0x39, "OP_CPYASYNC = 0x39")
    check(OP_RECORD == 0x3A, "OP_RECORD = 0x3A")
    check(OP_WAIT == 0x3B, "OP_WAIT = 0x3B")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from graph_ir import ComputeGraph
    g = ComputeGraph("test"); g.add_kernel("A")
    check(g.validate()[0], "Phase 15 graph still works")
    from graph_executor import GraphExecutor
    exec = GraphExecutor(g); exec.run()
    check(exec.stats["kernels_executed"] == 1, "Phase 16 executor still works")
    from stream import Stream
    check(Stream(0).pending() == 0, "Phase 17 stream works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 17: Multi-Stream & Async — Test Suite")
    print("=" * 60)
    test_isa()
    test_stream_basic()
    test_event()
    test_stream_event_sync()
    test_stream_manager()
    test_copy_engine()
    test_overlap()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
