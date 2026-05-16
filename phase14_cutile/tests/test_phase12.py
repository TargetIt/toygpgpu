#!/usr/bin/env python3
"""Phase 12 Test Suite — Warp Communication + Console"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simt_core import SIMTCore
from assembler import assemble
from isa import (OPCODE_NAMES, decode, OP_SHFL, OP_VOTE, OP_BALLOT,
                 OP_SETP, PRED_FLAG)

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def run_prog(f, checks, warp_size=8, num_warps=1):
    """Run an assembly program and verify memory results."""
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path, encoding='utf-8') as fp:
        prog = assemble(fp.read())
    simt = SIMTCore(warp_size=warp_size, num_warps=num_warps, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    if callable(checks):
        ok, msg = checks(simt)
        if ok: passed += 1; print(f"  OK {f}: {msg}")
        else: failed += 1; print(f"  FAIL {f}: {msg}")
    else:
        for addr, expected in checks.items():
            actual = simt.memory.read_word(addr)
            if actual == (expected & 0xFFFFFFFF):
                passed += 1; print(f"  OK {f}: mem[{addr}] = {expected}")
            else:
                failed += 1; print(f"  FAIL {f}: mem[{addr}] = {actual}, expected {expected}")


# ============================================================
# ISA Unit Tests
# ============================================================

def test_isa_warp_comm():
    print("\n--- Warp Communication ISA Tests ---")
    check(OP_SHFL == 0x30, "OP_SHFL = 0x30")
    check(OP_VOTE == 0x33, "OP_VOTE = 0x33")
    check(OP_BALLOT == 0x34, "OP_BALLOT = 0x34")
    check(OPCODE_NAMES[OP_SHFL] == "SHFL", "SHFL in OPCODE_NAMES")
    check(OPCODE_NAMES[OP_VOTE] == "VOTE", "VOTE in OPCODE_NAMES")
    check(OPCODE_NAMES[OP_BALLOT] == "BALLOT", "BALLOT in OPCODE_NAMES")


def test_assembler_warp_comm():
    print("\n--- Assembler Warp Comm Tests ---")

    # SHFL
    src = "SHFL r1, r2, 3, 0\nHALT\n"
    prog = assemble(src)
    instr = decode(prog[0])
    check(instr.opcode == OP_SHFL, "SHFL encode/decode opcode")
    check(instr.rd == 1, "SHFL rd=r1")
    check(instr.rs1 == 2, "SHFL rs1=r2")
    check(instr.rs2 == 3, "SHFL src_lane=3")
    check(instr.imm == 0, "SHFL mode=0 (IDX)")

    # SHFL with mode
    src2 = "SHFL r5, r3, 1, 2\nHALT\n"
    prog2 = assemble(src2)
    instr2 = decode(prog2[0])
    check(instr2.imm == 2, "SHFL mode=2 (DOWN)")

    # VOTE.ANY
    src3 = "VOTE.ANY r1, r2\nHALT\n"
    prog3 = assemble(src3)
    instr3 = decode(prog3[0])
    check(instr3.opcode == OP_VOTE, "VOTE.ANY encode/decode")
    check(instr3.imm == 0, "VOTE.ANY imm[0]=0")

    # VOTE.ALL
    src4 = "VOTE.ALL r3, r4\nHALT\n"
    prog4 = assemble(src4)
    instr4 = decode(prog4[0])
    check(instr4.opcode == OP_VOTE, "VOTE.ALL encode/decode")
    check(instr4.imm == 1, "VOTE.ALL imm[0]=1")

    # BALLOT
    src5 = "BALLOT r5, r6\nHALT\n"
    prog5 = assemble(src5)
    instr5 = decode(prog5[0])
    check(instr5.opcode == OP_BALLOT, "BALLOT encode/decode")


# ============================================================
# SHFL Demo Tests
# ============================================================

def test_shfl_program():
    print("\n--- SHFL Program Test ---")

    def check_shfl(simt):
        warp_size = 8
        # mem[0+tid] = tid * 10 (original values)
        for tid in range(warp_size):
            if simt.memory.read_word(0 + tid) != tid * 10:
                return False, f"mem[{0+tid}] != {tid*10}"
        # mem[8+tid] = 0 (SHFL IDX to thread 0 → all get 0)
        for tid in range(warp_size):
            if simt.memory.read_word(8 + tid) != 0:
                return False, f"SHFL IDX: mem[{8+tid}] != 0"
        # mem[16+tid] = (tid+1)%8 * 10 (SHFL DOWN 1)
        for tid in range(warp_size):
            expected = ((tid + 1) % warp_size) * 10
            if simt.memory.read_word(16 + tid) != expected:
                return False, f"SHFL DOWN: mem[{16+tid}] != {expected}"
        # mem[24+tid] = (tid^1) * 10 (SHFL XOR 1)
        for tid in range(warp_size):
            expected = (tid ^ 1) * 10
            if simt.memory.read_word(24 + tid) != expected:
                return False, f"SHFL XOR: mem[{24+tid}] != {expected}"
        return True, "all SHFL modes correct"

    run_prog("09_warp_shfl.asm", check_shfl)


# ============================================================
# VOTE & BALLOT Demo Tests
# ============================================================

def test_vote_program():
    print("\n--- VOTE & BALLOT Program Test ---")

    def check_vote(simt):
        # mem[0] = VOTE.ANY on even=1/odd=0 → 1 (some non-zero)
        if simt.memory.read_word(0) != 1:
            return False, f"VOTE.ANY: mem[0] = {simt.memory.read_word(0)}, expected 1"
        # mem[1] = VOTE.ALL on even=1/odd=0 → 0 (not all non-zero)
        if simt.memory.read_word(1) != 0:
            return False, f"VOTE.ALL: mem[1] = {simt.memory.read_word(1)}, expected 0"
        # mem[2] = BALLOT → 0b01010101 = 0x55 (warp_size=8)
        if simt.memory.read_word(2) != 0x55:
            return False, f"BALLOT: mem[2] = 0x{simt.memory.read_word(2):02X}, expected 0x55"
        # mem[8] = VOTE.ANY on all=1 → 1
        if simt.memory.read_word(8) != 1:
            return False, f"VOTE.ANY all-1: mem[8] = {simt.memory.read_word(8)}, expected 1"
        # mem[9] = VOTE.ALL on all=1 → 1
        if simt.memory.read_word(9) != 1:
            return False, f"VOTE.ALL all-1: mem[9] = {simt.memory.read_word(9)}, expected 1"
        return True, "VOTE/BALLOT correct"

    run_prog("10_warp_vote.asm", check_vote)


# ============================================================
# Backward Compat: Phase 11 functionality still works
# ============================================================

def test_backward_compat():
    print("\n--- Backward Compat Tests ---")

    # Test divergence still works
    path = os.path.join(os.path.dirname(__file__), 'programs', 'demo_divergence.asm')
    with open(path, encoding='utf-8') as f:
        prog = assemble(f.read())
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(prog)
    simt.run()
    check(simt.memory.read_word(100) == 2, "divergence: mem[100]=2 (even)")
    check(simt.memory.read_word(101) == 1, "divergence: mem[101]=1 (odd)")
    check(simt.memory.read_word(201) == 2, "divergence: mem[201]=2 (reconv)")

    # Test basic ALU still works
    prog2 = assemble("MOV r1, 10\nADD r2, r1, r1\nST r2, [0]\nHALT\n")
    simt2 = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt2.load_program(prog2)
    simt2.run()
    check(simt2.memory.read_word(0) == 20, "basic ALU: 10+10=20")

    # Test predication still works
    prog3 = assemble("TID r1\nMOV r2, 2\nDIV r3, r1, r2\nMUL r4, r3, r2\nSUB r5, r1, r4\n"
                     "SETP.EQ r5, r0\n@p0 MOV r6, 100\n@p0 ST r6, [0]\nHALT\n")
    simt3 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt3.load_program(prog3)
    simt3.run()
    check(simt3.memory.read_word(0) == 100, "pred: mem[0]=100 (tid0 even)")
    check(simt3.memory.read_word(1) == 0, "pred: mem[1]=0 (tid1 odd skipped)")

    # Test warp regs still work
    prog4 = assemble("WREAD r1, wid\nWREAD r2, ntid\nST r1, [0]\nST r2, [1]\nHALT\n")
    simt4 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt4.load_program(prog4)
    simt4.run()
    check(simt4.memory.read_word(0) == 0, "wreg: wid=0")
    check(simt4.memory.read_word(1) == 4, "wreg: ntid=4")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 12: Warp Communication — Test Suite")
    print("=" * 60)
    test_isa_warp_comm()
    test_assembler_warp_comm()
    test_shfl_program()
    test_vote_program()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
