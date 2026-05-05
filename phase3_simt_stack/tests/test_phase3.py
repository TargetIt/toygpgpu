#!/usr/bin/env python3
"""Phase 3 测试套件 — SIMT Stack + Branch"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isa import *
from simt_stack import SIMTStack, SIMTStackEntry
from warp import Warp
from simt_core import SIMTCore, popcount
from assembler import assemble

passed = 0
failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")

def run_prog(f, checks=None):
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path) as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=8, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    if checks:
        for addr, expected in checks.items():
            actual = simt.memory.read_word(addr)
            if actual == (expected & 0xFFFFFFFF):
                passed += 1; print(f"  ✅ {f}: mem[{addr}] = {expected}")
            else:
                failed += 1; print(f"  ❌ {f}: mem[{addr}] = {actual}, expected {expected}")

def run_prog_fn(f, check_fn):
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path) as fp: prog = assemble(fp.read())
    simt = SIMTCore(warp_size=8, num_warps=1, memory_size=1024)
    simt.load_program(prog)
    simt.run()
    ok, msg = check_fn(simt)
    if ok: passed += 1; print(f"  ✅ {f}: {msg}")
    else: failed += 1; print(f"  ❌ {f}: {msg}")

# ============================================================
def test_simt_stack():
    print("\n--- SIMT Stack Tests ---")
    s = SIMTStack()
    check(s.empty, "empty initially")
    s.push(SIMTStackEntry(reconv_pc=10, orig_mask=0xFF, taken_mask=0x0F, fallthrough_pc=5))
    check(len(s) == 1, "push adds entry")
    check(s.at_reconvergence(10), "at_reconvergence(10)")
    check(not s.at_reconvergence(5), "not at_reconvergence(5)")
    e = s.pop()
    check(e.reconv_pc == 10, "pop correct reconv_pc")
    check(s.empty, "empty after pop")

def test_isa_branch():
    print("\n--- ISA Branch Tests ---")
    for opcode, name in [(OP_JMP, "JMP"), (OP_BEQ, "BEQ"), (OP_BNE, "BNE")]:
        w = encode_rtype(opcode, 0, 1, 2, 42)
        instr = decode(w)
        check(instr.opcode == opcode, f"{name} encode/decode")
        check(instr.imm == 42, f"{name} offset")

def test_assembler():
    print("\n--- Assembler Branch Tests ---")
    src = """
    MOV r1, 1
    JMP target
    MOV r2, 2
    target:
    HALT
    """
    prog = assemble(src)
    check(len(prog) == 4, "4 instructions (label excluded, MOV not skipped)")
    check(decode(prog[0]).opcode == OP_MOV, "MOV")
    check(decode(prog[1]).opcode == OP_JMP, "JMP")
    check(decode(prog[3]).opcode == OP_HALT, "HALT at target")

    # Test BEQ
    src2 = """
    BEQ r1, r2, dest
    MOV r1, 0
    dest:
    HALT
    """
    prog2 = assemble(src2)
    check(decode(prog2[0]).opcode == OP_BEQ, "BEQ assembled")
    check(decode(prog2[0]).imm != 0, "BEQ offset non-zero")

# ============================================================
def test_programs():
    print("\n--- Assembly Program Tests ---")

    run_prog("01_jmp.asm", {0: 10, 2: 20})

    run_prog("02_beq_bne.asm", {1: 42})

    # Test 03: Divergence — even/odd threads
    def check_divergence(simt):
        for tid in range(8):
            val = simt.memory.read_word(200 + tid)
            expected = 2 if tid % 2 == 0 else 1
            if val != expected:
                return False, f"mem[{200+tid}] = {val}, expected {expected}"
            val2 = simt.memory.read_word(300 + tid)
            if val2 != tid:
                return False, f"mem[{300+tid}] = {val2}, expected {tid}"
        return True, "even→2, odd→1, all→tid"
    run_prog_fn("03_divergence.asm", check_divergence)

    # Test 04: Simple conditional
    def check_nested(simt):
        for tid in range(8):
            val = simt.memory.read_word(100 + tid)
            expected = 1 if tid < 4 else 0
            if val != expected:
                return False, f"mem[{100+tid}] = {val}, expected {expected}"
            val2 = simt.memory.read_word(200 + tid)
            if val2 != tid:
                return False, f"mem[{200+tid}] = {val2}, expected {tid}"
        return True, "low→1, high→0, all→tid"
    run_prog_fn("04_nested.asm", check_nested)

    run_prog("05_backward_compat.asm", {0: 13, 10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7})


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 3: SIMT Stack + Branch — Test Suite")
    print("=" * 60)
    test_simt_stack()
    test_isa_branch()
    test_assembler()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
