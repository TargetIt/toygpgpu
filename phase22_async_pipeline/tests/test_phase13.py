#!/usr/bin/env python3
"""Phase 13 Test Suite — Tiling Strategies + Backward Compat"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simt_core import SIMTCore
from assembler import assemble
from isa import (OPCODE_NAMES, decode, OP_TLCONF, OP_TLDS, OP_TLSTS)

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def run_prog(f, checks, warp_size=1, num_warps=1, preload=None):
    """Run an asm program and verify results. preload: {addr: val} for setup."""
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', f)
    with open(path, encoding='utf-8') as fp:
        prog = assemble(fp.read())
    simt = SIMTCore(warp_size=warp_size, num_warps=num_warps, memory_size=1024)
    if preload:
        for addr, val in preload.items():
            simt.memory.write_word(addr, val)
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

def test_isa_tiling():
    print("\n--- Tiling ISA Tests ---")
    check(OP_TLCONF == 0x35, "OP_TLCONF = 0x35")
    check(OP_TLDS == 0x36, "OP_TLDS = 0x36")
    check(OP_TLSTS == 0x37, "OP_TLSTS = 0x37")
    check(OPCODE_NAMES[OP_TLCONF] == "TLCONF", "TLCONF in OPCODE_NAMES")
    check(OPCODE_NAMES[OP_TLDS] == "TLDS", "TLDS in OPCODE_NAMES")
    check(OPCODE_NAMES[OP_TLSTS] == "TLSTS", "TLSTS in OPCODE_NAMES")


def test_assembler_tiling():
    print("\n--- Assembler Tiling Tests ---")

    # TLCONF
    src = "TLCONF 8, 8, 4\nHALT\n"
    prog = assemble(src)
    instr = decode(prog[0])
    check(instr.opcode == OP_TLCONF, "TLCONF encode/decode")
    check(instr.rd == 8, "TLCONF tile_M=8")
    check(instr.rs1 == 8, "TLCONF tile_N=8")
    check(instr.imm == 4, "TLCONF tile_K=4")

    # TLDS
    src2 = "TLDS 0, 100\nHALT\n"
    prog2 = assemble(src2)
    instr2 = decode(prog2[0])
    check(instr2.opcode == OP_TLDS, "TLDS encode/decode")
    check(instr2.rd == 0, "TLDS smem_off=0")
    check(instr2.rs1 == 100, "TLDS glob_base=100")

    # TLSTS
    src3 = "TLSTS 8, 200\nHALT\n"
    prog3 = assemble(src3)
    instr3 = decode(prog3[0])
    check(instr3.opcode == OP_TLSTS, "TLSTS encode/decode")
    check(instr3.rd == 8, "TLSTS smem_off=8")
    check(instr3.rs1 == 200, "TLSTS glob_base=200")


# ============================================================
# Tiled Matmul Test
# ============================================================

def test_tiled_matmul():
    print("\n--- Tiled Matmul Test ---")
    # Pre-load A[2x2]=[1,2;3,4] at mem[0..3], B[2x2]=[5,6;7,8] at mem[8..11]
    preload = {
        0: 1, 1: 2, 2: 3, 3: 4,       # A row-major
        8: 5, 9: 6, 10: 7, 11: 8,      # B row-major
    }
    run_prog("11_tiled_matmul.asm", {
        16: 19,  # C[0][0] = 1*5 + 2*7
        17: 22,  # C[0][1] = 1*6 + 2*8
        18: 43,  # C[1][0] = 3*5 + 4*7
        19: 50,  # C[1][1] = 3*6 + 4*8
    }, warp_size=1, preload=preload)


# ============================================================
# Double Buffer Test
# ============================================================

def test_double_buffer():
    print("\n--- Double Buffer Test ---")
    # Tile 0 data at mem[0..7], Tile 1 data at mem[8..15]
    preload = {}
    for i in range(16):
        preload[i] = i  # mem[i] = i
    run_prog("12_tile_double_buffer.asm", {
        100: sum(range(0, 8)),   # 0+1+...+7 = 28
        101: sum(range(8, 16)),  # 8+9+...+15 = 92
    }, warp_size=1, preload=preload)


# ============================================================
# SIMTCore Tile Config State Test
# ============================================================

def test_tile_config_state():
    print("\n--- Tile Config State Test ---")
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    check(simt.tile_m == 8, "default tile_m=8")
    check(simt.tile_n == 8, "default tile_n=8")
    check(simt.tile_k == 8, "default tile_k=8")

    prog = assemble("TLCONF 4, 4, 2\nHALT\n")
    simt.load_program(prog)
    simt.run()
    check(simt.tile_m == 4, "TLCONF sets tile_m=4")
    check(simt.tile_n == 4, "TLCONF sets tile_n=4")
    check(simt.tile_k == 2, "TLCONF sets tile_k=2")


# ============================================================
# Backward Compat: Phase 12 features still work
# ============================================================

def test_backward_compat():
    print("\n--- Backward Compat Tests ---")

    # SHFL still works
    prog = assemble("TID r1\nMOV r2, 10\nMUL r3, r1, r2\n"
                    "SHFL r4, r3, 0, 0\nST r4, [0]\nHALT\n")
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt.load_program(prog)
    simt.run()
    check(simt.memory.read_word(0) == 0, "SHFL IDX: all threads read thread0")

    # VOTE still works
    prog2 = assemble("MOV r1, 1\nVOTE.ALL r2, r1\nST r2, [0]\nHALT\n")
    simt2 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt2.load_program(prog2)
    simt2.run()
    check(simt2.memory.read_word(0) == 1, "VOTE.ALL all-1 = 1")

    # PRED still works
    prog3 = assemble("TID r1\nMOV r2, 2\nDIV r3, r1, r2\nMUL r4, r3, r2\n"
                     "SUB r5, r1, r4\nSETP.EQ r5, r0\n"
                     "@p0 MOV r6, 99\n@p0 ST r6, [0]\nHALT\n")
    simt3 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt3.load_program(prog3)
    simt3.run()
    check(simt3.memory.read_word(0) == 99, "PRED: mem[0]=99 (tid0 even)")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 13: Tiling Strategies — Test Suite")
    print("=" * 60)
    test_isa_tiling()
    test_assembler_tiling()
    test_tile_config_state()
    test_tiled_matmul()
    test_double_buffer()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
