#!/usr/bin/env python3
"""Phase 14 Test Suite — CuTile Programming Model"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simt_core import SIMTCore
from assembler import assemble
from isa import (OPCODE_NAMES, decode, OP_WGMMA, OP_TLCONF)
from cutile_parser import parse_cutile, generate_asm, assemble_cutile

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


# ============================================================
# CuTile Parser Unit Tests
# ============================================================

def test_cutile_parser():
    print("\n--- CuTile Parser Tests ---")

    src = """
    tile M=4, N=4, K=2
    kernel test(A:[M,K], B:[K,N], C:[M,N]) {
        load A[0:M, 0:K] -> smem[0]
        load B[0:K, 0:N] -> smem[16]
        mma smem[0], smem[16] -> smem[32]
        store smem[32] -> C[0:M, 0:N]
    }
    """
    kernel = parse_cutile(src)
    check(kernel.name == "test", "kernel name parsed")
    check(kernel.tile.M == 4, "tile M=4")
    check(kernel.tile.N == 4, "tile N=4")
    check(kernel.tile.K == 2, "tile K=2")
    check(len(kernel.ops) == 4, "4 operations parsed")
    check(kernel.ops[0]['op'] == 'load', "op[0] = load A")
    check(kernel.ops[1]['op'] == 'load', "op[1] = load B")
    check(kernel.ops[2]['op'] == 'mma', "op[2] = mma")
    check(kernel.ops[3]['op'] == 'store', "op[3] = store C")


def test_cutile_codegen():
    print("\n--- CuTile Code Generation Tests ---")

    # Generate assembly from a simple kernel
    src = """
    tile M=2, N=2, K=1
    kernel simple(A:[M,K], B:[K,N], C:[M,N]) {
        load A[0:M, 0:K] -> smem[0]
        load B[0:K, 0:N] -> smem[2]
        mma smem[0], smem[2] -> smem[4]
        store smem[4] -> C[0:M, 0:N]
    }
    """
    kernel = parse_cutile(src)
    asm = generate_asm(kernel, {'A': {'base': 0}, 'B': {'base': 8}, 'C': {'base': 16}})

    check('TLCONF 2, 2, 1' in asm, "generates TLCONF")
    check('TLDS 0, 0' in asm, "generates TLDS for A")
    check('TLDS 2, 8' in asm, "generates TLDS for B")
    check('SHLD' in asm, "generates SHLD for mma")
    check('MUL' in asm, "generates MUL for multiply")
    check('ADD' in asm, "generates ADD for accumulate")
    check('HALT' in asm, "generates HALT")


# ============================================================
# CuTile End-to-End Test
# ============================================================

def test_cutile_e2e():
    print("\n--- CuTile End-to-End Test ---")

    src = """
    tile M=2, N=2, K=2
    kernel e2e_test(A:[M,K], B:[K,N], C:[M,N]) {
        load A[0:M, 0:K] -> smem[0]
        load B[0:K, 0:N] -> smem[4]
        mma smem[0], smem[4] -> smem[8]
        store smem[8] -> C[0:M, 0:N]
    }
    """
    matrix_data = {
        'A': {'base': 0, 'M': 2, 'N': 2},
        'B': {'base': 8, 'M': 2, 'N': 2},
        'C': {'base': 16, 'M': 2, 'N': 2},
    }

    # Assemble from CuTile DSL
    code, asm_text = assemble_cutile(src, matrix_data)
    check(len(code) > 5, f"generated {len(code)} instructions")

    # Pre-load matrices: A=[1,2;3,4], B=[5,6;7,8]
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.memory.write_word(0, 1); simt.memory.write_word(1, 2)
    simt.memory.write_word(2, 3); simt.memory.write_word(3, 4)
    simt.memory.write_word(8, 5); simt.memory.write_word(9, 6)
    simt.memory.write_word(10, 7); simt.memory.write_word(11, 8)

    simt.load_program(code)
    simt.run()

    # Verify C = A × B
    expected = {16: 19, 17: 22, 18: 43, 19: 50}
    for addr, exp in expected.items():
        actual = simt.memory.read_word(addr)
        check(actual == exp, f"C[{addr-16}] = {exp} (got {actual})")


# ============================================================
# WGMMA Instruction Test
# ============================================================

def test_wgmma():
    print("\n--- WGMMA Instruction Test ---")

    # Pre-load tiles into shared memory, then WGMMA
    prog = assemble(
        "TLCONF 2, 2, 2\n"
        "TLDS 0, 0\n"     # A tile: smem[0..3] = mem[0..3] = [1,2,3,4]
        "TLDS 4, 8\n"     # B tile: smem[4..7] = mem[8..11] = [5,6,7,8]
        "WGMMA 0, 4, 8\n" # smem[8..11] += A[0..3] × B[4..7]
        "TLSTS 8, 16\n"   # Store result to mem[16..19]
        "HALT\n"
    )
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.memory.write_word(0, 1); simt.memory.write_word(1, 2)
    simt.memory.write_word(2, 3); simt.memory.write_word(3, 4)
    simt.memory.write_word(8, 5); simt.memory.write_word(9, 6)
    simt.memory.write_word(10, 7); simt.memory.write_word(11, 8)
    simt.load_program(prog)
    simt.run()

    check(simt.memory.read_word(16) == 19, "WGMMA: C[0][0] = 19")
    check(simt.memory.read_word(17) == 22, "WGMMA: C[0][1] = 22")
    check(simt.memory.read_word(18) == 43, "WGMMA: C[1][0] = 43")
    check(simt.memory.read_word(19) == 50, "WGMMA: C[1][1] = 50")


# ============================================================
# ISA Tests
# ============================================================

def test_isa_wgmma():
    print("\n--- WGMMA ISA Tests ---")
    check(OP_WGMMA == 0x38, "OP_WGMMA = 0x38")
    check(OPCODE_NAMES[OP_WGMMA] == "WGMMA", "WGMMA in OPCODE_NAMES")

    prog = assemble("WGMMA 0, 4, 8\nHALT\n")
    instr = decode(prog[0])
    check(instr.opcode == OP_WGMMA, "WGMMA encode/decode")
    check(instr.rd == 0, "WGMMA smem_a=0")
    check(instr.rs1 == 4, "WGMMA smem_b=4")
    check(instr.rs2 == 8, "WGMMA smem_c=8")


# ============================================================
# Backward Compat
# ============================================================

def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    # Phase 13 features (TLCONF, TLDS, TLSTS)
    prog = assemble("TLCONF 2,2,2\nTLDS 0,0\nTLDS 4,8\n"
                    "SHLD r1,0\nST r1,[100]\nHALT\n")
    simt = SIMTCore(warp_size=1, num_warps=1, memory_size=256)
    simt.memory.write_word(0, 99)
    simt.load_program(prog)
    simt.run()
    check(simt.memory.read_word(100) == 99, "Phase 13 TLDS+SHLD still works")

    # Phase 12 features (SHFL)
    prog2 = assemble("TID r1\nSHFL r2,r1,0,0\nST r2,[0]\nHALT\n")
    simt2 = SIMTCore(warp_size=4, num_warps=1, memory_size=256)
    simt2.load_program(prog2)
    simt2.run()
    check(simt2.memory.read_word(0) == 0, "Phase 12 SHFL still works")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 14: CuTile Programming Model — Test Suite")
    print("=" * 60)
    test_isa_wgmma()
    test_cutile_parser()
    test_cutile_codegen()
    test_wgmma()
    test_cutile_e2e()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
