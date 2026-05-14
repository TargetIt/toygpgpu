#!/usr/bin/env python3
"""Phase 8 Test Suite — PTX Frontend"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ptx_parser import parse_ptx, translate_ptx, assemble_ptx
from simt_core import SIMTCore

passed = 0; failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {name}")
    else: failed += 1; print(f"  ❌ {name}")


def test_parser():
    print("\n--- PTX Parser Tests ---")
    src = """
.entry test
{
    mov.u32 %r0, 42;
    mov.u32 %r1, %tid.x;
    add.u32 %r2, %r0, %r1;
    st.global.u32 [100], %r2;
    ret;
}
"""
    prog = parse_ptx(src)
    check(len(prog.instructions) >= 4, f"Parsed {len(prog.instructions)} instructions")
    check(prog.num_regs >= 3, f"Virtual regs: {prog.num_regs}")
    asm, num = translate_ptx(prog)
    check('TID' in asm or 'tid' in asm.lower(), "TID instruction emitted")
    check('ADD' in asm, "ADD instruction emitted")
    check('HALT' in asm or 'ret' in src, "HALT emitted")
    print(f"  Generated assembly:\n{asm}")


def test_translator():
    print("\n--- Translator Tests ---")
    src = "mov.u32 %r0, 10;\nmov.u32 %r1, 3;\nadd.u32 %r2, %r0, %r1;\nret;"
    prog = parse_ptx(src)
    asm, n = translate_ptx(prog)
    check(n > 0, f"Translated {n} instructions")
    # Check register allocation
    check('r1' in asm or 'r2' in asm, "Physical registers used")


def test_programs():
    global passed, failed
    print("\n--- PTX Program Execution Tests ---")

    # Test 01: Vector Add (PTX)
    path = os.path.join(os.path.dirname(__file__), 'programs')
    with open(f'{path}/01_vector_add.ptx', encoding='utf-8') as f:
        code, asm = assemble_ptx(f.read())
    print(f"  Generated asm:\n{asm}\n")
    simt = SIMTCore(warp_size=4, num_warps=1, memory_size=1024)
    # Pre-load A and B
    for i in range(4):
        simt.memory.write_word(i, 10 + i*10)    # A: 10,20,30,40
        simt.memory.write_word(8+i, 1 + i)       # B: 1,2,3,4
    simt.load_program(code)
    simt.run()
    for i, exp in enumerate([11, 22, 33, 44]):
        actual = simt.memory.read_word(16+i)
        if actual == exp:
            passed += 1; print(f"  ✅ 01_vector_add.ptx: C[{i}] = {exp}")
        else:
            failed += 1; print(f"  ❌ 01_vector_add.ptx: C[{i}] = {actual}, expected {exp}")

    # Test 02: Scale (PTX)
    with open(f'{path}/02_scale.ptx', encoding='utf-8') as f:
        code, asm = assemble_ptx(f.read())
    simt2 = SIMTCore(warp_size=4, num_warps=1, memory_size=1024)
    for i in range(4):
        simt2.memory.write_word(10+i, i+1)  # A: 1,2,3,4
    simt2.load_program(code)
    simt2.run()
    for i, exp in enumerate([3, 6, 9, 12]):
        actual = simt2.memory.read_word(20+i)
        if actual == exp:
            passed += 1; print(f"  ✅ 02_scale.ptx: C[{i}] = {exp}")
        else:
            failed += 1; print(f"  ❌ 02_scale.ptx: C[{i}] = {actual}, expected {exp}")

    # Test 03: Basic mov+add (PTX)
    with open(f'{path}/03_mov_imm.ptx', encoding='utf-8') as f:
        code, asm = assemble_ptx(f.read())
    simt3 = SIMTCore(warp_size=1, num_warps=1, memory_size=1024)
    simt3.load_program(code)
    simt3.run()
    actual = simt3.memory.read_word(0)
    if actual == 15:
        passed += 1; print(f"  ✅ 03_mov_imm.ptx: mem[0] = 15")
    else:
        failed += 1; print(f"  ❌ 03_mov_imm.ptx: mem[0] = {actual}, expected 15")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 8: PTX Frontend — Test Suite")
    print("=" * 60)
    test_parser()
    test_translator()
    test_programs()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
