; Test 05: Phase 0 Backward Compatibility
; Verify that all Phase 0 scalar instructions still work
MOV r1, 5
MOV r2, 3
ADD r3, r1, r2   ; r3 = 8
ST r3, [0]       ; mem[0] = 8

MOV r4, 6
MOV r5, 7
MUL r6, r4, r5   ; r6 = 42
ST r6, [1]       ; mem[1] = 42

LD r7, [0]       ; r7 = mem[0] = 8
LD r8, [1]       ; r8 = mem[1] = 42
ADD r9, r7, r8   ; r9 = 50
ST r9, [2]       ; mem[2] = 50

; r0 protection
ADD r0, r9, r9   ; should be ignored
ADD r10, r0, r9  ; r10 = 0 + 50 = 50
ST r10, [3]      ; mem[3] = 50

HALT
