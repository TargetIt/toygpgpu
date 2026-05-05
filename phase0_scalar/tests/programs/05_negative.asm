; Test 05: Negative Numbers
; Test signed arithmetic with negative values
MOV r1, -10      ; r1 = -10 (0xFFFFFFF6)
MOV r2, 3
MUL r3, r1, r2   ; r3 = -30
ST r3, [0]       ; mem[0] = -30 (0xFFFFFFE2)

MOV r4, -10
MOV r5, -5
ADD r6, r4, r5   ; r6 = -15
ST r6, [1]       ; mem[1] = -15

MOV r7, -20
MOV r8, 4
DIV r9, r7, r8   ; r9 = -5
ST r9, [2]       ; mem[2] = -5
HALT
