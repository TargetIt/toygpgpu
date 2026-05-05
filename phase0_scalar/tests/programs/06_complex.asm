; Test 06: Complex Expression
; Compute: (10 + 5) * (20 - 8) / 4
;        = 15 * 12 / 4
;        = 180 / 4
;        = 45
MOV r1, 10
MOV r2, 5
ADD r3, r1, r2   ; r3 = 15

MOV r4, 20
MOV r5, 8
SUB r6, r4, r5   ; r6 = 12

MUL r7, r3, r6   ; r7 = 180

MOV r8, 4
DIV r9, r7, r8   ; r9 = 45
ST r9, [0]       ; mem[0] = 45
HALT
