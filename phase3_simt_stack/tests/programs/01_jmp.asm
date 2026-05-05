; Test 01: Unconditional Jump
; Skip over some instructions
MOV r1, 10
ST r1, [0]        ; mem[0] = 10
JMP skip
MOV r1, 99        ; skipped
ST r1, [1]        ; skipped
skip:
MOV r1, 20
ST r1, [2]        ; mem[2] = 20
HALT
