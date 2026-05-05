; Test 02: Conditional Branch
; BEQ and BNE with register comparison
MOV r1, 5
MOV r2, 5
MOV r3, 10
BEQ r1, r2, equal   ; r1==r2 → jump to equal
ST r3, [0]          ; NOT executed (r1==r2)
HALT
equal:
MOV r4, 42
ST r4, [1]          ; mem[1] = 42
HALT
