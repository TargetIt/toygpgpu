; Test 01: Basic Arithmetic
; Compute 5 + 3 = 8, store result to mem[0]
MOV r1, 5        ; r1 = 5
MOV r2, 3        ; r2 = 3
ADD r3, r1, r2   ; r3 = 8
ST r3, [0]       ; mem[0] = 8
HALT
