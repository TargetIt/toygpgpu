; Test 03: Branch + I-Buffer interaction
MOV r1, 10
MOV r2, 5
BEQ r1, r2, skip    ; 10 != 5, no branch
MOV r3, 1
ST r3, [0]          ; mem[0] = 1 (r1 != r2 fallthrough)
JMP done
skip:
MOV r3, 2           ; NOT executed
ST r3, [1]
done:
MOV r4, 42
ST r4, [10]         ; mem[10] = 42
HALT
