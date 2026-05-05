; Test 02: Multiply and Divide
; Compute 6 * 7 = 42, then 42 / 6 = 7
MOV r1, 6
MOV r2, 7
MUL r3, r1, r2   ; r3 = 6 * 7 = 42
ST r3, [0]       ; mem[0] = 42

MOV r4, 6
DIV r5, r3, r4   ; r5 = 42 / 6 = 7
ST r5, [1]       ; mem[1] = 7
HALT
