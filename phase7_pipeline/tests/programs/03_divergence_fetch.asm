; Test 03: Branch + I-Buffer flush
TID r1
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2
BEQ r4, r1, even_path
MOV r6, 1
ST r6, [100]
JMP reconv
even_path:
MOV r6, 2
ST r6, [100]
JMP reconv
reconv:
TID r7
ST r7, [200]
HALT
