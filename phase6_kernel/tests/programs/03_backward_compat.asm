; Test 03: Full backward compatibility
TID r1
MOV r2, 10
MUL r3, r1, r2
ST r3, [0]
MOV r2, 2
DIV r4, r1, r2
MUL r4, r4, r2
BEQ r4, r1, even
MOV r5, 1
ST r5, [100]
JMP done
even:
MOV r5, 2
ST r5, [100]
done:
SHLD r6, [0]
ST r6, [200]
HALT
