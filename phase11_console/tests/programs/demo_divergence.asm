; Demo: Branch Divergence — ideal for learning console
TID r1
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2
MOV r5, 0
BEQ r4, r1, even_path
MOV r6, 1
ST r6, [100]
JMP done
even_path:
MOV r6, 2
ST r6, [100]
JMP done
done:
TID r7
ADD r7, r7, r7
ST r7, [200]
HALT
