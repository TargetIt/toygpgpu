; Test 05: Backward compatibility — Phase 0-3 programs
TID r1
MOV r2, 2
MUL r3, r1, r2
ST r3, [100]
ADD r3, r3, r2
ST r3, [200]
HALT
