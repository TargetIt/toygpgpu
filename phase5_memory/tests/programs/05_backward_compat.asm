; Test 05: Phase 0-4 backward compatibility
MOV r1, 10
MOV r2, 3
ADD r3, r1, r2
ST r3, [0]
TID r4
ST r4, [10]
HALT
