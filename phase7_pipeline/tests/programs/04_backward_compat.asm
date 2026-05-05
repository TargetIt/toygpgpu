; Test 04: Phase 0-6 backward compatibility
MOV r1, 10
ADD r2, r1, r1
ST r2, [0]
TID r3
ST r3, [10]
HALT
