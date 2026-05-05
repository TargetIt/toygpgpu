; Test 05: Phase 0/1/2 backward compatibility
; Scalar instructions still work (no branch involved)
MOV r1, 10
MOV r2, 3
ADD r3, r1, r2
ST r3, [0]         ; mem[0] = 13

TID r4
ST r4, [10]        ; mem[10+tid] = tid

HALT
