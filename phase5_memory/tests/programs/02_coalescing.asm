; Test 02: Memory Coalescing
; All threads access consecutive addresses → should coalesce
TID r1
MOV r2, 10
MUL r3, r1, r2      ; r3 = tid*10
ST r3, [50]          ; mem[50+tid] = tid*10 — contiguous access!
LD r4, [50]          ; read back
ST r4, [200]         ; verify
HALT
