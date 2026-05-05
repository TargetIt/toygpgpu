; Test 02: Thread-independent registers
; Each thread computes: tid * 10, stores to mem[200+tid]
TID r1
MOV r2, 10
MUL r3, r1, r2
ST r3, [200]
HALT
