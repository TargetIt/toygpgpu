; Test 03: Cache Basic Operation
; Repeated access to same address → cache hits
MOV r1, 42
ST r1, [0]           ; write to mem[0]
LD r2, [0]           ; first read → miss
LD r3, [0]           ; second read → hit!
ADD r4, r2, r3       ; r4 = 42+42 = 84
ST r4, [10]          ; mem[10] = 84
HALT
