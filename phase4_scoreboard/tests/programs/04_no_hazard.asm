; Test 04: No Hazard — independent registers
MOV r1, 10
MOV r2, 20
MOV r3, 30
ADD r4, r1, r2    ; r4=30 (r1,r2 independent, no hazard)
ADD r5, r3, r4    ; r5=60
ST r5, [0]        ; mem[0]=60
HALT
