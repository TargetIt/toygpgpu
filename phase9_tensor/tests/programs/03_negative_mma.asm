; Test 03: MMA with memory pre-load
; mem[0] = [a1=7, a0=1], mem[1] = [b1=2, b0=3]
; r4 = 1*3 + 7*2 + 10 = 27
LD r1, [0]
LD r2, [1]
MOV r3, 10
MMA r4, r1, r2, r3
ST r4, [10]
HALT
