; Test 01: MMA Dot Product (memory-packed operands)
; mem[0] = [a1=3, a0=2], mem[1] = [b1=5, b0=4]
; r4 = 2*4 + 3*5 + 10 = 33
LD r1, [0]       ; r1 = packed [3, 2]
LD r2, [1]       ; r2 = packed [5, 4]
MOV r3, 10
MMA r4, r1, r2, r3
ST r4, [10]
HALT
