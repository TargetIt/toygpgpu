; Test 02: 2x2 Matrix Multiply D = A×B + C
; mem[0..3] pre-loaded: A_row0, A_row1, B_col0, B_col1
LD r1, [0]       ; A_row0 = [a01=3, a00=2]
LD r2, [1]       ; A_row1 = [a11=4, a10=1]
LD r3, [2]       ; B_col0 = [b10=6, b00=4]
LD r4, [3]       ; B_col1 = [b11=7, b01=5]
MOV r5, 1         ; C = 1

MMA r6, r1, r3, r5   ; d00 = 2*4 + 3*6 + 1 = 27
ST r6, [10]
MMA r7, r1, r4, r5   ; d01 = 2*5 + 3*7 + 1 = 32
ST r7, [11]
MMA r8, r2, r3, r5   ; d10 = 1*4 + 4*6 + 1 = 29
ST r8, [12]
MMA r9, r2, r4, r5   ; d11 = 1*5 + 4*7 + 1 = 34
ST r9, [13]
HALT
