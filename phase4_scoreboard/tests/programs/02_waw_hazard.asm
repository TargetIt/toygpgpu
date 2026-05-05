; Test 02: WAW Hazard — consecutive writes same register
MOV r1, 10
MOV r1, 20       ; WAW: r1 has pending write from previous MOV
MOV r2, 30
ADD r3, r1, r2   ; r3 = 20+30 = 50 (should use second MOV value)
ST r3, [0]
HALT
