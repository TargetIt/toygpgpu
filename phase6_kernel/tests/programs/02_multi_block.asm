; Test 02: Multi-block kernel — vector add per block
TID r1
MOV r2, 2
MUL r3, r1, r2      ; r3 = tid * 2
ST r3, [50]          ; mem[50+tid] = tid*2
HALT
