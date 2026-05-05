; Test 02: Vector Multiply
; Scale: C[i] = A[i] * 3, i=0..7
MOV r1, 1
ST r1, [0]
MOV r1, 2
ST r1, [1]
MOV r1, 3
ST r1, [2]
MOV r1, 4
ST r1, [3]
MOV r1, 5
ST r1, [4]
MOV r1, 6
ST r1, [5]
MOV r1, 7
ST r1, [6]
MOV r1, 8
ST r1, [7]

VMOV v2, 3
VLD v1, [0]
VMUL v3, v1, v2
VST v3, [10]
HALT
