; Test 01: Vector Addition
; C[i] = A[i] + B[i], i=0..7
MOV r1, 10
ST r1, [0]
MOV r1, 20
ST r1, [1]
MOV r1, 30
ST r1, [2]
MOV r1, 40
ST r1, [3]
MOV r1, 50
ST r1, [4]
MOV r1, 60
ST r1, [5]
MOV r1, 70
ST r1, [6]
MOV r1, 80
ST r1, [7]

MOV r1, 1
ST r1, [8]
MOV r1, 2
ST r1, [9]
MOV r1, 3
ST r1, [10]
MOV r1, 4
ST r1, [11]
MOV r1, 5
ST r1, [12]
MOV r1, 6
ST r1, [13]
MOV r1, 7
ST r1, [14]
MOV r1, 8
ST r1, [15]

VLD v1, [0]
VLD v2, [8]
VADD v3, v1, v2
VST v3, [16]
HALT
