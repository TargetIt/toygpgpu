; Test 03: Vector Subtract and Divide
MOV r1, 100
ST r1, [0]
MOV r1, 80
ST r1, [1]
MOV r1, 60
ST r1, [2]
MOV r1, 40
ST r1, [3]
MOV r1, 20
ST r1, [4]
MOV r1, 10
ST r1, [5]
MOV r1, 8
ST r1, [6]
MOV r1, 4
ST r1, [7]

VMOV v2, 5
VLD v1, [0]
VSUB v3, v1, v2
VST v3, [10]

VMOV v4, 2
VLD v5, [10]
VDIV v6, v5, v4
VST v6, [20]
HALT
