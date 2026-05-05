; Test 01: I-Buffer Pipeline — basic scalar program through new pipeline
MOV r1, 10
MOV r2, 3
ADD r3, r1, r2
ST r3, [0]
HALT
