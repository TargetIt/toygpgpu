; Test 04: Multi-thread Vector Addition
; C[i] = A[i] + B[i], where thread i processes pair i
;
; A[i] is pre-stored at mem[0..7]  = [10,20,30,40,50,60,70,80]
; B[i] is pre-stored at mem[8..15]  = [1,2,3,4,5,6,7,8]
; C[i] goes to mem[16..23]

; Store A
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

; Store B
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

; Now each thread does C[tid] = A[tid] + B[tid]
TID r1
LD r2, [0]       ; loads mem[0+tid] = A[tid]
LD r3, [8]       ; loads mem[8+tid] = B[tid]
ADD r4, r2, r3   ; r4 = A[tid] + B[tid]
ST r4, [16]      ; mem[16+tid] = C[tid]
HALT
