; Test 03: SIMT Divergence — if/else with explicit merge
; Even threads: mem[200+tid] = 2
; Odd threads:  mem[200+tid] = 1
; All:         mem[300+tid] = tid

TID r1
MOV r2, 2
DIV r3, r1, r2
MUL r4, r3, r2

BEQ r4, r1, even_path

; Odd path
MOV r6, 1
ST r6, [200]
JMP reconv

even_path:
; Even path
MOV r6, 2
ST r6, [200]
JMP reconv

reconv:
TID r7
ST r7, [300]
HALT
