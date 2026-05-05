; Test 04: Conditional divergence — tid < 4
; tid 0-3 → mem[100+tid]=1, others → mem[100+tid]=0
; All → mem[200+tid]=tid

TID r1
MOV r2, 4
DIV r4, r1, r2     ; r4 = tid/4 (0 if tid<4)
MOV r5, 0
BEQ r4, r5, low_tid

; High tid (4-7)
MOV r6, 0
ST r6, [100]
JMP after

low_tid:
; Low tid (0-3)
MOV r6, 1
ST r6, [100]
JMP after

after:
TID r7
ST r7, [200]
HALT
