; Test 01: RAW Hazard — consecutive write+read same register
; Without scoreboard, r2 would be read before write completes
MOV r1, 10
MOV r2, 5
ADD r3, r1, r2   ; r3 = 15 (reads r1,r2 → OK after MOV latency)
ST r3, [0]
HALT
