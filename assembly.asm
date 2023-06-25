
;vp := vector packed blah blahb alh
; Remember mutability: zmm5 corresponds to the AVX512 register.
vpsrlw zmm5, zmmword ptr [rdi + rax], 7 ; 

; this roughly corresponds to the mask applied to input1
vpandq zmm5, zmm5, zmm3 ;apply a packed mask to zmm5, or similar, to get the lead bit of the packed bytes
vpaddb zmm4, zmm5, zmm4 ; add zmm5 to zmm4, which is probably the counter/runner

; this roughly corresponds to the mask applied to input2
vpsrlw zmm5, zmmword ptr [rdi + rax + 64], 7 ;  and shift right 7 bits again
vpandq zmm5, zmm5, zmm3 ; get the lead bit for packed byte again

; this roughly corresponds to the mask applied to input3
vpsrlw zmm6, zmmword ptr [rdi + rax + 128], 7 ; 
vpandq zmm6, zmm6, zmm3 ;get all lead bytes
vpaddb zmm5, zmm5, zmm6 ;add those leadbytes #3 to the leadbytes of input #2

; this roughly corresponds to the mask applied to input4
vpsrlw zmm6, zmmword ptr [rdi + rax + 192], 7
vpaddb zmm4, zmm4, zmm5 ; now add the leader bytes of put #2 and #3 to the runner
vpandq zmm5, zmm6, zmm3 ; mask input4 
vpaddb zmm4, zmm4, zmm5 ; and add the remaining lead bytes to the runner

;remember: rdx,rax,rcx are general purpose 64-bit registers
;rcx is generally the counter. The other two are typically generla purpose
; in this context, rax is the length...
lea rdx, [rax + 256] ; computer the address that is 256 bits after rax, and then store it in rdx
add rax, 512 ; add 512 to rax ...
cmp rax, rcx ;compare rax and rcx and adjust flags accordingly, jump based on this
mov rax, rdx ;rax =rdx
jbe .LBB0_4 ; jump if rax <= rcx , probably redo the loop if you can still read 512 bits?