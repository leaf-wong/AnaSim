// crt0.S
.section .text
.globl _start
_start:
    // Set up stack
    la sp, __stack_start

    // Optional BSS init
    la a0, __bss_start
    la a1, __bss_end

1:
    bgeu a0, a1, 2f
    sd zero, 0(a0)
    addi a0, a0, 8
    j 1b
2:

    // Call C runtime init
    # call __libc_init_array

    // Call main
    call main
    li a0, 0
    call _exit

    // Exit loop
//1:
//    wfi
//    j 1b