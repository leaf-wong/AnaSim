// crt0.S
.section .text
.globl _start
_start:
    // Set up stack
    la sp, __stack_start

    // Optional BSS init
    la a0, __bss_start
    la a1, __bss_end

    li t2, 0x80001000     # safe known location
    sd a0, 0(t2)          # store __bss_start
    sd a1, 8(t2)          # store __bss_end

    li t3, 0x80000000     # test address
    li t4, 0xdeadbeef
    sd t4, 0(t3)
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