// crt0.S
.section .text
.globl trap_entry
    .balign 4                # make sure trap_entry is 4-byte aligned to avoid trap entrying failure
trap_entry:
1:
    wfi
    j 1b
    
    /*addi sp, sp, -128

    sw ra,   0(sp)
    sw gp,   4(sp)
    sw tp,   8(sp)
    sw t0,  12(sp)
    sw t1,  16(sp)
    sw t2,  20(sp)
    sw s0,  24(sp)
    sw s1,  28(sp)
    sw a0,  32(sp)
    sw a1,  36(sp)
    sw a2,  40(sp)
    sw a3,  44(sp)
    sw a4,  48(sp)
    sw a5,  52(sp)
    sw a6,  56(sp)
    sw a7,  60(sp)
    sw s2,  64(sp)
    sw s3,  68(sp)
    sw s4,  72(sp)
    sw s5,  76(sp)
    sw s6,  80(sp)
    sw s7,  84(sp)
    sw s8,  88(sp)
    sw s9,  92(sp)
    sw s10, 96(sp)
    sw s11,100(sp)
    sw t3, 104(sp)
    sw t4, 108(sp)
    sw t5, 112(sp)
    sw t6, 116(sp)

    # Call the C handler (position-independent)
    la t0, external_interrupt_handler    # expands to auipc/addi
    jalr ra, t0, 0                       # call handler (ra gets return addr into trampoline)

    # After the C handler returns, advance mepc to skip the trapped instruction
    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0

    # Restore registers (reverse order)
    lw t6, 116(sp)
    lw t5, 112(sp)
    lw t4, 108(sp)
    lw t3, 104(sp)
    lw s11,100(sp)
    lw s10, 96(sp)
    lw s9,  92(sp)
    lw s8,  88(sp)
    lw s7,  84(sp)
    lw s6,  80(sp)
    lw s5,  76(sp)
    lw s4,  72(sp)
    lw s3,  68(sp)
    lw s2,  64(sp)
    lw a7,  60(sp)
    lw a6,  56(sp)
    lw a5,  52(sp)
    lw a4,  48(sp)
    lw a3,  44(sp)
    lw a2,  40(sp)
    lw a1,  36(sp)
    lw a0,  32(sp)
    lw s1,  28(sp)
    lw s0,  24(sp)
    lw t2,  20(sp)
    lw t1,  16(sp)
    lw t0,  12(sp)
    lw tp,   8(sp)
    lw gp,   4(sp)
    lw ra,   0(sp)

    addi sp, sp, 128
    mret*/

.globl _start
_start:
    //.cfi_startproc

    // Set up stack
    la sp, __stack_start

    // Optional BSS init
    la a0, __bss_start
    la a1, __bss_end

1:
    bgeu a0, a1, 2f
    sw zero, 0(a0)
    addi a0, a0, 4
    j 1b
2:

    // Install trap handler
    la t0, trap_entry
    //andi t0, t0, -4    # clear low two bits to force direct mode
    csrw mtvec, t0

    // Call C runtime init
    # call __libc_init_array

    // Call main
    call main
    li a0, 0
    call _exit

    //.cfi_endproc

    // Exit loop
//1:
//    wfi
//    j 1b