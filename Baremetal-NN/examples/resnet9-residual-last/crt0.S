// crt0.S
.section .text
.globl trap_entry
    .balign 4                # make sure trap_entry is 4-byte aligned to avoid trap entrying failure
trap_entry:
    la   t0, external_interrupt_handler  # pseudo-op expands to auipc+addi
    jr   t0

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