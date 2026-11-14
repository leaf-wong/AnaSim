// See LICENSE for license details.
//#include <sifive/smp.h>
//#include <encoding.h>







.section .vectors, "ax"
//.section .vectors

  // These functions are declared in `sw/device/lib/handler.h`.
//  .extern handler_exception
  .extern handler_irq_software
//  .extern handler_irq_timer
//  .extern handler_irq_external



// .balign 256
crt_interrupt_vector:
  .global crt_interrupt_vector

  // Switch off compressed instructions so we know each instruction below is
  // exactly 4 bytes (one entry).
  .option push
  .option norvc

  // exception Handler and user software interrupt
  //j handler_exception
  j simple_exception_handler 


  // supervisor software interrupt
  unimp
  // reserved
  unimp
  // machine software interrupt handler
  j handler_irq_software
  

  // user timer interrupt
  unimp
  // supervisor timer interrupt
  unimp
  // reserved
  unimp
  // machine timer interrupt handler
  j handler_irq_timer

  // user external interrupt
  unimp
  // supervisor external interrupt
  unimp
  // reserved
  unimp
  // machine external interrupt handler
  j handler_irq_external

  // reserved
  unimp
  unimp
  unimp
  unimp




  // machine fast interrupt handler
  j handler_irq_fast_DMA
  j handler_irq_fast_IMC
  j handler_irq_fast_UART
  j handler_irq_fast_uDMA

  // vendor interrupts: on Ibex interrupt ids 30-16 are the "fast" interrupts
  .rept 11
  unimp
  .endr

  // vendor interrupts: on Ibex interrupt id 31 is for non-maskable interrupts
  unimp

  // Re-enable compressed instructions
  .option pop

  // Set size so vector can be disassembled
  //.size crt_interrupt_vector, .-crt_interrupt_vector
















/* This is defined in sifive/platform.h, but that can't be included from
 * assembly. */
#define CLINT_CTRL_ADDR 0x02000000

	.section .init
	.globl _start
	.type _start,@function

_start:
	.cfi_startproc
	.cfi_undefined ra
.option push
.option norelax
	la gp, __global_pointer$
.option pop
	la sp, _sp

#if defined(ENABLE_SMP)
	smp_pause(t0, t1)
#endif

	/* Load data section */
/*	la a0, _data_lma
	la a1, _data
	la a2, _edata
	bgeu a1, a2, 2f
1:
	lw t0, (a0)
	sw t0, (a1)
	addi a0, a0, 4
	addi a1, a1, 4
	bltu a1, a2, 1b
2:
*/

	/* Clear bss section */
/*	la a0, __bss_start
	la a1, _end
	bgeu a0, a1, 2f
1:
	sw zero, (a0)
	addi a0, a0, 4
	bltu a0, a1, 1b
2:
*/

	/* Call global constructors */
/*	la a0, __libc_fini_array
	call atexit
	call __libc_init_array
*/



#ifndef __riscv_float_abi_soft
	/* Enable FPU */
	li t0, MSTATUS_FS
	csrs mstatus, t0
	csrr t1, mstatus
	and t1, t1, t0
	beqz t1, 1f
	fssr x0
1:
#endif

#if defined(ENABLE_SMP)
	smp_resume(t0, t1)

	csrr a0, mhartid
	bnez a0, 2f
#endif

	auipc ra, 0
	addi sp, sp, -16
#if __riscv_xlen == 32
	sw ra, 8(sp)
#else
	sd ra, 8(sp)
#endif

	/* argc = argv = 0 */
	li a0, 0
	li a1, 0
	call main
	tail exit
1:
	j 1b

#if defined(ENABLE_SMP)
2:
	la t0, trap_entry
	csrw mtvec, t0

	csrr a0, mhartid
	la t1, _sp
	slli t0, a0, 10
	sub sp, t1, t0

	auipc ra, 0
	addi sp, sp, -16
#if __riscv_xlen == 32
	sw ra, 8(sp)
#else
	sd ra, 8(sp)
#endif

	call secondary_main
	tail exit

1:
	j 1b
#endif
	.cfi_endproc



/* -------------------------------------- */

#define EXIT_SYSCALL 93
#define SIM_CTRL_BASE 0x20000
#define SIM_CTRL_OUT 0x0
#define SIM_CTRL_CTRL 0x8


.section .text

default_exc_handler:
/*  jal x0, simple_exc_handler */

timer_handler:
/*  jal x0, simple_timer_handler */

reset_handler:
  /* set all registers to zero */
/*
  mv  x1, x0
  mv  x2, x1
  mv  x3, x1
  mv  x4, x1
  mv  x5, x1
  mv  x6, x1
  mv  x7, x1
  mv  x8, x1
  mv  x9, x1
  mv x10, x1
  mv x11, x1
  mv x12, x1
  mv x13, x1
  mv x14, x1
  mv x15, x1
  mv x16, x1
  mv x17, x1
  mv x18, x1
  mv x19, x1
  mv x20, x1
  mv x21, x1
  mv x22, x1
  mv x23, x1
  mv x24, x1
  mv x25, x1
  mv x26, x1
  mv x27, x1
  mv x28, x1
  mv x29, x1
  mv x30, x1
  mv x31, x1
  */

  /* stack initilization */
/*  la   x2, _stack_start */




main_entry:
  /* jump to main program entry point (argc = argv = 0) */
  addi x10, x0, 0
  addi x11, x0, 0
  jal x1, main

//RVTEST_CODE_END
        li a7, EXIT_SYSCALL;
        ecall

  /* Halt simulation */
  li x5, SIM_CTRL_BASE + SIM_CTRL_CTRL
  li x6, 1
  sw x6, 0(x5)

  /* If execution ends up here just put the core to sleep */
sleep_loop:
  wfi
  j sleep_loop






