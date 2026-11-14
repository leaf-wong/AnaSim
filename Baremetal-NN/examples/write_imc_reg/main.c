/**
 * @file main.c
 * 
 * A simple example running an MLP model.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "riscv.h"
#include "nn.h"
//#include "model.h"

#define L2CACHE_BASE_ADDR 0x20000000

#define IMC_BASE_ADDR 0x10000000
#define IMC_AMEM_0   (IMC_BASE_ADDR + 0x300000)
#define IMC_IE     (IMC_BASE_ADDR + 0x10)
#define DRAM2SP_W (IMC_BASE_ADDR + 0x18)
#define DMA_ROWIDX (IMC_BASE_ADDR + 0x2c)
#define DRAM2SP_IN (IMC_BASE_ADDR + 0x14)
#define SP_IN2AMEM (IMC_BASE_ADDR + 0x24)

#define PLIC_BASE        0x0c000000
#define PLIC_CLAIM       (PLIC_BASE + 0x200004)
#define PLIC_COMPLETE    (PLIC_BASE + 0x200004)
#define PLIC_THRESHOLD    (PLIC_BASE + 0x200000)
#define PLIC_PRIORITY1  (PLIC_BASE + 4)
#define PLIC_ENABLE     (PLIC_BASE + 0x2000)

#define IMC_DIM 256
#define INPUT_SIZE 3072
#define INPUT_END_ADDRESS 8192

#define W_ZERO_POINT     32
#define BIAS_ZERO_POINT     32
#define BIAS_SCALE     1

#define INPUT_SCALE     0.07960

/*#define INCLUDE_FILE(section, path, sym) \
    __asm__(".section " section "\n" \
            ".global " #sym "_data\n" \
            #sym "_data:\n" \
            ".incbin \"" path "\"\n" \
            ".global " #sym "_end\n" \
            #sym "_end:\n" \
            ".byte 0\n" \
            ".previous\n");

#define INCLUDE_WORD_FILE(section, path, sym) \
    __asm__(".section " section "\n" \
            ".align 2\n" \
            ".global " #sym "_data\n" \
            #sym "_data:\n" \
            ".incbin \"" path "\"\n" \
            ".global " #sym "_end\n" \
            #sym "_end:\n" \
            ".byte 0\n" \
            ".previous\n");*/

  #define MAIN_INCLUDE_FILE(section, path, sym) \
    __asm__(".section " section "\n" \
            ".balign 4\n" \
            ".global " #sym "_start\n" \
            ".global " #sym "_data\n" \
            #sym "_start:\n" \
            #sym "_data:\n" \
            ".incbin \"" path "\"\n" \
            ".global " #sym "_end\n" \
            #sym "_end:\n" \
            ".balign 4\n" \
            ".previous\n");


        


volatile uint64_t tohost __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".fromhost"))) = 0;

static void enable_vector_operations() {
  #ifdef RISCV_VECTOR_EXTENSIONS
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}


//__attribute__((interrupt("machine")))
void external_interrupt_handler(void) {

  int irq = *(volatile uint32_t *)PLIC_CLAIM;


  write_reg(IMC_IE, 0);

  // Write back to complete it
  *(volatile uint32_t *)PLIC_COMPLETE = irq;

  //uintptr_t mepc;
  //asm volatile("csrr %0, mepc" : "=r"(mepc));
  //mepc += 4; // skip the trapped instruction (wfi is 4 bytes here)
  //asm volatile("csrw mepc, %0" :: "r"(mepc));
}

/*__attribute__((naked))
void trap_entry(void) {
    __asm__ volatile (
        "j external_interrupt_handler\n"
    );
}*/


int main() {

  enable_vector_operations();
  uint64_t mstatus;
  asm volatile("csrr %0, mstatus" : "=r"(mstatus));
  mstatus |= (1 << 13);  // FS[1:0] = 01 
  asm volatile("csrw mstatus, %0" :: "r"(mstatus));

  //interrupt_enable();
  //*(volatile uint32_t *)PLIC_PRIORITY1 = 1;
  //*(volatile uint32_t *)PLIC_ENABLE = (1 << 1);
  //asm volatile("csrw mtvec, %0" :: "r"(&external_interrupt_handler));
  //extern void trap_entry(void);
  //asm volatile("csrw mtvec, %0" :: "r"(&trap_entry));
  extern void external_interrupt_handler(void);
  
  asm volatile("csrs mie, %0" :: "r"(1 << 11)); // enable external interrupt
  asm volatile("csrs mstatus, %0" :: "r"(1 << 3)); // enable global interrupt
  
  write_reg(0x00218000, 1234);
  
  //printf("initalizing model...\n");

  
  return 0;
}
