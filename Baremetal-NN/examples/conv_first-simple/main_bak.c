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
#include "model.h"


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

int main() {

  enable_vector_operations();
  uint64_t mstatus;
  asm volatile("csrr %0, mstatus" : "=r"(mstatus));
  mstatus |= (1 << 13);  // FS[1:0] = 01
  asm volatile("csrw mstatus, %0" :: "r"(mstatus));
  
  Model *model = malloc(sizeof(Model));

  size_t cycles;
  
  printf("initalizing model...\n");
  model_init(model);

  
  printf("setting input data...\n");
  for (int i = 0; i < 2; i += 1) {
    int8_t in = 1;
    //model->input_1.data[i] = as_f16(in);
    model->input_1.data[i] = in;
    //printf("input=0x%04x\n", model->input_1.data[i]);
  }
  printf("input finish\n");

  #ifdef CONFIG_TOOLCHAIN_RISCV
    cycles = READ_CSR("mcycle");
  #endif
  model_forward(model);
  #ifdef CONFIG_TOOLCHAIN_RISCV
    cycles = READ_CSR("mcycle") - cycles;
    printf("cycles: %lu\n", cycles);
  #endif

  printf("output:\n");
  nn_print_tensor2d_i6(&model->output);
  //for (int i = 0; i < 2; i += 1) {
  //  printf("%ld th output =%f\n", i, ((model->output.data[i])-32*3)/31.5);
  //}
  // tensor([[ 0.098,  0.041,  0.022, -0.045, -0.162, -0.034,  0.084,  0.035,  0.021,  0.102,  0.032,  0.047]])
  
  return 0;
}
