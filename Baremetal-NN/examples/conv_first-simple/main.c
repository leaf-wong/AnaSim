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

#define L2CACHE_BASE_ADDR 0x20000000

#define IMC_BASE_ADDR 0x10000000
#define IMC_AMEM_0   (IMC_BASE_ADDR + 0x300000)
#define DMA_ROWIDX (IMC_BASE_ADDR + 0x2c)
#define DRAM2SP_IN (IMC_BASE_ADDR + 0x14)
#define SP_IN2AMEM (IMC_BASE_ADDR + 0x24)

#define IMC_DIM 64
#define INPUT_SIZE 3072
#define INPUT_END_ADDRESS 65536

#define W_ZERO_POINT     32
#define BIAS_ZERO_POINT     32
#define BIAS_SCALE     1

#define INPUT_SCALE     0.07960

#define INCLUDE_FILE(section, path, sym) \
    __asm__(".section " section "\n" \
            ".global " #sym "_data\n" \
            #sym "_data:\n" \
            ".incbin \"" path "\"\n" \
            ".global " #sym "_end\n" \
            #sym "_end:\n" \
            ".byte 0\n" \
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

  
  // load the weight data block from the model.bin file
  INCLUDE_FILE(".rodata", "../cifar-10-batches-bin/test_batch.bin", cifar10_test_batch);
  extern uint8_t cifar10_test_batch_data[];
  extern size_t cifar10_test_batch_start[];
  extern size_t cifar10_test_batch_end[];

  //loading input from off-chip
  //TO DO: Input padding and reshuffle. Can be fixed?
  printf("setting input data...\n");
  printf("input label=%d\n", cifar10_test_batch_data[0]);
  

  for (int i = 0; i < 32; i += 1) {
    for (int j = 0; j < 32; j += 1) {
      for (int k = 0; k < IMC_DIM; k += 1) {
        if (k == 0) {
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i * 32 * IMC_DIM + j * IMC_DIM + k] = (uint8_t)((cifar10_test_batch_data[1 + k * 32 * 32 + i * 32  + j]*19385+1000000/2)/1000000) ;
        }
        else if (k == 1) {
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i * 32 * IMC_DIM + j * IMC_DIM + k] = (uint8_t)((cifar10_test_batch_data[1 + k * 32 * 32 + i * 32  + j]*19667+1000000/2)/1000000) ;
        }
        else if (k == 2) {
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i * 32 * IMC_DIM + j * IMC_DIM + k] = (uint8_t)((cifar10_test_batch_data[1 + k * 32 * 32 + i * 32  + j]*19510+1000000/2)/1000000) ;
        }
        else {
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i * 32 * IMC_DIM + j * IMC_DIM + k] = 0;
        }
      }
    }
  }
  printf("first pixel in R=%d G=%d B=%d\n", cifar10_test_batch_data[1], cifar10_test_batch_data[1025], cifar10_test_batch_data[2049]);
  printf("first pixel after transform in R=%d G=%d B=%d\n", (uint8_t)((cifar10_test_batch_data[1]*19385+1000000/2)/1000000), (uint8_t)((cifar10_test_batch_data[1025]*19667+1000000/2)/1000000), (uint8_t)((cifar10_test_batch_data[2049]*19510+1000000/2)/1000000));

  
  for (size_t j = 0; j < 32; j += 1) {
    for (size_t k = 0; k < 32; k += 1) {
      for (size_t l = 0; l < 1; l += 1) {
        write_reg(DMA_ROWIDX, j* 32*1 + k*1 + l);
        write_reg(DRAM2SP_IN, j* 32*1 + k*1 + l);
        write_reg(SP_IN2AMEM, j* 32*1 + k*1 + l);
      }
    }
  }

    /*if (i < INPUT_SIZE)
      ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i] =  1;
    else 
      ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[i] =  0;*/
    //printf("input=0x%04x\n", model->input_1.data[i]);
  printf("input finish\n");

  
  
  //loading weight from off-chip
  int input_div_q = model->actor_0_weight.shape[1] / IMC_DIM;
  int input_div_r = model->actor_0_weight.shape[1] % IMC_DIM;
  input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  int output_div_q = model->actor_0_weight.shape[0] / IMC_DIM;
  int output_div_r = model->actor_0_weight.shape[0] % IMC_DIM;
  output_div_q = (output_div_r > 0) ? output_div_q + 1 : output_div_q;

  //printf("input_div_q=%ld, input_div_r=%ld, output_div_q=%ld, output_div_r=%ld\n", input_div_q, input_div_r, output_div_q, output_div_r);
  // Conv Layer weight loading
  for (int i = 0; i < output_div_q; i += 1) {
    for (int j = 0; j < model->actor_0_weight.shape[2]; j += 1) {
      for (int k = 0; k < model->actor_0_weight.shape[3]; k += 1) {
        for (int l = 0; l < input_div_q; l += 1) {
          for (int m = 0; m < IMC_DIM; m += 1) {
            for (int n = 0; n < IMC_DIM; n += 1) {
              if (input_div_r > 0)
                if (n<input_div_r)
                  ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+l*IMC_DIM*IMC_DIM+n*IMC_DIM+m] =  model->actor_0_weight.data[i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*input_div_r+j*model->actor_0_weight.shape[3]+k+l*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_r+m*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*input_div_r+n*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]];
                else
                  ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+l*IMC_DIM*IMC_DIM+n*IMC_DIM+m] =  W_ZERO_POINT;
              else 
                ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+l*IMC_DIM*IMC_DIM+n*IMC_DIM+m] =  model->actor_0_weight.data[i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]+k+l*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*IMC_DIM+m*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM+n*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]];
            }
          }
        }
        //printf("%ld th row of DRAM, 1st W=%d, 3th W=%d, 6th W=%d, 9th W=%d\n", k, model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+0], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+2], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+5], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+8]);
        printf("First output channel %d th row, %ld th col of Weight, 1st W=%d, 2nd W=%d, 3rd W=%d 4th W=%d\n",j, k, ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+0],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+1*IMC_DIM],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+2*IMC_DIM],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+3*IMC_DIM]);
        printf("Last output channel %d th row, %ld th col of Weight, 1st W=%d, 2nd W=%d, 3rd W=%d 4th W=%d\n",j, k, ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+63],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+1*IMC_DIM+63],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+2*IMC_DIM+63],
               ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+j*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+k*input_div_q*IMC_DIM*IMC_DIM+3*IMC_DIM+63]);
      }
    }
  }

  //loading bias from off-chip
  for (int i = 0; i < output_div_q; i += 1) {
    for (int j = 0; j < IMC_DIM; j += 1) {
      if (input_div_r > 0)
        ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*input_div_r+i*IMC_DIM+j];
      else
        ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j];
    }
    printf("%ld th row, 1st BIAS=%d, 10th BIAS=%d, last BIAS=%d\n", i, ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+0], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+9], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*model->actor_0_weight.shape[2]*model->actor_0_weight.shape[3]*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+63]);
  }

  /* Linear Layer weight loading
   * The weight is divided into 64x64 blocks, each block is loaded into L2Cache.
   * The input and output are divided into q and r, where q is the number of blocks and r is the remainder.
   */
  /*
  for (int i = 0; i < output_div_q; i += 1) {
    for (int j = 0; j < input_div_q; j += 1) {
      for (int k = 0; k < IMC_DIM; k += 1) {
        for (int l = 0; l < IMC_DIM; l += 1) {
          if ((input_div_r > 0)&&(output_div_r>0))
            if ((k<output_div_r)&&(l<input_div_r))
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  model->actor_0_weight.data[i*input_div_q*output_div_r*input_div_r+j*output_div_r*input_div_r+k*input_div_r+l];
            else 
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  W_ZERO_POINT;
          else if ((input_div_r > 0)&&(output_div_r==0))
            if (l<input_div_r) {
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  model->actor_0_weight.data[i*input_div_q*IMC_DIM*input_div_r+j*IMC_DIM*input_div_r+k*input_div_r+l];
            }
            else 
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  W_ZERO_POINT;
          else if ((input_div_r == 0)&&(output_div_r>0))
            if (k<output_div_r) {
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  model->actor_0_weight.data[i*input_div_q*output_div_r*IMC_DIM+j*IMC_DIM+k*input_div_q*IMC_DIM+l];
            }
            else 
              ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  W_ZERO_POINT;
          else 
            ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l] =  model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+l];
        }
        //printf("%ld th row of DRAM, 1st W=%d, 3th W=%d, 6th W=%d, 9th W=%d\n", k, model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+0], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+2], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+5], model->actor_0_weight.data[i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*input_div_r+8]);
        if (k<output_div_r)
          printf("%d th array, %ld th row of L2Cache, 1st W=%d, 3th W=%d, 6th W=%d, last W=%d\n",j, k, ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+0], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+2], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+5], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+i*input_div_q*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM+63]);
      }
    }
  }

  //loading bias from off-chip
  for (int i = 0; i < output_div_q; i += 1) {
    for (int j = 0; j < IMC_DIM; j += 1) {
      if ((input_div_r > 0)&&(output_div_r>0))
        if (j<output_div_r)
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[output_div_r*input_div_r+i*output_div_r+j];
        else
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  BIAS_ZERO_POINT;
      else if ((input_div_r > 0)&&(output_div_r==0))
        ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[output_div_q*input_div_r*IMC_DIM+i*IMC_DIM+j];
      else if ((input_div_r == 0)&&(output_div_r > 0))
        if (j<output_div_r)
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[input_div_q*output_div_r*IMC_DIM+i*output_div_r+j];
        else
          ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  BIAS_ZERO_POINT;
      else
        ((volatile uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j] =  model->actor_0_weight.data[output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+j];
    }
    printf("%ld th row, 1st BIAS=%d, 10th BIAS=%d, last BIAS=%d\n", i, ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+0], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+9], ((uint8_t*)(L2CACHE_BASE_ADDR))[INPUT_END_ADDRESS+output_div_q*input_div_q*IMC_DIM*IMC_DIM+i*IMC_DIM+63]);
  }*/

  
  #ifdef CONFIG_TOOLCHAIN_RISCV
    cycles = READ_CSR("mcycle");
  #endif
  model_forward(model);
  #ifdef CONFIG_TOOLCHAIN_RISCV
    cycles = READ_CSR("mcycle") - cycles;
    printf("RISC-V cycles: %lu\n", cycles);
  #endif

  for (int j = 0; j < model->output.shape[1]; j += 1) {
      volatile int8_t *imc_ptr = (volatile int8_t *)IMC_AMEM_0; // using volatile to prevent the compiler from optimizing per-byte accessing into bulk loads 
      //printf("%ld th output =%ld\n", j, ((int8_t*)(IMC_AMEM_0))[16*IMC_DIM + j]);
      model->output.data[ + j] = imc_ptr[ INPUT_END_ADDRESS + 59520 + j];
      //model->output.data[ + j] = imc_ptr[ INPUT_END_ADDRESS + 0 + j];
  }
  //printf("output:\n");
  //nn_print_tensor2d_i6(&model->output);
  //nn_print_tensor2d_i6_to_f32(&model->output, 0.000992063492);
  // tensor([[ 0.098,  0.041,  0.022, -0.045, -0.162, -0.034,  0.084,  0.035,  0.021,  0.102,  0.032,  0.047]])
  /* output tensor([[ 0.008, -0.077, -0.310, -0.859, -0.536,  0.845,  0.183,  0.163,  0.978,
         -0.733]])*/
  
  return 0;
}
