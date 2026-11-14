/**
 * @file nn_i6.h
 * @brief Baremetal-NN Library functions for INT6 (i6) numbers
 *
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_I6_H
#define __NN_I6_H

#include "float16.h"
#include "nn_f32.h"

#define IMC_BASE_ADDR 0x10000000

#define IMC_REG_CONTROL (IMC_BASE_ADDR + 0x00)
#define IMC_REG_STATUS  (IMC_BASE_ADDR + 0x04)
#define IMC_REG_OUTPUT  (IMC_BASE_ADDR + 0x08)
#define AMEM_RD_OFFSET  (IMC_BASE_ADDR + 0x0c)
#define IMC_IE     (IMC_BASE_ADDR + 0x10)

#define DRAM2SP_IN (IMC_BASE_ADDR + 0x14)
#define DRAM2SP_W (IMC_BASE_ADDR + 0x18)
#define DRAM2SP_IN_BURST (IMC_BASE_ADDR + 0x1c)
#define DRAM2SP_W_BURST (IMC_BASE_ADDR + 0x20)
#define SP_IN2AMEM (IMC_BASE_ADDR + 0x24)
#define SP_W2IMC (IMC_BASE_ADDR + 0x28)
#define DMA_ROWIDX (IMC_BASE_ADDR + 0x2c)

#define ANU_ADD (IMC_BASE_ADDR + 0x30)
#define ANU_ADD_RELU (IMC_BASE_ADDR + 0x34)

#define AMEM_ROW_REFRESH (IMC_BASE_ADDR + 0x38)

#define BIAS_ROW_RD_OFFSET (IMC_BASE_ADDR + 0x3c)
#define SP_W2BIAS_ROW (IMC_BASE_ADDR + 0x40)
#define BIAS_EXIST (IMC_BASE_ADDR + 0x44)

#define DMA_BURST_NUM (IMC_BASE_ADDR + 0x48)
#define SP_W_WR_BASE (IMC_BASE_ADDR + 0x4c)
#define AMEM_RD_BASE (IMC_BASE_ADDR + 0x50)
#define INPUT_ITR (IMC_BASE_ADDR + 0x54)
#define OUTPUT_ITR (IMC_BASE_ADDR + 0x58)
#define OUTPUT_COL (IMC_BASE_ADDR + 0x5c)
#define OUTPUT_ROW (IMC_BASE_ADDR + 0x60)
#define WEIGHT_COL (IMC_BASE_ADDR + 0x64)
#define WEIGHT_ROW (IMC_BASE_ADDR + 0x68)
#define INPUT_REMAINDER (IMC_BASE_ADDR + 0x6c)
#define OUTPUT_REMAINDER (IMC_BASE_ADDR + 0x70)
#define NEXT_CONV_CHANNEL (IMC_BASE_ADDR + 0x74)
#define PADDING (IMC_BASE_ADDR + 0x78)
#define STRIDE (IMC_BASE_ADDR + 0x7c)
#define DMA_READ_BASE (IMC_BASE_ADDR + 0x80)
#define AMEM_SCALE_CFG (IMC_BASE_ADDR + 0x84)
#define BIAS_SCALE_CFG (IMC_BASE_ADDR + 0x88)
#define AMEM_RD_BASE2 (IMC_BASE_ADDR + 0x8c)
#define REORG_CAT (IMC_BASE_ADDR + 0x90)
#define DMA_READ_INPUT_BASE (IMC_BASE_ADDR + 0x94)

#define LINEAR_LAYER (IMC_BASE_ADDR + 0x100)
#define CONV_LAYER (IMC_BASE_ADDR + 0x104)
#define AVGPOOL_LAYER (IMC_BASE_ADDR + 0x108)
#define MAXPOOL_LAYER (IMC_BASE_ADDR + 0x120)
#define ReLU_LAYER (IMC_BASE_ADDR + 0x10c)
#define ADD_LAYER (IMC_BASE_ADDR + 0x110)

#define CONV_LAYER_A (IMC_BASE_ADDR + 0x114)
#define CONV_LAYER_A_CONV1 (IMC_BASE_ADDR + 0x118)
#define CONV_LAYER_A_OTHER (IMC_BASE_ADDR + 0x11c)

#define IMC_SP_INPUT  (IMC_BASE_ADDR + 0x1000)
#define IMC_SP_WEIGHT  (IMC_BASE_ADDR + 0x2000)
#define IMC_AMEM_0   (IMC_BASE_ADDR + 0x500000)
//#define IMC_ARRAY_0  (IMC_BASE_ADDR + 0x400000)

#define NUM_FUSED_LAYERS (8)
#define LAYER_FUSED (IMC_BASE_ADDR + 0x124)
#define LAYER_FUSE_ENABLE (IMC_BASE_ADDR + 0x128)
#define LAYER_INDEX (IMC_BASE_ADDR + 0x12c)
#define LAYER_TYPE (IMC_BASE_ADDR + 0x130)

#define IMC_CONTROL_MVM     (1)
#define IMC_CONTROL_RST     (0)


#define IMC_DIM (1024)
//#define IMC_AMEM_TEMP_BASE   8192 // start from 0x380000 


#ifdef CONFIG_BACKEND_RISCV_V
  #include "riscv_vector.h"
#endif

/**
 * Tensor0D_I6
 *
 * @brief A 0D tensor (scalar) with a int8_t data type.
 */
typedef struct {
  int8_t data;
} Tensor0D_I6;

/**
 * Tensor1D_I6
 *
 * @brief A 1D tensor with a int8_t data type.
 */
typedef struct {
  size_t shape[1];
  int8_t *data;
} Tensor1D_I6;

/**
 * Tensor2D_I6
 *
 * @brief A 2D tensor with a int8_t data type.
 */
typedef struct {
  size_t shape[2];
  int8_t *data;
} Tensor2D_I6;

/**
 * Tensor4D_I6
 *
 * @brief A 4D tensor with a int8_t data type.
 */
typedef struct {
  size_t shape[4];
  int8_t *data;
} Tensor4D_I6;

static inline void write_reg(uint32_t addr, uint32_t val) {
    //printf("Write to addr=0x%x val=0x%x\n", addr, val);
    *(volatile uint32_t *)addr = val;

}

static inline uint32_t read_reg(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}




/**
 * nn_linear_i6
 *
 * @brief Linear neural network layer.
 *
 * y[i][j] = x[i][k] * weight[j][k] + bias[j]
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 */
void nn_linear_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x, const Tensor2D_I6 *weight, const Tensor1D_I6 *bias, char * dmem_weight_address, int sp_w_write_index, int amem_write_index, int amem_read_index, int amem_scale, int bias_scale) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  volatile int batch_size = x->shape[0];
  volatile int in_features = x->shape[1];
  volatile int out_features = weight->shape[0];

  int8_t *x_batch_data = x->data;
  int8_t *y_batch_data = y->data;

  volatile int input_div_q = weight->shape[1] / IMC_DIM;
  volatile int input_div_r = weight->shape[1] % IMC_DIM;
  input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  volatile int output_div_q = weight->shape[0] / IMC_DIM;
  volatile int output_div_r = weight->shape[0] % IMC_DIM;
  output_div_q = (output_div_r > 0) ? output_div_q + 1 : output_div_q;
  
  volatile int sp_weight_write_base=sp_w_write_index;
  //int l2cache_read_base=input_div_q*batch_size;
  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=amem_read_index;

  write_reg(LAYER_FUSE_ENABLE, 0);
  /*for (int i = 0; i < output_div_q; i += 1) {
    write_reg(DMA_ROWIDX, sp_weight_write_base+i);
    write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*input_div_q*output_div_r*IMC_DIM+i*IMC_DIM);
    asm volatile("wfi");
  }*/

  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address);

  //TODO: input should alreay in AMEM
  /*for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < input_div_q; j += 1) {
      write_reg(DMA_ROWIDX, i*input_div_q + j);
      write_reg(DRAM2SP_IN, i*input_div_q + j);
      write_reg(SP_IN2AMEM, i*input_div_q + j);
    }
  }*/

  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  write_reg(INPUT_ITR, input_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(AMEM_SCALE_CFG, amem_scale);
  write_reg(BIAS_SCALE_CFG, bias_scale);



  if (bias) {
    write_reg(BIAS_EXIST, 1);
  }


  for (int i = 0; i < batch_size; i += 1) {
    int8_t *x_data = x_batch_data;
    int8_t *y_data = y_batch_data;

    write_reg(DMA_ROWIDX, amem_write_base+i*output_div_q);
    write_reg(AMEM_RD_BASE, amem_read_base+i*input_div_q);

    write_reg(LINEAR_LAYER, 1);
    asm volatile("wfi");

    
    //just for debug print
    //printf("output read start\n");

    /*
    for (int j = 0; j < out_features; j += 1) {
      volatile int8_t *imc_ptr = (volatile int8_t *)IMC_AMEM_0; // using volatile to prevent the compiler from optimizing per-byte accessing into bulk loads 

      y_data[ + j] = imc_ptr[ (amem_write_base+ i*output_div_q)*IMC_DIM + j];
    }
    */

    
    
    x_batch_data += in_features;
    y_batch_data += out_features;
  }
}

/**
 * nn_conv2d_i6
 *
 * @brief Convlutional neural network layer.
 *
 * y[i][j] = x[i][k] * weight[j][k] + bias[j]
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 * @param stride The stride of the convolution (1,1).
 * @param padding The padding of the convolution (1,1).
 * @param dilation The dilation of the convolution (1,1).   
 * @param groups The number of groups in the convolution (1).
 *  volatile is for hardware interaction (Memory-Mapped I/O), used to tell the compiler that a variable's value can be modified by factors outside the program's normal control flow
 */
void nn_conv2d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, const Tensor4D_I6 *weight, const Tensor1D_I6 *bias, char * dmem_weight_address, int sp_w_write_index, int amem_write_index, int amem_read_index, int next_conv_channel, int amem_scale, int bias_scale, int stride, int padding, int dilation, int groups) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");

  volatile int batch_size = x->shape[0];
  volatile int in_features = (x->shape[1] * x->shape[2] * x->shape[3]);
  volatile int out_features = weight->shape[0];

  int8_t *x_batch_data = x->data;
  int8_t *y_batch_data = y->data;

  //volatile int input_div_q = (weight->shape[1] * weight->shape[2] * weight->shape[3]) / IMC_DIM;
  //volatile int input_div_r = (weight->shape[1] * weight->shape[2] * weight->shape[3]) % IMC_DIM;
  //input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  volatile int output_div_q = (y->shape[1] / IMC_DIM);
  volatile int output_div_r = (y->shape[1] % IMC_DIM);
  output_div_q = ((output_div_r > 0) ? output_div_q + 1 : output_div_q);

  volatile int amem_div_q = (x->shape[1] / IMC_DIM);
  volatile int input_div_r = (x->shape[1] % IMC_DIM);
  amem_div_q = (input_div_r > 0) ? amem_div_q + 1 : amem_div_q;
  
  volatile int sp_weight_write_base=sp_w_write_index; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index; //16
  volatile int amem_read_base=amem_read_index; //0

  volatile int i=0;
  volatile int j=0;

  //printf("output_div_q=%d, input_div_q=%d, amem_div_q=%d\n", output_div_q, input_div_q, amem_div_q);

  // SP_W layout for conv2-8
  //   |   0-255   |
  // 0 |   Weight  | (H W CIN COUT)
  // 1 |   Weight  | Weight
  //        ...
  // N |    Bias   | No matter COUT = 128 or 256, bias always occupy one row (supply BIAS_ZERO_POINT if needed) to align sp_weight_write_base of every layer

  //write_reg(DMA_BURST_NUM, IMC_DIM);
  //conv2 layer
  /*if (input_div_r == (IMC_DIM/4)){
    for ( i = 0; i < (output_div_q); i += 1) {
      write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2]-1) * weight->shape[3]*input_div_r*output_div_r/IMC_DIM+i);
      write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+i*IMC_DIM);
      //printf("DRAM2SP_W: first=%d 3rd=%d last-1=%d last=%d\n", dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ 2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-1]);
      asm volatile("wfi");
      //asm volatile("wfi" ::: "memory");
      write_reg(DMA_BURST_NUM, (weight->shape[3]*input_div_r*output_div_r)/IMC_DIM);
      for (  j = 1; j < (amem_div_q* weight->shape[2]) ; j += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r/IMC_DIM+(j-1)*weight->shape[3]*input_div_r*output_div_r/IMC_DIM);
        write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+j*weight->shape[3]*input_div_r*output_div_r));
        asm volatile("wfi");
        //printf("DRAM2SP_W_BURST: first=%d 12th=%d last-1=%d last=%d\n", dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM + IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+11*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-2)*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-1)*IMC_DIM+ IMC_DIM-1]);
      }
    }
    write_reg(DMA_BURST_NUM, (2*output_div_r*output_div_r)/IMC_DIM);
    write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2]-1) * weight->shape[3]*input_div_r*output_div_r/IMC_DIM+output_div_q);
    write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+output_div_q*output_div_r));
    asm volatile("wfi");
  }
  //conv3-5 layer
  else if (input_div_r == (IMC_DIM/2)){
    if (output_div_r >0){
      for ( i = 0; i < (output_div_q); i += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-2)*input_div_r*output_div_r/IMC_DIM+i);
        write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+i*IMC_DIM);
        //printf("DRAM2SP_W: first=%d 3rd=%d last-1=%d last=%d\n", dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ 2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-1]);
        asm volatile("wfi");
        //asm volatile("wfi" ::: "memory");
        write_reg(DMA_BURST_NUM, (2*input_div_r*output_div_r)/IMC_DIM);
        for ( j = 1; j < (5) ; j += 1) {
          if (j>2){
            write_reg(DMA_BURST_NUM, (input_div_r*output_div_r)/IMC_DIM);
          }

          write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r/IMC_DIM+(j-1)*2*input_div_r*output_div_r/IMC_DIM);
          if (j<3){
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+ j*weight->shape[3]*input_div_r*output_div_r));
            asm volatile("wfi");
          }
          else if (j==3){
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+ 0*weight->shape[3]*input_div_r*output_div_r+ 2*input_div_r*output_div_r));
            asm volatile("wfi");
            write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r/IMC_DIM+((j-1)*2+1)*input_div_r*output_div_r/IMC_DIM);
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+ 1*weight->shape[3]*input_div_r*output_div_r+ 2*input_div_r*output_div_r));
            asm volatile("wfi");

          }
          else{
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+ 2*weight->shape[3]*input_div_r*output_div_r+ 2*input_div_r*output_div_r));
            asm volatile("wfi");
          }

          //printf("j=%d\n", j);
          //printf("DRAM2SP_W_BURST: first=%d 12th=%d last-1=%d last=%d\n", dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM + IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+11*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-2)*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-1)*IMC_DIM+ IMC_DIM-1]);
        }
      }
      write_reg(DMA_BURST_NUM, (2*next_conv_channel*output_div_r)/IMC_DIM);
      write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-2)*input_div_r*output_div_r/IMC_DIM+output_div_q);
      write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*output_div_r+output_div_q*output_div_r));
      asm volatile("wfi");
    }

    else {
      for ( i = 0; i < (output_div_q); i += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-2)*input_div_r+i);
        write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+i*IMC_DIM);
        //printf("DRAM2SP_W: first=%d 3rd=%d last-1=%d last=%d\n", dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ 2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-1]);
        asm volatile("wfi");
        //asm volatile("wfi" ::: "memory");
        write_reg(DMA_BURST_NUM, 2*input_div_r);
        for ( j = 1; j < (5) ; j += 1) {
          if (j>2)
            write_reg(DMA_BURST_NUM, input_div_r);
          write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r+(j-1)*2*input_div_r);

          if (j<3){
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+ j*weight->shape[3]*input_div_r*IMC_DIM));
            asm volatile("wfi");
          }
          else if (j==3){
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+ 0*weight->shape[3]*input_div_r*IMC_DIM+ 2*input_div_r*IMC_DIM));
            asm volatile("wfi");
            write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r+( (j-1)*2+1)*input_div_r);
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+ 1*weight->shape[3]*input_div_r*IMC_DIM+ 2*input_div_r*IMC_DIM));
            asm volatile("wfi");

          }
          else{
            write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+ 2*weight->shape[3]*input_div_r*IMC_DIM+ 2*input_div_r*IMC_DIM));
            asm volatile("wfi");
          }

          //write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+j*2*input_div_r*IMC_DIM));
          //printf("DRAM2SP_W_BURST: first=%d 12th=%d last-1=%d last=%d\n", dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM + IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+11*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-2)*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-1)*IMC_DIM+ IMC_DIM-1]);
        }
      }
      write_reg(DMA_BURST_NUM, IMC_DIM);
      write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-2)*input_div_r+output_div_q);
      write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*input_div_r*IMC_DIM+output_div_q*IMC_DIM));
      asm volatile("wfi");
    }
    
  }

  //conv6-8 layer
  else {
    for ( i = 0; i < (output_div_q); i += 1) {
      write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-1)*IMC_DIM+i);
      write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+i*IMC_DIM);
      //printf("DRAM2SP_W: first=%d 3rd=%d last-1=%d last=%d\n", dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ 2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-2], dmem_weight_address[output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+ IMC_DIM-1]);
      asm volatile("wfi");
      //asm volatile("wfi" ::: "memory");
      write_reg(DMA_BURST_NUM, IMC_DIM);
      for ( j = 1; j < (amem_div_q* weight->shape[2] * weight->shape[3]) ; j += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM+(j-1)*IMC_DIM);
        //const uint8_t *p = dmem_weight_address + i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM + j * (IMC_DIM * IMC_DIM); // bytes
        //printf("HOST: addr=%p p[0]=%u p[14]=%u p[11]=%u p[12]=%u p[last]=%u\n", (void*)p, p[IMC_DIM-1], p[14*IMC_DIM + IMC_DIM-1], p[11*IMC_DIM + IMC_DIM-1], p[12*IMC_DIM + IMC_DIM-1], p[(IMC_DIM-1)*IMC_DIM + IMC_DIM-1]);
        //printf("DRAM2SP_W_BURST=%lu\n", (uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM);
        write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM));
        asm volatile("wfi");
        //printf("DRAM2SP_W_BURST: first=%d 12th=%d last-1=%d last=%d\n", dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM + IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+11*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-2)*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-1)*IMC_DIM+ IMC_DIM-1]);
      }
    }
    if (next_conv_channel==14)
      write_reg(DMA_BURST_NUM, 40);
    
    write_reg(DMA_ROWIDX, sp_weight_write_base+(output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]-1)*IMC_DIM+output_div_q);
    write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+output_div_q*IMC_DIM));
    asm volatile("wfi");
  }*/
  write_reg(LAYER_FUSE_ENABLE, 0);
  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address);
  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);

  write_reg(PADDING, padding);
  write_reg(STRIDE, stride);
  write_reg(AMEM_SCALE_CFG, amem_scale);
  write_reg(BIAS_SCALE_CFG, bias_scale);


  if (bias) {
    write_reg(BIAS_EXIST, 1);
  }
  else {
    write_reg(BIAS_EXIST, 0);
  }


  for ( i = 0; i < batch_size; i += 1) {
    int8_t *x_data = x_batch_data;
    int8_t *y_data = y_batch_data;

    write_reg(DMA_ROWIDX, amem_write_base+i*output_div_q* y->shape[2] * y->shape[3]);
    write_reg(AMEM_RD_BASE, amem_read_base+i*amem_div_q* x->shape[2] * x->shape[3]); // remove batch supporting can save RSICV cycle?

    //for resnet9
    //write_reg(CONV_LAYER_A_OTHER, 1);
    
    write_reg(CONV_LAYER_A, 1);

    asm volatile("wfi");

    //just for debug prints
    //while (read_reg(IMC_REG_STATUS) == 0) { printf("waiting\n");}
    //printf("output read start\n");

    /*
    for (int j = 0; j < out_features; j += 1) {
      volatile int8_t *imc_ptr = (volatile int8_t *)IMC_AMEM_0; // using volatile to prevent the compiler from optimizing per-byte accessing into bulk loads 

      y_data[ + j] = imc_ptr[ (amem_write_base+ i*output_div_q)*IMC_DIM + j];
    }
    */

    
    
    x_batch_data += in_features;
    y_batch_data += out_features;
  }
}

/**
 * nn_conv2d_layer1_i6
 *
 * @brief Convlutional neural network first layer.
 *
 * y[i][j] = x[i][k] * weight[j][k] + bias[j]
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 * @param stride The stride of the convolution (1,1).
 * @param padding The padding of the convolution (1,1).
 * @param dilation The dilation of the convolution (1,1).   
 * @param groups The number of groups in the convolution (1).
 */
void nn_conv2d_layer1_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, const Tensor4D_I6 *weight, const Tensor1D_I6 *bias, char * dmem_weight_address, char * dmem_input_address, int sp_w_write_index, int amem_write_index, int amem_read_index, int next_conv_channel, int amem_scale, int bias_scale, int stride, int padding, int dilation, int groups) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");

  volatile int batch_size = x->shape[0];
  volatile int in_features = (x->shape[1] * x->shape[2] * x->shape[3]);
  volatile int out_features = weight->shape[0];

  int8_t *x_batch_data = x->data;
  int8_t *y_batch_data = y->data;
  

  //volatile int input_div_q = (weight->shape[1] * weight->shape[2] * weight->shape[3]) / IMC_DIM;
  //volatile int input_div_r = (weight->shape[1] * weight->shape[2] * weight->shape[3]) % IMC_DIM;
  //input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  volatile int output_div_q = (y->shape[1] / IMC_DIM);
  volatile int output_div_r = (y->shape[1] % IMC_DIM);
  output_div_q = ((output_div_r > 0) ? output_div_q + 1 : output_div_q);

  volatile int amem_div_q = (x->shape[1] / IMC_DIM);
  volatile int input_div_r = (x->shape[1] % IMC_DIM);
  amem_div_q = (input_div_r > 0) ? amem_div_q + 1 : amem_div_q;
  
  volatile int sp_weight_write_base=sp_w_write_index; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index; //16
  volatile int amem_read_base=amem_read_index; //0

  //printf("output_div_q=%d, input_div_q=%d, amem_div_q=%d\n", output_div_q, input_div_q, amem_div_q);

  // SP_W layout for first conv layer: 7-row * 256-col
  //   |    0-63   |   64-127  |  128-191  |  192-255  |
  // 0 |   H0W0C0  |   H0W0C1  |   H0W0C2  |   H0W1C0  |
  // 1 |   H0W1C1  |   H0W1C2  |   H0W2C0  |   H0W2C1  |
  //                          ...
  // 6 |   H2W2C0  |   H2W2C1  |   H2W2C2  |    BIAS   |

  
  /*write_reg(DMA_BURST_NUM, 143);
  
  for (volatile int i = 0; i < (output_div_q); i += 1) {
    write_reg(DMA_ROWIDX, sp_weight_write_base);
    write_reg(DRAM2SP_W_BURST, ((uintptr_t)dmem_weight_address));
    asm volatile("wfi");
  }*/

  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address);
  write_reg(DMA_READ_INPUT_BASE, (uintptr_t)dmem_input_address);

  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  write_reg(INPUT_ITR, amem_div_q);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);
  write_reg(AMEM_SCALE_CFG, amem_scale);
  write_reg(BIAS_SCALE_CFG, bias_scale);


  if (bias) {
    write_reg(BIAS_EXIST, 1);
  }


  for (volatile int i = 0; i < batch_size; i += 1) {
    int8_t *x_data = x_batch_data;
    int8_t *y_data = y_batch_data;

    write_reg(DMA_ROWIDX, amem_write_base+i*output_div_q* y->shape[2] * y->shape[3]);
    write_reg(AMEM_RD_BASE, amem_read_base+i*amem_div_q* x->shape[2] * x->shape[3]);

    write_reg(CONV_LAYER_A_CONV1, 1);
    //write_reg(DMA_ROWIDX, 0);

    asm volatile("wfi");
    
    
    x_batch_data += in_features;
    y_batch_data += out_features;
  }
}


/**
 * nn_relu4d_i6
 *
 * @brief Applies the ReLU activation function to a 4D tensor.
 *
 * y[i][j] = max(x[i][j], 0)
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_relu4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x,  int amem_write_index) {
  //nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1] && x->shape[2] == y->shape[2] && x->shape[3] == y->shape[3], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=0;
  // if input channlel > IMC_DIM but not multiple of IMC_DIM, this cause error. Refer to pooling.
  volatile int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;
  volatile int amem_div_r =((x->shape[1] % IMC_DIM) > 0) ? (IMC_DIM/x->shape[1]) : 1;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, (y->shape[3]/amem_div_r));
  write_reg(OUTPUT_ROW, y->shape[2]);

  write_reg(DMA_ROWIDX, amem_write_base);
  //write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(ReLU_LAYER, 1);
}

/**
 * nn_add4d_i6
 *
 * @brief Applies the Add function (shortcut) to 2 4D tensors.
 *
 * y[i][j] = x1[i][j]+ x2[i][j]
 *
 * @param y The result tensor.
 * @param x1 The input tensor.
 * @param x2 The input tensor.
 */
void nn_add4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x1, const Tensor4D_I6 *x2, int amem_write_index, int amem_read_index) {
  //nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1] && x->shape[2] == y->shape[2] && x->shape[3] == y->shape[3], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x1_data = x1->data;
  int8_t *y_data = y->data;

  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=amem_read_index;

  volatile int amem_div_q = y->shape[1] / IMC_DIM;
  amem_div_q = ((y->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;
  volatile int amem_div_r =((y->shape[1] % IMC_DIM) > 0) ? (IMC_DIM/y->shape[1]) : 1;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, (y->shape[3]/amem_div_r));
  write_reg(OUTPUT_ROW, y->shape[2]);

  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(ADD_LAYER, 1);
}

/**
 * nn_layer_fused_i6
 *
 * @brief Fused conv+max.
 *
 * y[i][j] = x[i][k] * weight[j][k] + bias[j]
 *
 * @param y The result tensor.
 * @param x The input tensor.
 * @param weight The weight tensor.
 * @param bias The bias tensor.
 * @param stride The stride of the convolution (1,1).
 * @param padding The padding of the convolution (1,1).
 * @param dilation The dilation of the convolution (1,1).   
 * @param groups The number of groups in the convolution (1).
 *  volatile is for hardware interaction (Memory-Mapped I/O), used to tell the compiler that a variable's value can be modified by factors outside the program's normal control flow
 */
void nn_layer_fused_i6(Tensor4D_I6 *y0, const Tensor4D_I6 *x0, const Tensor4D_I6 *weight0, const Tensor1D_I6 *bias0, char * dmem_weight_address0, char * dmem_input_address, int sp_w_write_index0, int amem_write_index0, int amem_read_index0, int next_conv_channel0, int amem_scale0, int bias_scale0, int stride0, int padding0, int dilation0, int groups0,
                      Tensor4D_I6 *y1, const Tensor4D_I6 *x1, int amem_write_index1, int amem_read_index1,
                      Tensor4D_I6 *y2, const Tensor4D_I6 *x2, const Tensor4D_I6 *weight2, const Tensor1D_I6 *bias2, char * dmem_weight_address2, int sp_w_write_index2, int amem_write_index2, int amem_read_index2, int next_conv_channel2, int amem_scale2, int bias_scale2, int stride2, int padding2, int dilation2, int groups2,
                      Tensor4D_I6 *y3, const Tensor4D_I6 *x3, int amem_write_index3, int amem_read_index3,
                      Tensor4D_I6 *y4, const Tensor4D_I6 *x4, const Tensor4D_I6 *weight4, const Tensor1D_I6 *bias4, char * dmem_weight_address4, int sp_w_write_index4, int amem_write_index4, int amem_read_index4, int next_conv_channel4, int amem_scale4, int bias_scale4, int stride4, int padding4, int dilation4, int groups4,
                      Tensor4D_I6 *y5, const Tensor4D_I6 *x5, const Tensor4D_I6 *weight5, const Tensor1D_I6 *bias5, char * dmem_weight_address5, int sp_w_write_index5, int amem_write_index5, int amem_read_index5, int next_conv_channel5, int amem_scale5, int bias_scale5, int stride5, int padding5, int dilation5, int groups5,
                      Tensor4D_I6 *y6, const Tensor4D_I6 *x6, const Tensor4D_I6 *weight6, const Tensor1D_I6 *bias6, char * dmem_weight_address6, int sp_w_write_index6, int amem_write_index6, int amem_read_index6, int next_conv_channel6, int amem_scale6, int bias_scale6, int stride6, int padding6, int dilation6, int groups6,
                      Tensor4D_I6 *y7, const Tensor4D_I6 *x7, int amem_write_index7, int amem_read_index7) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");


  volatile int input_div_q[NUM_FUSED_LAYERS];
  volatile int output_div_q[NUM_FUSED_LAYERS];
  volatile int output_col[NUM_FUSED_LAYERS];
  volatile int output_row[NUM_FUSED_LAYERS];
  volatile int weight_col[NUM_FUSED_LAYERS];
  volatile int weight_row[NUM_FUSED_LAYERS];
  volatile int input_div_r[NUM_FUSED_LAYERS];
  volatile int output_div_r[NUM_FUSED_LAYERS];
  volatile int padding[NUM_FUSED_LAYERS];
  volatile int stride[NUM_FUSED_LAYERS];
  volatile int amem_scale[NUM_FUSED_LAYERS];
  volatile int bias_scale[NUM_FUSED_LAYERS];
  
  output_div_q[0] = (y0->shape[1] / IMC_DIM);
  output_div_r[0] = (y0->shape[1] % IMC_DIM);
  output_div_q[0] = ((output_div_r[0] > 0) ? output_div_q[0] + 1 : output_div_q[0]);
  input_div_q[0] = (x0->shape[1] / IMC_DIM);
  input_div_r[0] = (x0->shape[1] % IMC_DIM);
  input_div_q[0] = (input_div_r[0] > 0) ? input_div_q[0] + 1 : input_div_q[0];
  output_row[0] = y0->shape[2];
  output_col[0] = y0->shape[3];
  weight_row[0] = weight0->shape[2];
  weight_col[0] = weight0->shape[3];
  padding[0] = padding0;
  stride[0] = stride0;
  amem_scale[0] = amem_scale0;
  bias_scale[0] = bias_scale0;

  output_div_q[1] = (y1->shape[1] / IMC_DIM);
  output_div_r[1] = (y1->shape[1] % IMC_DIM);
  output_div_q[1] = ((output_div_r[1] > 0) ? output_div_q[1] + 1 : output_div_q[1]);
  input_div_q[1] = (x1->shape[1] / IMC_DIM);
  input_div_r[1] = (x1->shape[1] % IMC_DIM);
  input_div_q[1] = (input_div_r[1] > 0) ? input_div_q[1] + 1 : input_div_q[1];
  output_row[1] = y1->shape[2];
  output_col[1] = y1->shape[3];
  weight_row[1] = 2;
  weight_col[1] = 2;
  padding[1] = 0;
  stride[1] = 2;

  output_div_q[2] = (y2->shape[1] / IMC_DIM);
  output_div_r[2] = (y2->shape[1] % IMC_DIM);
  output_div_q[2] = ((output_div_r[2] > 0) ? output_div_q[2] + 1 : output_div_q[2]);
  input_div_q[2] = (x2->shape[1] / IMC_DIM);
  input_div_r[2] = (x2->shape[1] % IMC_DIM);
  input_div_q[2] = (input_div_r[2] > 0) ? input_div_q[2] + 1 : input_div_q[2];
  output_row[2] = y2->shape[2];
  output_col[2] = y2->shape[3];
  weight_row[2] = weight2->shape[2];
  weight_col[2] = weight2->shape[3];
  padding[2] = padding2;
  stride[2] = stride2;
  amem_scale[2] = amem_scale2;
  bias_scale[2] = bias_scale2;

  output_div_q[3] = (y3->shape[1] / IMC_DIM);
  output_div_r[3] = (y3->shape[1] % IMC_DIM);
  output_div_q[3] = ((output_div_r[3] > 0) ? output_div_q[3] + 1 : output_div_q[3]);
  input_div_q[3] = (x3->shape[1] / IMC_DIM);
  input_div_r[3] = (x3->shape[1] % IMC_DIM);
  input_div_q[3] = (input_div_r[3] > 0) ? input_div_q[3] + 1 : input_div_q[3];
  output_row[3] = y3->shape[2];
  output_col[3] = y3->shape[3];
  weight_row[3] = 2;
  weight_col[3] = 2;
  padding[3] = 0;
  stride[3] = 2;

  output_div_q[4] = (y4->shape[1] / IMC_DIM);
  output_div_r[4] = (y4->shape[1] % IMC_DIM);
  output_div_q[4] = ((output_div_r[4] > 0) ? output_div_q[4] + 1 : output_div_q[4]);
  input_div_q[4] = (x4->shape[1] / IMC_DIM);
  input_div_r[4] = (x4->shape[1] % IMC_DIM);
  input_div_q[4] = (input_div_r[4] > 0) ? input_div_q[4] + 1 : input_div_q[4];
  output_row[4] = y4->shape[2];
  output_col[4] = y4->shape[3];
  weight_row[4] = weight4->shape[2];
  weight_col[4] = weight4->shape[3];
  padding[4] = padding4;
  stride[4] = stride4;
  amem_scale[4] = amem_scale4;
  bias_scale[4] = bias_scale4;

  output_div_q[5] = (y5->shape[1] / IMC_DIM);
  output_div_r[5] = (y5->shape[1] % IMC_DIM);
  output_div_q[5] = ((output_div_r[5] > 0) ? output_div_q[5] + 1 : output_div_q[5]);
  input_div_q[5] = (x5->shape[1] / IMC_DIM);
  input_div_r[5] = (x5->shape[1] % IMC_DIM);
  input_div_q[5] = (input_div_r[5] > 0) ? input_div_q[5] + 1 : input_div_q[5];
  output_row[5] = y5->shape[2];
  output_col[5] = y5->shape[3];
  weight_row[5] = weight5->shape[2];
  weight_col[5] = weight5->shape[3];
  padding[5] = padding5;
  stride[5] = stride5;
  amem_scale[5] = amem_scale5;
  bias_scale[5] = bias_scale5;

  output_div_q[6] = (y6->shape[1] / IMC_DIM);
  output_div_r[6] = (y6->shape[1] % IMC_DIM);
  output_div_q[6] = ((output_div_r[6] > 0) ? output_div_q[6] + 1 : output_div_q[6]);
  input_div_q[6] = (x6->shape[1] / IMC_DIM);
  input_div_r[6] = (x6->shape[1] % IMC_DIM);
  input_div_q[6] = (input_div_r[6] > 0) ? input_div_q[6] + 1 : input_div_q[6];
  output_row[6] = y6->shape[2];
  output_col[6] = y6->shape[3];
  weight_row[6] = weight6->shape[2];
  weight_col[6] = weight6->shape[3];
  padding[6] = padding6;
  stride[6] = stride6;
  amem_scale[6] = amem_scale6;
  bias_scale[6] = bias_scale6;

  output_div_q[7] = (y7->shape[1] / IMC_DIM);
  output_div_r[7] = (y7->shape[1] % IMC_DIM);
  output_div_q[7] = ((output_div_r[7] > 0) ? output_div_q[7] + 1 : output_div_q[7]);
  input_div_q[7] = (x7->shape[1] / IMC_DIM);
  input_div_r[7] = (x7->shape[1] % IMC_DIM);
  input_div_q[7] = (input_div_r[7] > 0) ? input_div_q[7] + 1 : input_div_q[7];
  output_row[7] = y7->shape[2];
  output_col[7] = y7->shape[3];
  weight_row[7] = 2;
  weight_col[7] = 2;
  padding[7] = 0;
  stride[7] = 2;
  
  volatile int sp_weight_write_base=sp_w_write_index0; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index0; //16
  volatile int amem_read_base=amem_read_index0; //0

  volatile int i=0;
  volatile int j=0;

  write_reg(LAYER_FUSE_ENABLE, 1);
  for ( i = 0; i < NUM_FUSED_LAYERS; i += 1) {
    write_reg(LAYER_INDEX, i);
    write_reg(INPUT_ITR, input_div_q[i]);
    write_reg(OUTPUT_ITR, output_div_q[i]);
    write_reg(OUTPUT_COL, output_col[i]);
    write_reg(OUTPUT_ROW, output_row[i]);
    write_reg(WEIGHT_COL, weight_col[i]);
    write_reg(WEIGHT_ROW, weight_row[i]);
    write_reg(INPUT_REMAINDER, input_div_r[i]);
    write_reg(OUTPUT_REMAINDER, output_div_r[i]);
    write_reg(PADDING, padding[i]);
    write_reg(STRIDE, stride[i]);
    write_reg(AMEM_SCALE_CFG, amem_scale[i]);
    write_reg(BIAS_SCALE_CFG, bias_scale[i]);
    if (i==1||i==3||i==7){
      write_reg(LAYER_TYPE, 1);
    }
    else{
      write_reg(LAYER_TYPE, 0);
    }
  }


  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address0);
  write_reg(DMA_READ_INPUT_BASE, (uintptr_t)dmem_input_address);
  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  /*write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);

  write_reg(PADDING, padding);
  write_reg(STRIDE, stride);*/


  if (bias0) {
    write_reg(BIAS_EXIST, 1);
  }



  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base); // remove batch supporting can save RSICV cycle?

  //for resnet9
  //write_reg(CONV_LAYER_A_OTHER, 1);
  
  write_reg(LAYER_FUSED, 1);

  asm volatile("wfi");

    
    

}

void nn_layer_fused_fully_i6(Tensor4D_I6 *y0, const Tensor4D_I6 *x0, const Tensor4D_I6 *weight0, const Tensor1D_I6 *bias0, char * dmem_weight_address0, char * dmem_input_address, int sp_w_write_index0, int amem_write_index0, int amem_read_index0, int next_conv_channel0, int amem_scale0, int bias_scale0, int stride0, int padding0, int dilation0, int groups0,
                      Tensor4D_I6 *y1, const Tensor4D_I6 *x1, int amem_write_index1, int amem_read_index1,
                      Tensor4D_I6 *y2, const Tensor4D_I6 *x2, const Tensor4D_I6 *weight2, const Tensor1D_I6 *bias2, char * dmem_weight_address2, int sp_w_write_index2, int amem_write_index2, int amem_read_index2, int next_conv_channel2, int amem_scale2, int bias_scale2, int stride2, int padding2, int dilation2, int groups2,
                      Tensor4D_I6 *y3, const Tensor4D_I6 *x3, int amem_write_index3, int amem_read_index3,
                      Tensor4D_I6 *y4, const Tensor4D_I6 *x4, const Tensor4D_I6 *weight4, const Tensor1D_I6 *bias4, char * dmem_weight_address4, int sp_w_write_index4, int amem_write_index4, int amem_read_index4, int next_conv_channel4, int amem_scale4, int bias_scale4, int stride4, int padding4, int dilation4, int groups4,
                      Tensor4D_I6 *y5, const Tensor4D_I6 *x5, const Tensor4D_I6 *weight5, const Tensor1D_I6 *bias5, char * dmem_weight_address5, int sp_w_write_index5, int amem_write_index5, int amem_read_index5, int next_conv_channel5, int amem_scale5, int bias_scale5, int stride5, int padding5, int dilation5, int groups5,
                      Tensor4D_I6 *y6, const Tensor4D_I6 *x6, const Tensor4D_I6 *weight6, const Tensor1D_I6 *bias6, char * dmem_weight_address6, int sp_w_write_index6, int amem_write_index6, int amem_read_index6, int next_conv_channel6, int amem_scale6, int bias_scale6, int stride6, int padding6, int dilation6, int groups6,
                      Tensor4D_I6 *y7, const Tensor4D_I6 *x7, int amem_write_index7, int amem_read_index7,
                      Tensor4D_I6 *y8, const Tensor4D_I6 *x8, const Tensor4D_I6 *weight8, const Tensor1D_I6 *bias8, char * dmem_weight_address8, int sp_w_write_index8, int amem_write_index8, int amem_read_index8, int next_conv_channel8, int amem_scale8, int bias_scale8, int stride8, int padding8, int dilation8, int groups8,
                      Tensor4D_I6 *y9, const Tensor4D_I6 *x9, const Tensor4D_I6 *weight9, const Tensor1D_I6 *bias9, char * dmem_weight_address9, int sp_w_write_index9, int amem_write_index9, int amem_read_index9, int next_conv_channel9, int amem_scale9, int bias_scale9, int stride9, int padding9, int dilation9, int groups9,
                      Tensor4D_I6 *y10, const Tensor4D_I6 *x10, const Tensor4D_I6 *weight10, const Tensor1D_I6 *bias10, char * dmem_weight_address10, int sp_w_write_index10, int amem_write_index10, int amem_read_index10, int next_conv_channel10, int amem_scale10, int bias_scale10, int stride10, int padding10, int dilation10, int groups10,
                      Tensor4D_I6 *y11, const Tensor4D_I6 *x11, int amem_write_index11, int amem_read_index11,
                      Tensor4D_I6 *y12, const Tensor4D_I6 *x12, const Tensor4D_I6 *weight12, const Tensor1D_I6 *bias12, char * dmem_weight_address12, int sp_w_write_index12, int amem_write_index12, int amem_read_index12, int next_conv_channel12, int amem_scale12, int bias_scale12, int stride12, int padding12, int dilation12, int groups12,
                      Tensor4D_I6 *y13, const Tensor4D_I6 *x13, const Tensor4D_I6 *weight13, const Tensor1D_I6 *bias13, char * dmem_weight_address13, int sp_w_write_index13, int amem_write_index13, int amem_read_index13, int next_conv_channel13, int amem_scale13, int bias_scale13, int stride13, int padding13, int dilation13, int groups13,
                      Tensor4D_I6 *y14, const Tensor4D_I6 *x14, const Tensor4D_I6 *weight14, const Tensor1D_I6 *bias14, char * dmem_weight_address14, int sp_w_write_index14, int amem_write_index14, int amem_read_index14, int next_conv_channel14, int amem_scale14, int bias_scale14, int stride14, int padding14, int dilation14, int groups14,
                      Tensor4D_I6 *y15, const Tensor4D_I6 *x15, const Tensor4D_I6 *weight15, const Tensor1D_I6 *bias15, char * dmem_weight_address15, int sp_w_write_index15, int amem_write_index15, int amem_read_index15, int next_conv_channel15, int amem_scale15, int bias_scale15, int stride15, int padding15, int dilation15, int groups15,
                      Tensor4D_I6 *y16, const Tensor4D_I6 *x16, const Tensor4D_I6 *weight16, const Tensor1D_I6 *bias16, char * dmem_weight_address16, int sp_w_write_index16, int amem_write_index16, int amem_read_index16, int next_conv_channel16, int amem_scale16, int bias_scale16, int stride16, int padding16, int dilation16, int groups16
                      ) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");


  volatile int input_div_q[NUM_FUSED_LAYERS];
  volatile int output_div_q[NUM_FUSED_LAYERS];
  volatile int output_col[NUM_FUSED_LAYERS];
  volatile int output_row[NUM_FUSED_LAYERS];
  volatile int weight_col[NUM_FUSED_LAYERS];
  volatile int weight_row[NUM_FUSED_LAYERS];
  volatile int input_div_r[NUM_FUSED_LAYERS];
  volatile int output_div_r[NUM_FUSED_LAYERS];
  volatile int padding[NUM_FUSED_LAYERS];
  volatile int stride[NUM_FUSED_LAYERS];
  volatile int amem_scale[NUM_FUSED_LAYERS];
  volatile int bias_scale[NUM_FUSED_LAYERS];
  
  output_div_q[0] = (y0->shape[1] / IMC_DIM);
  output_div_r[0] = (y0->shape[1] % IMC_DIM);
  output_div_q[0] = ((output_div_r[0] > 0) ? output_div_q[0] + 1 : output_div_q[0]);
  input_div_q[0] = (x0->shape[1] / IMC_DIM);
  input_div_r[0] = (x0->shape[1] % IMC_DIM);
  input_div_q[0] = (input_div_r[0] > 0) ? input_div_q[0] + 1 : input_div_q[0];
  output_row[0] = y0->shape[2];
  output_col[0] = y0->shape[3];
  weight_row[0] = weight0->shape[2];
  weight_col[0] = weight0->shape[3];
  padding[0] = padding0;
  stride[0] = stride0;
  amem_scale[0] = amem_scale0;
  bias_scale[0] = bias_scale0;

  output_div_q[1] = (y1->shape[1] / IMC_DIM);
  output_div_r[1] = (y1->shape[1] % IMC_DIM);
  output_div_q[1] = ((output_div_r[1] > 0) ? output_div_q[1] + 1 : output_div_q[1]);
  input_div_q[1] = (x1->shape[1] / IMC_DIM);
  input_div_r[1] = (x1->shape[1] % IMC_DIM);
  input_div_q[1] = (input_div_r[1] > 0) ? input_div_q[1] + 1 : input_div_q[1];
  output_row[1] = y1->shape[2];
  output_col[1] = y1->shape[3];
  weight_row[1] = 2;
  weight_col[1] = 2;
  padding[1] = 0;
  stride[1] = 2;

  output_div_q[2] = (y2->shape[1] / IMC_DIM);
  output_div_r[2] = (y2->shape[1] % IMC_DIM);
  output_div_q[2] = ((output_div_r[2] > 0) ? output_div_q[2] + 1 : output_div_q[2]);
  input_div_q[2] = (x2->shape[1] / IMC_DIM);
  input_div_r[2] = (x2->shape[1] % IMC_DIM);
  input_div_q[2] = (input_div_r[2] > 0) ? input_div_q[2] + 1 : input_div_q[2];
  output_row[2] = y2->shape[2];
  output_col[2] = y2->shape[3];
  weight_row[2] = weight2->shape[2];
  weight_col[2] = weight2->shape[3];
  padding[2] = padding2;
  stride[2] = stride2;
  amem_scale[2] = amem_scale2;
  bias_scale[2] = bias_scale2;

  output_div_q[3] = (y3->shape[1] / IMC_DIM);
  output_div_r[3] = (y3->shape[1] % IMC_DIM);
  output_div_q[3] = ((output_div_r[3] > 0) ? output_div_q[3] + 1 : output_div_q[3]);
  input_div_q[3] = (x3->shape[1] / IMC_DIM);
  input_div_r[3] = (x3->shape[1] % IMC_DIM);
  input_div_q[3] = (input_div_r[3] > 0) ? input_div_q[3] + 1 : input_div_q[3];
  output_row[3] = y3->shape[2];
  output_col[3] = y3->shape[3];
  weight_row[3] = 2;
  weight_col[3] = 2;
  padding[3] = 0;
  stride[3] = 2;

  output_div_q[4] = (y4->shape[1] / IMC_DIM);
  output_div_r[4] = (y4->shape[1] % IMC_DIM);
  output_div_q[4] = ((output_div_r[4] > 0) ? output_div_q[4] + 1 : output_div_q[4]);
  input_div_q[4] = (x4->shape[1] / IMC_DIM);
  input_div_r[4] = (x4->shape[1] % IMC_DIM);
  input_div_q[4] = (input_div_r[4] > 0) ? input_div_q[4] + 1 : input_div_q[4];
  output_row[4] = y4->shape[2];
  output_col[4] = y4->shape[3];
  weight_row[4] = weight4->shape[2];
  weight_col[4] = weight4->shape[3];
  padding[4] = padding4;
  stride[4] = stride4;
  amem_scale[4] = amem_scale4;
  bias_scale[4] = bias_scale4;

  output_div_q[5] = (y5->shape[1] / IMC_DIM);
  output_div_r[5] = (y5->shape[1] % IMC_DIM);
  output_div_q[5] = ((output_div_r[5] > 0) ? output_div_q[5] + 1 : output_div_q[5]);
  input_div_q[5] = (x5->shape[1] / IMC_DIM);
  input_div_r[5] = (x5->shape[1] % IMC_DIM);
  input_div_q[5] = (input_div_r[5] > 0) ? input_div_q[5] + 1 : input_div_q[5];
  output_row[5] = y5->shape[2];
  output_col[5] = y5->shape[3];
  weight_row[5] = weight5->shape[2];
  weight_col[5] = weight5->shape[3];
  padding[5] = padding5;
  stride[5] = stride5;
  amem_scale[5] = amem_scale5;
  bias_scale[5] = bias_scale5;

  output_div_q[6] = (y6->shape[1] / IMC_DIM);
  output_div_r[6] = (y6->shape[1] % IMC_DIM);
  output_div_q[6] = ((output_div_r[6] > 0) ? output_div_q[6] + 1 : output_div_q[6]);
  input_div_q[6] = (x6->shape[1] / IMC_DIM);
  input_div_r[6] = (x6->shape[1] % IMC_DIM);
  input_div_q[6] = (input_div_r[6] > 0) ? input_div_q[6] + 1 : input_div_q[6];
  output_row[6] = y6->shape[2];
  output_col[6] = y6->shape[3];
  weight_row[6] = weight6->shape[2];
  weight_col[6] = weight6->shape[3];
  padding[6] = padding6;
  stride[6] = stride6;
  amem_scale[6] = amem_scale6;
  bias_scale[6] = bias_scale6;

  output_div_q[7] = (y7->shape[1] / IMC_DIM);
  output_div_r[7] = (y7->shape[1] % IMC_DIM);
  output_div_q[7] = ((output_div_r[7] > 0) ? output_div_q[7] + 1 : output_div_q[7]);
  input_div_q[7] = (x7->shape[1] / IMC_DIM);
  input_div_r[7] = (x7->shape[1] % IMC_DIM);
  input_div_q[7] = (input_div_r[7] > 0) ? input_div_q[7] + 1 : input_div_q[7];
  output_row[7] = y7->shape[2];
  output_col[7] = y7->shape[3];
  weight_row[7] = 2;
  weight_col[7] = 2;
  padding[7] = 0;
  stride[7] = 2;

  output_div_q[8] = (y8->shape[1] / IMC_DIM);
  output_div_r[8] = (y8->shape[1] % IMC_DIM);
  output_div_q[8] = ((output_div_r[8] > 0) ? output_div_q[8] + 1 : output_div_q[8]);
  input_div_q[8] = (x8->shape[1] / IMC_DIM);
  input_div_r[8] = (x8->shape[1] % IMC_DIM);
  input_div_q[8] = (input_div_r[8] > 0) ? input_div_q[8] + 1 : input_div_q[8];
  output_row[8] = y8->shape[2];
  output_col[8] = y8->shape[3];
  weight_row[8] = weight8->shape[2];
  weight_col[8] = weight8->shape[3];
  padding[8] = padding8;
  stride[8] = stride8;
  amem_scale[8] = amem_scale8;
  bias_scale[8] = bias_scale8;

  output_div_q[9] = (y9->shape[1] / IMC_DIM);
  output_div_r[9] = (y9->shape[1] % IMC_DIM);
  output_div_q[9] = ((output_div_r[9] > 0) ? output_div_q[9] + 1 : output_div_q[9]);
  input_div_q[9] = (x9->shape[1] / IMC_DIM);
  input_div_r[9] = (x9->shape[1] % IMC_DIM);
  input_div_q[9] = (input_div_r[9] > 0) ? input_div_q[9] + 1 : input_div_q[9];
  output_row[9] = y9->shape[2];
  output_col[9] = y9->shape[3];
  weight_row[9] = weight9->shape[2];
  weight_col[9] = weight9->shape[3];
  padding[9] = padding9;
  stride[9] = stride9;
  amem_scale[9] = amem_scale9;
  bias_scale[9] = bias_scale9;

  output_div_q[10] = (y10->shape[1] / IMC_DIM);
  output_div_r[10] = (y10->shape[1] % IMC_DIM);
  output_div_q[10] = ((output_div_r[10] > 0) ? output_div_q[10] + 1 : output_div_q[10]);
  input_div_q[10] = (x10->shape[1] / IMC_DIM);
  input_div_r[10] = (x10->shape[1] % IMC_DIM);
  input_div_q[10] = (input_div_r[10] > 0) ? input_div_q[10] + 1 : input_div_q[10];
  output_row[10] = y10->shape[2];
  output_col[10] = y10->shape[3];
  weight_row[10] = weight10->shape[2];
  weight_col[10] = weight10->shape[3];
  padding[10] = padding10;
  stride[10] = stride10;
  amem_scale[10] = amem_scale10;
  bias_scale[10] = bias_scale10;

  output_div_q[11] = (y11->shape[1] / IMC_DIM);
  output_div_r[11] = (y11->shape[1] % IMC_DIM);
  output_div_q[11] = ((output_div_r[11] > 0) ? output_div_q[11] + 1 : output_div_q[11]);
  input_div_q[11] = (x11->shape[1] / IMC_DIM);
  input_div_r[11] = (x11->shape[1] % IMC_DIM);
  input_div_q[11] = (input_div_r[11] > 0) ? input_div_q[11] + 1 : input_div_q[11];
  output_row[11] = y11->shape[2];
  output_col[11] = y11->shape[3];
  weight_row[11] = 2;
  weight_col[11] = 2;
  padding[11] = 0;
  stride[11] = 2;

  output_div_q[12] = (y12->shape[1] / IMC_DIM);
  output_div_r[12] = (y12->shape[1] % IMC_DIM);
  output_div_q[12] = ((output_div_r[12] > 0) ? output_div_q[12] + 1 : output_div_q[12]);
  input_div_q[12] = (x12->shape[1] / IMC_DIM);
  input_div_r[12] = (x12->shape[1] % IMC_DIM);
  input_div_q[12] = (input_div_r[12] > 0) ? input_div_q[12] + 1 : input_div_q[12];
  output_row[12] = y12->shape[2];
  output_col[12] = y12->shape[3];
  weight_row[12] = weight12->shape[2];
  weight_col[12] = weight12->shape[3];
  padding[12] = padding12;
  stride[12] = stride12;
  amem_scale[12] = amem_scale12;
  bias_scale[12] = bias_scale12;

  output_div_q[13] = (y13->shape[1] / IMC_DIM);
  output_div_r[13] = (y13->shape[1] % IMC_DIM);
  output_div_q[13] = ((output_div_r[13] > 0) ? output_div_q[13] + 1 : output_div_q[13]);
  input_div_q[13] = (x13->shape[1] / IMC_DIM);
  input_div_r[13] = (x13->shape[1] % IMC_DIM);
  input_div_q[13] = (input_div_r[13] > 0) ? input_div_q[13] + 1 : input_div_q[13];
  output_row[13] = y13->shape[2];
  output_col[13] = y13->shape[3];
  weight_row[13] = weight13->shape[2];
  weight_col[13] = weight13->shape[3];
  padding[13] = padding13;
  stride[13] = stride13;
  amem_scale[13] = amem_scale13;
  bias_scale[13] = bias_scale13;

  output_div_q[14] = (y14->shape[1] / IMC_DIM);
  output_div_r[14] = (y14->shape[1] % IMC_DIM);
  output_div_q[14] = ((output_div_r[14] > 0) ? output_div_q[14] + 1 : output_div_q[14]);
  input_div_q[14] = (x14->shape[1] / IMC_DIM);
  input_div_r[14] = (x14->shape[1] % IMC_DIM);
  input_div_q[14] = (input_div_r[14] > 0) ? input_div_q[14] + 1 : input_div_q[14];
  output_row[14] = y14->shape[2];
  output_col[14] = y14->shape[3];
  weight_row[14] = weight14->shape[2];
  weight_col[14] = weight14->shape[3];
  padding[14] = padding14;
  stride[14] = stride14;
  amem_scale[14] = amem_scale14;
  bias_scale[14] = bias_scale14;

  output_div_q[15] = (y15->shape[1] / IMC_DIM);
  output_div_r[15] = (y15->shape[1] % IMC_DIM);
  output_div_q[15] = ((output_div_r[15] > 0) ? output_div_q[15] + 1 : output_div_q[15]);
  input_div_q[15] = (x15->shape[1] / IMC_DIM);
  input_div_r[15] = (x15->shape[1] % IMC_DIM);
  input_div_q[15] = (input_div_r[15] > 0) ? input_div_q[15] + 1 : input_div_q[15];
  output_row[15] = y15->shape[2];
  output_col[15] = y15->shape[3];
  weight_row[15] = weight15->shape[2];
  weight_col[15] = weight15->shape[3];
  padding[15] = padding15;
  stride[15] = stride15;
  amem_scale[15] = amem_scale15;
  bias_scale[15] = bias_scale15;

  output_div_q[16] = (y16->shape[1] / IMC_DIM);
  output_div_r[16] = (y16->shape[1] % IMC_DIM);
  output_div_q[16] = ((output_div_r[16] > 0) ? output_div_q[16] + 1 : output_div_q[16]);
  input_div_q[16] = (x16->shape[1] / IMC_DIM);
  input_div_r[16] = (x16->shape[1] % IMC_DIM);
  input_div_q[16] = (input_div_r[16] > 0) ? input_div_q[16] + 1 : input_div_q[16];
  output_row[16] = y16->shape[2];
  output_col[16] = y16->shape[3];
  weight_row[16] = weight16->shape[2];
  weight_col[16] = weight16->shape[3];
  padding[16] = padding16;
  stride[16] = stride16;
  amem_scale[16] = amem_scale16;
  bias_scale[16] = bias_scale16;

  
  volatile int sp_weight_write_base=sp_w_write_index0; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index0; //16
  volatile int amem_read_base=amem_read_index0; //0

  volatile int i=0;
  volatile int j=0;

  write_reg(LAYER_FUSE_ENABLE, 1);
  for ( i = 0; i < NUM_FUSED_LAYERS; i += 1) {
    write_reg(LAYER_INDEX, i);
    write_reg(INPUT_ITR, input_div_q[i]);
    write_reg(OUTPUT_ITR, output_div_q[i]);
    write_reg(OUTPUT_COL, output_col[i]);
    write_reg(OUTPUT_ROW, output_row[i]);
    write_reg(WEIGHT_COL, weight_col[i]);
    write_reg(WEIGHT_ROW, weight_row[i]);
    write_reg(INPUT_REMAINDER, input_div_r[i]);
    write_reg(OUTPUT_REMAINDER, output_div_r[i]);
    write_reg(PADDING, padding[i]);
    write_reg(STRIDE, stride[i]);
    write_reg(AMEM_SCALE_CFG, amem_scale[i]);
    write_reg(BIAS_SCALE_CFG, bias_scale[i]);
    if (i==1||i==3||i==7||i==11){
      write_reg(LAYER_TYPE, 1);
    }
    else{
      write_reg(LAYER_TYPE, 0);
    }
  }


  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address0);
  write_reg(DMA_READ_INPUT_BASE, (uintptr_t)dmem_input_address);
  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  /*write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);

  write_reg(PADDING, padding);
  write_reg(STRIDE, stride);*/


  if (bias0) {
    write_reg(BIAS_EXIST, 1);
  }



  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base); // remove batch supporting can save RSICV cycle?

  //for resnet9
  //write_reg(CONV_LAYER_A_OTHER, 1);
  
  write_reg(LAYER_FUSED, 1);

  asm volatile("wfi");

}

void nn_layer_fused_resnet_i6(Tensor4D_I6 *y0, const Tensor4D_I6 *x0, const Tensor4D_I6 *weight0, const Tensor1D_I6 *bias0, char * dmem_weight_address0, char * dmem_input_address, int sp_w_write_index0, int amem_write_index0, int amem_read_index0, int next_conv_channel0, int amem_scale0, int bias_scale0, int stride0, int padding0, int dilation0, int groups0,
                      Tensor4D_I6 *y1, const Tensor4D_I6 *x1, const Tensor4D_I6 *weight1, const Tensor1D_I6 *bias1, char * dmem_weight_address1, int sp_w_write_index1, int amem_write_index1, int amem_read_index1, int next_conv_channel1, int amem_scale1, int bias_scale1, int stride1, int padding1, int dilation1, int groups1,
                      Tensor4D_I6 *y2, const Tensor4D_I6 *x2, int amem_write_index2, int amem_read_index2,
                      Tensor4D_I6 *y3, const Tensor4D_I6 *x3, const Tensor4D_I6 *weight3, const Tensor1D_I6 *bias3, char * dmem_weight_address3, int sp_w_write_index3, int amem_write_index3, int amem_read_index3, int next_conv_channel3, int amem_scale3, int bias_scale3, int stride3, int padding3, int dilation3, int groups3,
                      Tensor4D_I6 *y4, const Tensor4D_I6 *x4, const Tensor4D_I6 *weight4, const Tensor1D_I6 *bias4, char * dmem_weight_address4, int sp_w_write_index4, int amem_write_index4, int amem_read_index4, int next_conv_channel4, int amem_scale4, int bias_scale4, int stride4, int padding4, int dilation4, int groups4,
                      Tensor4D_I6 *y5, const Tensor4D_I6 *x5, const Tensor4D_I6 *x5_2, int amem_write_index5, int amem_read_index5,
                      Tensor4D_I6 *y6, const Tensor4D_I6 *x6, const Tensor4D_I6 *weight6, const Tensor1D_I6 *bias6, char * dmem_weight_address6, int sp_w_write_index6, int amem_write_index6, int amem_read_index6, int next_conv_channel6, int amem_scale6, int bias_scale6, int stride6, int padding6, int dilation6, int groups6,
                      Tensor4D_I6 *y7, const Tensor4D_I6 *x7, int amem_write_index7, int amem_read_index7
                      ) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");


  volatile int input_div_q[NUM_FUSED_LAYERS];
  volatile int output_div_q[NUM_FUSED_LAYERS];
  volatile int output_col[NUM_FUSED_LAYERS];
  volatile int output_row[NUM_FUSED_LAYERS];
  volatile int weight_col[NUM_FUSED_LAYERS];
  volatile int weight_row[NUM_FUSED_LAYERS];
  volatile int input_div_r[NUM_FUSED_LAYERS];
  volatile int output_div_r[NUM_FUSED_LAYERS];
  volatile int padding[NUM_FUSED_LAYERS];
  volatile int stride[NUM_FUSED_LAYERS];
  volatile int amem_scale[NUM_FUSED_LAYERS];
  volatile int bias_scale[NUM_FUSED_LAYERS];
  
  output_div_q[0] = (y0->shape[1] / IMC_DIM);
  output_div_r[0] = (y0->shape[1] % IMC_DIM);
  output_div_q[0] = ((output_div_r[0] > 0) ? output_div_q[0] + 1 : output_div_q[0]);
  input_div_q[0] = (x0->shape[1] / IMC_DIM);
  input_div_r[0] = (x0->shape[1] % IMC_DIM);
  input_div_q[0] = (input_div_r[0] > 0) ? input_div_q[0] + 1 : input_div_q[0];
  output_row[0] = y0->shape[2];
  output_col[0] = y0->shape[3];
  weight_row[0] = weight0->shape[2];
  weight_col[0] = weight0->shape[3];
  padding[0] = padding0;
  stride[0] = stride0;
  amem_scale[0] = amem_scale0;
  bias_scale[0] = bias_scale0;

  output_div_q[1] = (y1->shape[1] / IMC_DIM);
  output_div_r[1] = (y1->shape[1] % IMC_DIM);
  output_div_q[1] = ((output_div_r[1] > 0) ? output_div_q[1] + 1 : output_div_q[1]);
  input_div_q[1] = (x1->shape[1] / IMC_DIM);
  input_div_r[1] = (x1->shape[1] % IMC_DIM);
  input_div_q[1] = (input_div_r[1] > 0) ? input_div_q[1] + 1 : input_div_q[1];
  output_row[1] = y1->shape[2];
  output_col[1] = y1->shape[3];
  weight_row[1] = weight1->shape[2];
  weight_col[1] = weight1->shape[3];
  padding[1] = padding1;
  stride[1] = stride1;
  amem_scale[1] = amem_scale1;
  bias_scale[1] = bias_scale1;

  output_div_q[2] = (y2->shape[1] / IMC_DIM);
  output_div_r[2] = (y2->shape[1] % IMC_DIM);
  output_div_q[2] = ((output_div_r[2] > 0) ? output_div_q[2] + 1 : output_div_q[2]);
  input_div_q[2] = (x2->shape[1] / IMC_DIM);
  input_div_r[2] = (x2->shape[1] % IMC_DIM);
  input_div_q[2] = (input_div_r[2] > 0) ? input_div_q[2] + 1 : input_div_q[2];
  output_row[2] = y2->shape[2];
  output_col[2] = y2->shape[3];
  weight_row[2] = 2;
  weight_col[2] = 2;
  padding[2] = 0;
  stride[2] = 2;

  output_div_q[3] = (y3->shape[1] / IMC_DIM);
  output_div_r[3] = (y3->shape[1] % IMC_DIM);
  output_div_q[3] = ((output_div_r[3] > 0) ? output_div_q[3] + 1 : output_div_q[3]);
  input_div_q[3] = (x3->shape[1] / IMC_DIM);
  input_div_r[3] = (x3->shape[1] % IMC_DIM);
  input_div_q[3] = (input_div_r[3] > 0) ? input_div_q[3] + 1 : input_div_q[3];
  output_row[3] = y3->shape[2];
  output_col[3] = y3->shape[3];
  weight_row[3] = weight3->shape[2];
  weight_col[3] = weight3->shape[3];
  padding[3] = padding3;
  stride[3] = stride3;
  amem_scale[3] = amem_scale3;
  bias_scale[3] = bias_scale3;

  output_div_q[4] = (y4->shape[1] / IMC_DIM);
  output_div_r[4] = (y4->shape[1] % IMC_DIM);
  output_div_q[4] = ((output_div_r[4] > 0) ? output_div_q[4] + 1 : output_div_q[4]);
  input_div_q[4] = (x4->shape[1] / IMC_DIM);
  input_div_r[4] = (x4->shape[1] % IMC_DIM);
  input_div_q[4] = (input_div_r[4] > 0) ? input_div_q[4] + 1 : input_div_q[4];
  output_row[4] = y4->shape[2];
  output_col[4] = y4->shape[3];
  weight_row[4] = weight4->shape[2];
  weight_col[4] = weight4->shape[3];
  padding[4] = padding4;
  stride[4] = stride4;
  amem_scale[4] = amem_scale4;
  bias_scale[4] = bias_scale4;

  output_div_q[5] = (y5->shape[1] / IMC_DIM);
  output_div_r[5] = (y5->shape[1] % IMC_DIM);
  output_div_q[5] = ((output_div_r[5] > 0) ? output_div_q[5] + 1 : output_div_q[5]);
  input_div_q[5] = (x5->shape[1] / IMC_DIM);
  input_div_r[5] = (x5->shape[1] % IMC_DIM);
  input_div_q[5] = (input_div_r[5] > 0) ? input_div_q[5] + 1 : input_div_q[5];
  output_row[5] = y5->shape[2];
  output_col[5] = y5->shape[3];
  weight_row[5] = 1;
  weight_col[5] = 1;
  padding[5] = 0;
  stride[5] = 1;

  output_div_q[6] = (y6->shape[1] / IMC_DIM);
  output_div_r[6] = (y6->shape[1] % IMC_DIM);
  output_div_q[6] = ((output_div_r[6] > 0) ? output_div_q[6] + 1 : output_div_q[6]);
  input_div_q[6] = (x6->shape[1] / IMC_DIM);
  input_div_r[6] = (x6->shape[1] % IMC_DIM);
  input_div_q[6] = (input_div_r[6] > 0) ? input_div_q[6] + 1 : input_div_q[6];
  output_row[6] = y6->shape[2];
  output_col[6] = y6->shape[3];
  weight_row[6] = weight6->shape[2];
  weight_col[6] = weight6->shape[3];
  padding[6] = padding6;
  stride[6] = stride6;
  amem_scale[6] = amem_scale6;
  bias_scale[6] = bias_scale6;

  output_div_q[7] = (y7->shape[1] / IMC_DIM);
  output_div_r[7] = (y7->shape[1] % IMC_DIM);
  output_div_q[7] = ((output_div_r[7] > 0) ? output_div_q[7] + 1 : output_div_q[7]);
  input_div_q[7] = (x7->shape[1] / IMC_DIM);
  input_div_r[7] = (x7->shape[1] % IMC_DIM);
  input_div_q[7] = (input_div_r[7] > 0) ? input_div_q[7] + 1 : input_div_q[7];
  output_row[7] = y7->shape[2];
  output_col[7] = y7->shape[3];
  weight_row[7] = 2;
  weight_col[7] = 2;
  padding[7] = 0;
  stride[7] = 2;

  
  volatile int sp_weight_write_base=sp_w_write_index0; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index0; //16
  volatile int amem_read_base=amem_read_index0; //0

  volatile int i=0;
  volatile int j=0;

  write_reg(LAYER_FUSE_ENABLE, 1);
  for ( i = 0; i < NUM_FUSED_LAYERS; i += 1) {
    write_reg(LAYER_INDEX, i);
    write_reg(INPUT_ITR, input_div_q[i]);
    write_reg(OUTPUT_ITR, output_div_q[i]);
    write_reg(OUTPUT_COL, output_col[i]);
    write_reg(OUTPUT_ROW, output_row[i]);
    write_reg(WEIGHT_COL, weight_col[i]);
    write_reg(WEIGHT_ROW, weight_row[i]);
    write_reg(INPUT_REMAINDER, input_div_r[i]);
    write_reg(OUTPUT_REMAINDER, output_div_r[i]);
    write_reg(PADDING, padding[i]);
    write_reg(STRIDE, stride[i]);
    write_reg(AMEM_SCALE_CFG, amem_scale[i]);
    write_reg(BIAS_SCALE_CFG, bias_scale[i]);
    if (i==2||i==7||i==9||i==13){
      write_reg(LAYER_TYPE, 1);
    }
    else if (i==5||i==12){
      write_reg(LAYER_TYPE, 2);
    } 
    else{
      write_reg(LAYER_TYPE, 0);
    }
  }


  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address0);
  write_reg(DMA_READ_INPUT_BASE, (uintptr_t)dmem_input_address);
  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  /*write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);

  write_reg(PADDING, padding);
  write_reg(STRIDE, stride);*/


  if (bias0) {
    write_reg(BIAS_EXIST, 1);
  }



  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base); // remove batch supporting can save RSICV cycle?

  //for resnet9
  //write_reg(CONV_LAYER_A_OTHER, 1);
  
  write_reg(LAYER_FUSED, 1);

  asm volatile("wfi");

}


void nn_layer_fused_fully_resnet_i6(Tensor4D_I6 *y0, const Tensor4D_I6 *x0, const Tensor4D_I6 *weight0, const Tensor1D_I6 *bias0, char * dmem_weight_address0, char * dmem_input_address, int sp_w_write_index0, int amem_write_index0, int amem_read_index0, int next_conv_channel0, int amem_scale0, int bias_scale0, int stride0, int padding0, int dilation0, int groups0,
                      Tensor4D_I6 *y1, const Tensor4D_I6 *x1, const Tensor4D_I6 *weight1, const Tensor1D_I6 *bias1, char * dmem_weight_address1, int sp_w_write_index1, int amem_write_index1, int amem_read_index1, int next_conv_channel1, int amem_scale1, int bias_scale1, int stride1, int padding1, int dilation1, int groups1,
                      Tensor4D_I6 *y2, const Tensor4D_I6 *x2, int amem_write_index2, int amem_read_index2,
                      Tensor4D_I6 *y3, const Tensor4D_I6 *x3, const Tensor4D_I6 *weight3, const Tensor1D_I6 *bias3, char * dmem_weight_address3, int sp_w_write_index3, int amem_write_index3, int amem_read_index3, int next_conv_channel3, int amem_scale3, int bias_scale3, int stride3, int padding3, int dilation3, int groups3,
                      Tensor4D_I6 *y4, const Tensor4D_I6 *x4, const Tensor4D_I6 *weight4, const Tensor1D_I6 *bias4, char * dmem_weight_address4, int sp_w_write_index4, int amem_write_index4, int amem_read_index4, int next_conv_channel4, int amem_scale4, int bias_scale4, int stride4, int padding4, int dilation4, int groups4,
                      Tensor4D_I6 *y5, const Tensor4D_I6 *x5, const Tensor4D_I6 *x5_2, int amem_write_index5, int amem_read_index5,
                      Tensor4D_I6 *y6, const Tensor4D_I6 *x6, const Tensor4D_I6 *weight6, const Tensor1D_I6 *bias6, char * dmem_weight_address6, int sp_w_write_index6, int amem_write_index6, int amem_read_index6, int next_conv_channel6, int amem_scale6, int bias_scale6, int stride6, int padding6, int dilation6, int groups6,
                      Tensor4D_I6 *y7, const Tensor4D_I6 *x7, int amem_write_index7, int amem_read_index7,
                      Tensor4D_I6 *y8, const Tensor4D_I6 *x8, const Tensor4D_I6 *weight8, const Tensor1D_I6 *bias8, char * dmem_weight_address8, int sp_w_write_index8, int amem_write_index8, int amem_read_index8, int next_conv_channel8, int amem_scale8, int bias_scale8, int stride8, int padding8, int dilation8, int groups8,
                      Tensor4D_I6 *y9, const Tensor4D_I6 *x9, int amem_write_index9, int amem_read_index9,
                      Tensor4D_I6 *y10, const Tensor4D_I6 *x10, const Tensor4D_I6 *weight10, const Tensor1D_I6 *bias10, char * dmem_weight_address10, int sp_w_write_index10, int amem_write_index10, int amem_read_index10, int next_conv_channel10, int amem_scale10, int bias_scale10, int stride10, int padding10, int dilation10, int groups10,
                      Tensor4D_I6 *y11, const Tensor4D_I6 *x11, const Tensor4D_I6 *weight11, const Tensor1D_I6 *bias11, char * dmem_weight_address11, int sp_w_write_index11, int amem_write_index11, int amem_read_index11, int next_conv_channel11, int amem_scale11, int bias_scale11, int stride11, int padding11, int dilation11, int groups11,
                      Tensor4D_I6 *y12, const Tensor4D_I6 *x12, const Tensor4D_I6 *x12_2, int amem_write_index12, int amem_read_index12,
                      Tensor4D_I6 *y13, const Tensor4D_I6 *x13, int amem_write_index13, int amem_read_index13
                      ) {
  //nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  //nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");


  volatile int input_div_q[NUM_FUSED_LAYERS];
  volatile int output_div_q[NUM_FUSED_LAYERS];
  volatile int output_col[NUM_FUSED_LAYERS];
  volatile int output_row[NUM_FUSED_LAYERS];
  volatile int weight_col[NUM_FUSED_LAYERS];
  volatile int weight_row[NUM_FUSED_LAYERS];
  volatile int input_div_r[NUM_FUSED_LAYERS];
  volatile int output_div_r[NUM_FUSED_LAYERS];
  volatile int padding[NUM_FUSED_LAYERS];
  volatile int stride[NUM_FUSED_LAYERS];
  volatile int amem_scale[NUM_FUSED_LAYERS];
  volatile int bias_scale[NUM_FUSED_LAYERS];
  
  output_div_q[0] = (y0->shape[1] / IMC_DIM);
  output_div_r[0] = (y0->shape[1] % IMC_DIM);
  output_div_q[0] = ((output_div_r[0] > 0) ? output_div_q[0] + 1 : output_div_q[0]);
  input_div_q[0] = (x0->shape[1] / IMC_DIM);
  input_div_r[0] = (x0->shape[1] % IMC_DIM);
  input_div_q[0] = (input_div_r[0] > 0) ? input_div_q[0] + 1 : input_div_q[0];
  output_row[0] = y0->shape[2];
  output_col[0] = y0->shape[3];
  weight_row[0] = weight0->shape[2];
  weight_col[0] = weight0->shape[3];
  padding[0] = padding0;
  stride[0] = stride0;
  amem_scale[0] = amem_scale0;
  bias_scale[0] = bias_scale0;

  output_div_q[1] = (y1->shape[1] / IMC_DIM);
  output_div_r[1] = (y1->shape[1] % IMC_DIM);
  output_div_q[1] = ((output_div_r[1] > 0) ? output_div_q[1] + 1 : output_div_q[1]);
  input_div_q[1] = (x1->shape[1] / IMC_DIM);
  input_div_r[1] = (x1->shape[1] % IMC_DIM);
  input_div_q[1] = (input_div_r[1] > 0) ? input_div_q[1] + 1 : input_div_q[1];
  output_row[1] = y1->shape[2];
  output_col[1] = y1->shape[3];
  weight_row[1] = weight1->shape[2];
  weight_col[1] = weight1->shape[3];
  padding[1] = padding1;
  stride[1] = stride1;
  amem_scale[1] = amem_scale1;
  bias_scale[1] = bias_scale1;

  output_div_q[2] = (y2->shape[1] / IMC_DIM);
  output_div_r[2] = (y2->shape[1] % IMC_DIM);
  output_div_q[2] = ((output_div_r[2] > 0) ? output_div_q[2] + 1 : output_div_q[2]);
  input_div_q[2] = (x2->shape[1] / IMC_DIM);
  input_div_r[2] = (x2->shape[1] % IMC_DIM);
  input_div_q[2] = (input_div_r[2] > 0) ? input_div_q[2] + 1 : input_div_q[2];
  output_row[2] = y2->shape[2];
  output_col[2] = y2->shape[3];
  weight_row[2] = 2;
  weight_col[2] = 2;
  padding[2] = 0;
  stride[2] = 2;

  output_div_q[3] = (y3->shape[1] / IMC_DIM);
  output_div_r[3] = (y3->shape[1] % IMC_DIM);
  output_div_q[3] = ((output_div_r[3] > 0) ? output_div_q[3] + 1 : output_div_q[3]);
  input_div_q[3] = (x3->shape[1] / IMC_DIM);
  input_div_r[3] = (x3->shape[1] % IMC_DIM);
  input_div_q[3] = (input_div_r[3] > 0) ? input_div_q[3] + 1 : input_div_q[3];
  output_row[3] = y3->shape[2];
  output_col[3] = y3->shape[3];
  weight_row[3] = weight3->shape[2];
  weight_col[3] = weight3->shape[3];
  padding[3] = padding3;
  stride[3] = stride3;
  amem_scale[3] = amem_scale3;
  bias_scale[3] = bias_scale3;

  output_div_q[4] = (y4->shape[1] / IMC_DIM);
  output_div_r[4] = (y4->shape[1] % IMC_DIM);
  output_div_q[4] = ((output_div_r[4] > 0) ? output_div_q[4] + 1 : output_div_q[4]);
  input_div_q[4] = (x4->shape[1] / IMC_DIM);
  input_div_r[4] = (x4->shape[1] % IMC_DIM);
  input_div_q[4] = (input_div_r[4] > 0) ? input_div_q[4] + 1 : input_div_q[4];
  output_row[4] = y4->shape[2];
  output_col[4] = y4->shape[3];
  weight_row[4] = weight4->shape[2];
  weight_col[4] = weight4->shape[3];
  padding[4] = padding4;
  stride[4] = stride4;
  amem_scale[4] = amem_scale4;
  bias_scale[4] = bias_scale4;

  output_div_q[5] = (y5->shape[1] / IMC_DIM);
  output_div_r[5] = (y5->shape[1] % IMC_DIM);
  output_div_q[5] = ((output_div_r[5] > 0) ? output_div_q[5] + 1 : output_div_q[5]);
  input_div_q[5] = (x5->shape[1] / IMC_DIM);
  input_div_r[5] = (x5->shape[1] % IMC_DIM);
  input_div_q[5] = (input_div_r[5] > 0) ? input_div_q[5] + 1 : input_div_q[5];
  output_row[5] = y5->shape[2];
  output_col[5] = y5->shape[3];
  weight_row[5] = 1;
  weight_col[5] = 1;
  padding[5] = 0;
  stride[5] = 1;

  output_div_q[6] = (y6->shape[1] / IMC_DIM);
  output_div_r[6] = (y6->shape[1] % IMC_DIM);
  output_div_q[6] = ((output_div_r[6] > 0) ? output_div_q[6] + 1 : output_div_q[6]);
  input_div_q[6] = (x6->shape[1] / IMC_DIM);
  input_div_r[6] = (x6->shape[1] % IMC_DIM);
  input_div_q[6] = (input_div_r[6] > 0) ? input_div_q[6] + 1 : input_div_q[6];
  output_row[6] = y6->shape[2];
  output_col[6] = y6->shape[3];
  weight_row[6] = weight6->shape[2];
  weight_col[6] = weight6->shape[3];
  padding[6] = padding6;
  stride[6] = stride6;
  amem_scale[6] = amem_scale6;
  bias_scale[6] = bias_scale6;

  output_div_q[7] = (y7->shape[1] / IMC_DIM);
  output_div_r[7] = (y7->shape[1] % IMC_DIM);
  output_div_q[7] = ((output_div_r[7] > 0) ? output_div_q[7] + 1 : output_div_q[7]);
  input_div_q[7] = (x7->shape[1] / IMC_DIM);
  input_div_r[7] = (x7->shape[1] % IMC_DIM);
  input_div_q[7] = (input_div_r[7] > 0) ? input_div_q[7] + 1 : input_div_q[7];
  output_row[7] = y7->shape[2];
  output_col[7] = y7->shape[3];
  weight_row[7] = 2;
  weight_col[7] = 2;
  padding[7] = 0;
  stride[7] = 2;

  output_div_q[8] = (y8->shape[1] / IMC_DIM);
  output_div_r[8] = (y8->shape[1] % IMC_DIM);
  output_div_q[8] = ((output_div_r[8] > 0) ? output_div_q[8] + 1 : output_div_q[8]);
  input_div_q[8] = (x8->shape[1] / IMC_DIM);
  input_div_r[8] = (x8->shape[1] % IMC_DIM);
  input_div_q[8] = (input_div_r[8] > 0) ? input_div_q[8] + 1 : input_div_q[8];
  output_row[8] = y8->shape[2];
  output_col[8] = y8->shape[3];
  weight_row[8] = weight8->shape[2];
  weight_col[8] = weight8->shape[3];
  padding[8] = padding8;
  stride[8] = stride8;
  amem_scale[8] = amem_scale8;
  bias_scale[8] = bias_scale8;

  output_div_q[9] = (y9->shape[1] / IMC_DIM);
  output_div_r[9] = (y9->shape[1] % IMC_DIM);
  output_div_q[9] = ((output_div_r[9] > 0) ? output_div_q[9] + 1 : output_div_q[9]);
  input_div_q[9] = (x9->shape[1] / IMC_DIM);
  input_div_r[9] = (x9->shape[1] % IMC_DIM);
  input_div_q[9] = (input_div_r[9] > 0) ? input_div_q[9] + 1 : input_div_q[9];
  output_row[9] = y9->shape[2];
  output_col[9] = y9->shape[3];
  weight_row[9] = 2;
  weight_col[9] = 2;
  padding[9] = 0;
  stride[9] = 2;

  output_div_q[10] = (y10->shape[1] / IMC_DIM);
  output_div_r[10] = (y10->shape[1] % IMC_DIM);
  output_div_q[10] = ((output_div_r[10] > 0) ? output_div_q[10] + 1 : output_div_q[10]);
  input_div_q[10] = (x10->shape[1] / IMC_DIM);
  input_div_r[10] = (x10->shape[1] % IMC_DIM);
  input_div_q[10] = (input_div_r[10] > 0) ? input_div_q[10] + 1 : input_div_q[10];
  output_row[10] = y10->shape[2];
  output_col[10] = y10->shape[3];
  weight_row[10] = weight10->shape[2];
  weight_col[10] = weight10->shape[3];
  padding[10] = padding10;
  stride[10] = stride10;
  amem_scale[10] = amem_scale10;
  bias_scale[10] = bias_scale10;

  output_div_q[11] = (y11->shape[1] / IMC_DIM);
  output_div_r[11] = (y11->shape[1] % IMC_DIM);
  output_div_q[11] = ((output_div_r[11] > 0) ? output_div_q[11] + 1 : output_div_q[11]);
  input_div_q[11] = (x11->shape[1] / IMC_DIM);
  input_div_r[11] = (x11->shape[1] % IMC_DIM);
  input_div_q[11] = (input_div_r[11] > 0) ? input_div_q[11] + 1 : input_div_q[11];
  output_row[11] = y11->shape[2];
  output_col[11] = y11->shape[3];
  weight_row[11] = weight11->shape[2];
  weight_col[11] = weight11->shape[3];
  padding[11] = padding11;
  stride[11] = stride11;
  amem_scale[11] = amem_scale11;
  bias_scale[11] = bias_scale11;

  output_div_q[12] = (y12->shape[1] / IMC_DIM);
  output_div_r[12] = (y12->shape[1] % IMC_DIM);
  output_div_q[12] = ((output_div_r[12] > 0) ? output_div_q[12] + 1 : output_div_q[12]);
  input_div_q[12] = (x12->shape[1] / IMC_DIM);
  input_div_r[12] = (x12->shape[1] % IMC_DIM);
  input_div_q[12] = (input_div_r[12] > 0) ? input_div_q[12] + 1 : input_div_q[12];
  output_row[12] = y12->shape[2];
  output_col[12] = y12->shape[3];
  weight_row[12] = 1;
  weight_col[12] = 1;
  padding[12] = 0;
  stride[12] = 1;

  output_div_q[13] = (y13->shape[1] / IMC_DIM);
  output_div_r[13] = (y13->shape[1] % IMC_DIM);
  output_div_q[13] = ((output_div_r[13] > 0) ? output_div_q[13] + 1 : output_div_q[13]);
  input_div_q[13] = (x13->shape[1] / IMC_DIM);
  input_div_r[13] = (x13->shape[1] % IMC_DIM);
  input_div_q[13] = (input_div_r[13] > 0) ? input_div_q[13] + 1 : input_div_q[13];
  output_row[13] = y13->shape[2];
  output_col[13] = y13->shape[3];
  weight_row[13] = 2;
  weight_col[13] = 2;
  padding[13] = 0;
  stride[13] = 2;

  
  volatile int sp_weight_write_base=sp_w_write_index0; //0
  //volatile int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  volatile int amem_write_base=amem_write_index0; //16
  volatile int amem_read_base=amem_read_index0; //0

  volatile int i=0;
  volatile int j=0;

  write_reg(LAYER_FUSE_ENABLE, 1);
  for ( i = 0; i < NUM_FUSED_LAYERS; i += 1) {
    write_reg(LAYER_INDEX, i);
    write_reg(INPUT_ITR, input_div_q[i]);
    write_reg(OUTPUT_ITR, output_div_q[i]);
    write_reg(OUTPUT_COL, output_col[i]);
    write_reg(OUTPUT_ROW, output_row[i]);
    write_reg(WEIGHT_COL, weight_col[i]);
    write_reg(WEIGHT_ROW, weight_row[i]);
    write_reg(INPUT_REMAINDER, input_div_r[i]);
    write_reg(OUTPUT_REMAINDER, output_div_r[i]);
    write_reg(PADDING, padding[i]);
    write_reg(STRIDE, stride[i]);
    write_reg(AMEM_SCALE_CFG, amem_scale[i]);
    write_reg(BIAS_SCALE_CFG, bias_scale[i]);
    if (i==2||i==7||i==9||i==13){
      write_reg(LAYER_TYPE, 1);
    }
    else if (i==5||i==12){
      write_reg(LAYER_TYPE, 2);
    } 
    else{
      write_reg(LAYER_TYPE, 0);
    }
  }


  write_reg(DMA_READ_BASE, (uintptr_t)dmem_weight_address0);
  write_reg(DMA_READ_INPUT_BASE, (uintptr_t)dmem_input_address);
  //write_reg(SP_W_WR_BASE, sp_weight_write_base);
  //write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  /*write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);
  write_reg(INPUT_REMAINDER, input_div_r);
  write_reg(OUTPUT_REMAINDER, output_div_r);
  write_reg(NEXT_CONV_CHANNEL, next_conv_channel);

  write_reg(PADDING, padding);
  write_reg(STRIDE, stride);*/


  if (bias0) {
    write_reg(BIAS_EXIST, 1);
  }



  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base); // remove batch supporting can save RSICV cycle?

  //for resnet9
  //write_reg(CONV_LAYER_A_OTHER, 1);
  
  write_reg(LAYER_FUSED, 1);

  asm volatile("wfi");

}

/* ======================================================================================================== */
/*                                              pooling                                                     */
/* ======================================================================================================== */
/**
 * nn_avgpool4d_i6
 *
 * @brief Applies the Average Pooling function to a 4D tensor.
 *
 * 
 *fixed pool_size and stride to 2
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_avgpool4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, int amem_write_index, int amem_read_index) {
  //nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform Pooling on tensors of different channels");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=amem_read_index;

  volatile int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(OUTPUT_REMAINDER, (y->shape[1] % IMC_DIM));

  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(AVGPOOL_LAYER, 1);


}

/**
 * nn_maxpool4d_i6
 *
 * @brief Applies the Maximum Pooling function to a 4D tensor.
 *
 * 
 *fixed pool_size and stride to 2
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_maxpool4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, int amem_write_index, int amem_read_index) {
  //nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform Pooling on tensors of different channels");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=amem_read_index;

  volatile int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(OUTPUT_REMAINDER, (y->shape[1] % IMC_DIM));

  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(MAXPOOL_LAYER, 1);


}

void nn_reorg4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, int amem_write_index, int amem_read_index, int amem_read_index2) {
  //nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform Pooling on tensors of different channels");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  volatile int amem_write_base=amem_write_index;
  volatile int amem_read_base=amem_read_index;
  volatile int amem_read_base2=amem_read_index2;

  volatile int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;

  volatile int out_div_q = y->shape[1] / IMC_DIM;
  out_div_q = ((y->shape[1] % IMC_DIM) > 0) ? out_div_q + 1 : out_div_q;

  write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, out_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(OUTPUT_REMAINDER, (y->shape[1] % IMC_DIM));

  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base);
  write_reg(AMEM_RD_BASE2, amem_read_base2);


  write_reg(REORG_CAT, 1);


}

/**
 * Decodes pre-permuted, dequantized logits shaped [batch, cells, anchors, 5+classes].
 * The iou output is the sigmoid of the objectness logit, matching the Python decoder.
 */

void nn_yolo_decoding(Tensor4D_F32 *bbox, Tensor4D_F32 *iou,
                      Tensor4D_F32 *prob, const Tensor4D_I6 *x_int, int amem_read_index, int zero_point, int scale) {
  


  Tensor4D_F32 x;
  x.shape[0] = x_int->shape[0];
  x.shape[1] = x_int->shape[2]*x_int->shape[3];
  x.shape[2] = 5;
  x.shape[3] = x_int->shape[1]/5;
  nn_assert(scale != 0, "YOLO decoding scale must not be zero");

  const size_t x_num_elements =
      x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3];
  x.data = (float *)malloc(x_num_elements * sizeof(float));
  nn_assert(x.data != NULL, "Failed to allocate YOLO decoding buffer");

  volatile int amem_read_base=amem_read_index;
  volatile int8_t *imc_ptr = (volatile int8_t *)IMC_AMEM_0; // using volatile to prevent the compiler from optimizing per-byte accessing into bulk loads 
  for (int i = 0; i < x.shape[1]; i += 1) {
    for (int j = 0; j < x.shape[2]; j += 1) {
      for (int k = 0; k < x.shape[3]; k += 1) {
        const int quantized_value =
            imc_ptr[amem_read_base * IMC_DIM +
                    i * x.shape[2] * x.shape[3] + j * x.shape[3] + k];
        x.data[i * x.shape[2] * x.shape[3] + j * x.shape[3] + k] =
            (float)(quantized_value + zero_point) / (float)scale;
      }
    }
  }

  const size_t batch_size = x.shape[0];
  const size_t num_cells = x.shape[1];
  const size_t num_anchors = x.shape[2];
  const size_t num_features = x.shape[3];
  const size_t num_classes = num_features - 5;



  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t cell = 0; cell < num_cells; ++cell) {
      for (size_t anchor = 0; anchor < num_anchors; ++anchor) {
        const size_t input_offset =
            ((batch * num_cells + cell) * num_anchors + anchor) * num_features;
        const size_t bbox_offset =
            ((batch * num_cells + cell) * num_anchors + anchor) * 4;
        const size_t iou_offset =
            (batch * num_cells + cell) * num_anchors + anchor;
        const size_t prob_offset =
            ((batch * num_cells + cell) * num_anchors + anchor) * num_classes;

        bbox->data[bbox_offset] =
            1.0f / (1.0f + expf(-x.data[input_offset]));
        bbox->data[bbox_offset + 1] =
            1.0f / (1.0f + expf(-x.data[input_offset + 1]));
        bbox->data[bbox_offset + 2] = expf(x.data[input_offset + 2]);
        bbox->data[bbox_offset + 3] = expf(x.data[input_offset + 3]);
        iou->data[iou_offset] =
            1.0f / (1.0f + expf(-x.data[input_offset + 4]));

        float max_logit = x.data[input_offset + 5];
        for (size_t class_index = 1; class_index < num_classes; ++class_index) {
          const float logit = x.data[input_offset + 5 + class_index];
          if (logit > max_logit) {
            max_logit = logit;
          }
        }

        float exp_sum = 0.0f;
        for (size_t class_index = 0; class_index < num_classes; ++class_index) {
          const float exp_logit =
              expf(x.data[input_offset + 5 + class_index] - max_logit);
          prob->data[prob_offset + class_index] = exp_logit;
          exp_sum += exp_logit;
        }
        for (size_t class_index = 0; class_index < num_classes; ++class_index) {
          prob->data[prob_offset + class_index] /= exp_sum;
        }
      }
    }
  }
  free(x.data);
}

#endif // __NN_Q8_0_H