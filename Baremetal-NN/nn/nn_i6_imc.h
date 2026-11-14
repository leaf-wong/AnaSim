/**
 * @file nn_i6.h
 * @brief Baremetal-NN Library functions for INT6 (i6) numbers
 *
 * This file contains the declarations of the functions and structures for the Baremetal-NN Library.
 */

#ifndef __NN_I6_H
#define __NN_I6_H

#include "float16.h"

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
#define BIAS_ENABLE (IMC_BASE_ADDR + 0x44)

#define L2CACHE_RD_BASE (IMC_BASE_ADDR + 0x48)
#define SP_W_WR_BASE (IMC_BASE_ADDR + 0x4c)
#define AMEM_RD_BASE (IMC_BASE_ADDR + 0x50)
#define INPUT_ITR (IMC_BASE_ADDR + 0x54)
#define OUTPUT_ITR (IMC_BASE_ADDR + 0x58)
#define OUTPUT_COL (IMC_BASE_ADDR + 0x5c)
#define OUTPUT_ROW (IMC_BASE_ADDR + 0x60)
#define WEIGHT_COL (IMC_BASE_ADDR + 0x64)
#define WEIGHT_ROW (IMC_BASE_ADDR + 0x68)

#define LINEAR_LAYER (IMC_BASE_ADDR + 0x100)
#define CONV_LAYER (IMC_BASE_ADDR + 0x104)
#define AVGPOOL_LAYER (IMC_BASE_ADDR + 0x108)
#define ReLU_LAYER (IMC_BASE_ADDR + 0x10c)

#define CONV_LAYER_A (IMC_BASE_ADDR + 0x110)

#define IMC_SP_INPUT  (IMC_BASE_ADDR + 0x1000)
#define IMC_SP_WEIGHT  (IMC_BASE_ADDR + 0x2000)
#define IMC_AMEM_0   (IMC_BASE_ADDR + 0x300000)
#define IMC_ARRAY_0  (IMC_BASE_ADDR + 0x400000)

#define IMC_CONTROL_MVM     (1)
#define IMC_CONTROL_RST     (0)


#define IMC_DIM 256
#define IMC_AMEM_TEMP_BASE   8192 // start from 0x380000 


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

/* ======================================================================================================== */
/*                                           Tensor Creation                                                */
/* ======================================================================================================== */
/**
 * nn_tensor0d_i6
 *
 * @brief Creates a 0D tensor with type I6.
 *
 * @param data The data to store in the tensor.
 */
Tensor0D_I6 *nn_tensor0d_i6(int8_t data) {
  Tensor0D_I6 *tensor = (Tensor0D_I6 *)malloc(sizeof(Tensor0D_I6));
  tensor->data = data;
  return tensor;
}

/**
 * nn_tensor1d_i6
 *
 * @brief Creates a 1D tensor with type I6.
 *
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor1D_I6 *nn_tensor1d_i6(size_t shape[1], const int8_t *data) {
  Tensor1D_I6 *tensor = (Tensor1D_I6 *)malloc(sizeof(Tensor1D_I6));
  tensor->shape[0] = shape[0];

  size_t n_bytes = shape[0] * sizeof(int8_t);
  tensor->data = (int8_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
    }
  return tensor;
}

/**
 * nn_tensor2d_i6
 *
 * @brief Creates a 2D tensor with type I6.
 *
 * @param shape The shape of the tensor.
 * @param data The data to store in the tensor.
 */
Tensor2D_I6 *nn_tensor2d_i6(size_t shape[2], const int8_t *data) {
  Tensor2D_I6 *tensor = (Tensor2D_I6 *)malloc(sizeof(Tensor2D_I6));
  tensor->shape[0] = shape[0];
  tensor->shape[1] = shape[1];

  size_t n_bytes = shape[0] * shape[1] * sizeof(int8_t);
  tensor->data = (int8_t *)malloc(n_bytes);
  if (data != NULL) {
    memcpy(tensor->data, data, n_bytes);
  }
  return tensor;
}

/**
 * nn_zeros0d_i6
 *
 * @brief Creates a 0D tensor with type I6 and initializes it to 0.
 *
 * @return The created tensor.
 */
Tensor0D_I6 *nn_zeros0d_i6() {
  Tensor0D_I6 *tensor = nn_tensor0d_i6(0);
  return tensor;
}

/**
 * nn_zeros1d_i6
 *
 * @brief Creates a 1D tensor with type I6 and initializes it to 0.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_I6 *nn_zeros1d_i6(size_t shape[1]) {
  Tensor1D_I6 *tensor = nn_tensor1d_i6(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * nn_zeros2d_i6
 *
 * @brief Creates a 2D tensor with type I6 and initializes it to 0.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor2D_I6 *nn_zeros2d_i6(size_t shape[2]) {
  Tensor2D_I6 *tensor = nn_tensor2d_i6(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 0;
  }
  return tensor;
}

/**
 * nn_ones0d_i6
 *
 * @brief Creates a 0D tensor with type I6 and initializes it to 1.
 *
 * @return The created tensor.
 */
Tensor0D_I6 *nn_ones0d_i6() {
  Tensor0D_I6 *tensor = nn_tensor0d_i6(1);
  return tensor;
}

/**
 * nn_ones1d_i6
 *
 * @brief Creates a 1D tensor with type I6 and initializes it to 1.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_I6 *nn_ones1d_i6(size_t shape[1]) {
  Tensor1D_I6 *tensor = nn_tensor1d_i6(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * nn_ones2d_i6
 *
 * @brief Creates a 2D tensor with type I6 and initializes it to 1.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor2D_I6 *nn_ones2d_i6(size_t shape[2]) {
  Tensor2D_I6 *tensor = nn_tensor2d_i6(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = 1;
  }
  return tensor;
}

/**
 * nn_full0d_i6
 *
 * @brief Creates a 0D tensor with type I6 and initializes it to a given value.
 *
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor0D_I6 *nn_full0d_i6(int8_t data) {
  Tensor0D_I6 *tensor = nn_tensor0d_i6(data);
  return tensor;
}

/**
 * nn_full1d_i6
 *
 * @brief Creates a 1D tensor with type I6 and initializes it to a given value.
 *
 * @param shape The shape of the tensor.
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor1D_I6 *nn_full1d_i6(size_t shape[1], int8_t data) {
  Tensor1D_I6 *tensor = nn_tensor1d_i6(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * nn_full2d_i6
 *
 * @brief Creates a 2D tensor with type I6 and initializes it to a given value.
 *
 * @param shape The shape of the tensor.
 * @param data The value to initialize the tensor to.
 * @return The created tensor.
 */
Tensor2D_I6 *nn_full2d_i6(size_t shape[2], int8_t data) {
  Tensor2D_I6 *tensor = nn_tensor2d_i6(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = data;
  }
  return tensor;
}

/**
 * nn_rand0d_i6
 *
 * @brief Creates a 0D tensor with type I6 and initializes it to a random value.
 *
 * @return The created tensor.
 */
Tensor0D_I6 *nn_rand0d_i6() {
  Tensor0D_I6 *tensor = nn_tensor0d_i6(rand());
  return tensor;
}

/**
 * nn_rand1d_i6
 *
 * @brief Creates a 1D tensor with type I6 and initializes it to a random value.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor1D_I6 *nn_rand1d_i6(size_t shape[1]) {
  Tensor1D_I6 *tensor = nn_tensor1d_i6(shape, NULL);
  size_t n = shape[0];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}

/**
 * nn_rand2d_i6
 *
 * @brief Creates a 2D tensor with type I6 and initializes it to a random value.
 *
 * @param shape The shape of the tensor.
 * @return The created tensor.
 */
Tensor2D_I6 *nn_rand2d_i6(size_t shape[2]) {
  Tensor2D_I6 *tensor = nn_tensor2d_i6(shape, NULL);
  size_t n = shape[0] * shape[1];
  for (size_t i = 0; i < n; i += 1) {
    tensor->data[i] = rand();
  }
  return tensor;
}


/* ======================================================================================================== */
/*                                           Tensor Prints                                                  */
/* ======================================================================================================== */
/**
 * nn_print_i6
 *
 * @brief Prints a int8_t number.
 *
 * @param v The int8_t to print.
 */
void nn_print_i6(int8_t v) {
  printf("%d", v);
}


/**
 * nn_print_tensor1d_f16
 *
 * @brief Prints the content of a 1D tensor with type I6.
 *
 * @param tensor The 1D tensor to print.
 */
void nn_print_tensor1d_i6(const Tensor1D_I6 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    nn_print_i6(*((int8_t *)tensor->data + i));
    if (i < tensor->shape[0]-1) {
      printf(" ");
    }
  }
  printf("]\n");
}

/**
 * nn_print_tensor2d_f16
 *
 * @brief Prints the content of a 2D tensor with type I6.
 *
 * @param tensor The 2D tensor to print.
 */
void nn_print_tensor2d_i6(const Tensor2D_I6 *tensor) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      nn_print_i6(*((int8_t *)tensor->data + i*tensor->shape[1] + j));
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf("]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

/**
 * nn_print_tensor2d_i6_to_f32
 *
 * @brief Prints the content of a 2D INT6 data tensor to float.
 *
 * @param tensor The 2D floating-point data tensor to print.
 */
void nn_print_tensor2d_i6_to_f32(const Tensor2D_I6 *tensor, float scale) {
  printf("[");
  for (size_t i=0; i<tensor->shape[0]; i+=1) {
    if (i == 0) {
      printf("[");
    }
    else {
      printf(" [");
    }
    for (size_t j=0; j<tensor->shape[1]; j+=1) {
      int8_t v = *((int8_t *)tensor->data + i * tensor->shape[1] + j);
      nn_print_f32((float)v * scale, 3);
      if (j < tensor->shape[1]-1) {
        printf(" ");
      }
    }
    printf(" ]");
    if (i < tensor->shape[0]-1) {
      printf("\n");
    }
  }
  printf("]\n");
}

/* ======================================================================================================== */
/*                                           Comparision                                                    */
/* ======================================================================================================== */
/**
 * nn_equals0d_i6
 *
 * @brief Checks if two 0D tensors with type I6 are equal.
 *
 * @param a The first 0D tensor.
 * @param b The second 0D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals0d_i6(const Tensor0D_I6 *a, const Tensor0D_I6 *b) {
  return a->data == b->data;
}

/**
 * nn_equals1d_i6
 *
 * @brief Checks if two 1D tensors with type I6 are equal.
 *
 * @param a The first 1D tensor.
 * @param b The second 1D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals1d_i6(const Tensor1D_I6 *a, const Tensor1D_I6 *b) {
  nn_assert(a->shape[0] == b->shape[0], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0];
  for (size_t i = 0; i < n; i += 1) {
    if (a->data[i] != b->data[i]) {
      return 0;
    }
  }
  return 1;
}

/**
 * nn_equals2d_i6
 *
 * @brief Checks if two 2D tensors with type I6 are equal.
 *
 * @param a The first 2D tensor.
 * @param b The second 2D tensor.
 * @return 1 if the tensors are equal, 0 otherwise.
 */
uint8_t nn_equals2d_i6(const Tensor2D_I6 *a, const Tensor2D_I6 *b) {
  nn_assert(a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1], "Cannot compare tensors of different shapes");

  size_t n = a->shape[0] * a->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    if (a->data[i] != b->data[i]) {
      return 0;
    }
  }
  return 1;
}


/* ======================================================================================================== */
/*                                           Unary                                                          */
/* ======================================================================================================== */




/* ======================================================================================================== */
/*                                           Addition                                                       */
/* ======================================================================================================== */
/**
 * nn_add1d_i6
 *
 * @brief Adds x1 and x2 element-wise and stores the result in y.
 *
 * y[i] = x1[i] + x2[i]
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_add1d_i6(Tensor1D_I6 *y, const Tensor1D_I6 *x1, const Tensor1D_I6 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  int8_t *x1_data = x1->data;
  int8_t *x2_data = x2->data;
  int8_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_VECTOR
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint32m1_t vec_x1 = __riscv_vle32_v_i6m1(x1_data, vl);
      vint32m1_t vec_x2 = __riscv_vle32_v_i6m1(x2_data, vl);
      vint32m1_t vec_y = __riscv_vfadd_vv_i6m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_i6m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y_data[i] = x1_data[i] + x2_data[i];
    }
  #endif
}



/**
 * nn_add2d_i6
 *
 * @brief Adds x1 and x2 element-wise and stores the result in y.
 *
 * y[i][j] = x1[i][j] + x2[i][j]
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_add2d_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x1, const Tensor2D_I6 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0] && x1->shape[1] == x2->shape[1], "Cannot add tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x1->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x1_data = x1->data;
  int8_t *x2_data = x2->data;
  int8_t *y_data = y->data;

  #ifdef CONFIG_BACKEND_RISCV_VECTOR
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e8m1(n);
      vint32m1_t vec_x1 = __riscv_vle32_v_i6m1(x1_data, vl);
      vint32m1_t vec_x2 = __riscv_vle32_v_i6m1(x2_data, vl);
      vint32m1_t vec_y = __riscv_vfadd_vv_i6m1(vec_x1, vec_x2, vl);
      __riscv_vse32_v_i6m1(y_data, vec_y, vl);
      x1_data += vl;
      x2_data += vl;
      y_data += vl;
      n -= vl;
    }
  #else  // scalar implementation
    for (size_t i = 0; i < n; i += 1) {
      y_data[i] = x1_data[i] + x2_data[i];
    }
  #endif
}

/**
 * nn_addscalar1d_i6
 *
 * @brief Adds a scalar to a 1D tensor and stores the result in y.
 *
 * y[i] = x[i] + scalar
 *
 * @param y The result tensor.
 * @param x The tensor to add the scalar to.
 * @param scalar The scalar to add.
 */
void nn_addscalar1d_i6(Tensor1D_I6 *y, const Tensor1D_I6 *x, int8_t scalar) {
  nn_assert(y->shape[0] == x->shape[0], "Cannot add tensors of different shapes");

  size_t n = y->shape[0];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    y_data[i] = x_data[i] + scalar;
  }
}

/**
 * nn_addscalar2d_i6
 *
 * @brief Adds a scalar to a 2D tensor and stores the result in y.
 *
 * y[i][j] = x[i][j] + scalar
 *
 * @param y The result tensor.
 * @param x The tensor to add the scalar to.
 * @param scalar The scalar to add.
 */
void nn_addscalar2d_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x, int8_t scalar) {
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[1], "Cannot add tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    y_data[i] = x_data[i] + scalar;
  }
}




/* ======================================================================================================== */
/*                                           Multiplication                                                 */
/* ======================================================================================================== */




/* ======================================================================================================== */
/*                                           MatMul                                                         */
/* ======================================================================================================== */

/**
 * nn_dot_i6
 *
 * @brief Computes the dot product of two 1D tensors and stores the result in y.
 *
 * y[0] = x1[0] * x2[0] + x1[1] * x2[1] + ... + x1[n-1] * x2[n-1]
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_dot_i6(Tensor1D_I6 *y, const Tensor1D_I6 *x1, const Tensor1D_I6 *x2) {
  nn_assert(x1->shape[0] == x2->shape[0], "Cannot dot tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0], "Cannot dot tensors of different shapes");

  size_t n = y->shape[0];
  int8_t *x1_data = x1->data;
  int8_t *x2_data = x2->data;
  int8_t *y_data = y->data;

  int8_t sum_i6 = 0;
  for (size_t i = 0; i < n; i += 1) {
    sum_i6 += x1_data[i] * x2_data[i];
  }
  y_data[0] = sum_i6;
}

/**
 * nn_mm_i6
 *
 * @brief Performs matrix multiplication of two 2D tensors and stores the result in y.
 *
 * y[i][j] = x1[i][k] * x2[k][j]
 *
 * @param y The result tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_mm_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x1, const Tensor2D_I6 *x2) {
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];
  int8_t *x1_data = x1->data;
  int8_t *x2_data = x2->data;
  int8_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < p; j += 1) {
      int8_t sum = 0;
      for (size_t k = 0; k < m; k += 1) {
        sum += x1_data[i * m + k] * x2_data[k * p + j];
      }
      y_data[i * p + j] = sum;
    }
  }
}

/**
 * nn_addmm_i6
 *
 * @brief Performs matrix multiplication of two 2D tensors and adds the result to a third tensor.
 *
 * y[i][j] = x1[i][k] * x2[k][j] + c[i][j]
 *
 * @param y The result tensor.
 * @param c The third tensor.
 * @param x1 The first tensor.
 * @param x2 The second tensor.
 */
void nn_addmm_i6(Tensor2D_I6 *y, const Tensor2D_I6 *c, const Tensor2D_I6 *x1, const Tensor2D_I6 *x2) {
  nn_assert(x1->shape[1] == x2->shape[0], "Cannot perform MatMul on tensors of different shapes");
  nn_assert(y->shape[0] == x1->shape[0] && y->shape[1] == x2->shape[1], "Cannot perform MatMul on tensors of different shapes");

  const size_t n = x1->shape[0];
  const size_t m = x1->shape[1];
  const size_t p = x2->shape[1];
  int8_t *x1_data = x1->data;
  int8_t *x2_data = x2->data;
  int8_t *c_data = c->data;
  int8_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < p; j += 1) {
      int8_t sum = 0;
      for (size_t k = 0; k < m; k += 1) {
        sum += x1_data[i * m + k] * x2_data[k * p + j];
      }
      y_data[i * p + j] = sum + c_data[i * p + j];
    }
  }
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
void nn_linear_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x, const Tensor2D_I6 *weight, const Tensor1D_I6 *bias) {
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
  nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = x->shape[1];
  const size_t out_features = weight->shape[0];

  int8_t *x_batch_data = x->data;
  int8_t *y_batch_data = y->data;

  int input_div_q = weight->shape[1] / IMC_DIM;
  int input_div_r = weight->shape[1] % IMC_DIM;
  input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  int output_div_q = weight->shape[0] / IMC_DIM;
  int output_div_r = weight->shape[0] % IMC_DIM;
  output_div_q = (output_div_r > 0) ? output_div_q + 1 : output_div_q;

  
  
  int sp_weight_write_base=0;
  int l2cache_read_base=input_div_q*batch_size;
  int amem_write_base=16;
  int amem_read_base=0;


  for (size_t i = 0; i < output_div_q; i += 1) {
  write_reg(DMA_ROWIDX, sp_weight_write_base+output_div_q*input_div_q*IMC_DIM+i);
  write_reg(DRAM2SP_W, l2cache_read_base+output_div_q*input_div_q*IMC_DIM+i);
    for (size_t j = 0; j < input_div_q; j += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+i*input_div_q*IMC_DIM+j*IMC_DIM);
        write_reg(DRAM2SP_W_BURST, l2cache_read_base+i*input_div_q*IMC_DIM+j*IMC_DIM);
    }
  }

  //TODO: input should alreay in AMEM
  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < input_div_q; j += 1) {
      write_reg(DMA_ROWIDX, i*input_div_q + j);
      write_reg(DRAM2SP_IN, i*input_div_q + j);
      write_reg(SP_IN2AMEM, i*input_div_q + j);
    }
  }

  write_reg(SP_W_WR_BASE, sp_weight_write_base);
  write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  write_reg(INPUT_ITR, input_div_q);
  write_reg(OUTPUT_ITR, output_div_q);


  if (bias) {
    write_reg(BIAS_ENABLE, 1);
  }


  for (int i = 0; i < batch_size; i += 1) {
    int8_t *x_data = x_batch_data;
    int8_t *y_data = y_batch_data;

    write_reg(DMA_ROWIDX, amem_write_base+i*output_div_q);
    write_reg(AMEM_RD_BASE, amem_read_base+i*input_div_q);

    write_reg(LINEAR_LAYER, 1);

    
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
 */
void nn_conv2d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, const Tensor4D_I6 *weight, const Tensor1D_I6 *bias, char * dmem_weight_address,int stride, int padding, int dilation, int groups) {
  nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Conv on tensors of different shapes");
  nn_assert(!bias || bias->shape[0] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");
  nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Conv on tensors of different shapes");

  const size_t batch_size = x->shape[0];
  const size_t in_features = (x->shape[1] * x->shape[2] * x->shape[3]);
  const size_t out_features = weight->shape[0];

  int8_t *x_batch_data = x->data;
  int8_t *y_batch_data = y->data;

  int input_div_q = (weight->shape[1] * weight->shape[2] * weight->shape[3]) / IMC_DIM;
  int input_div_r = (weight->shape[1] * weight->shape[2] * weight->shape[3]) % IMC_DIM;
  input_div_q = (input_div_r > 0) ? input_div_q + 1 : input_div_q;

  int output_div_q = weight->shape[0] / IMC_DIM;
  int output_div_r = weight->shape[0] % IMC_DIM;
  output_div_q = (output_div_r > 0) ? output_div_q + 1 : output_div_q;

  int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;
  
  int sp_weight_write_base=0;
  int l2cache_read_base=(x->shape[2]*x->shape[3])*batch_size;
  int amem_write_base=16;
  int amem_read_base=0;

  
  for (size_t i = 0; i < output_div_q; i += 1) {
  
    for (size_t j = 0; j < (amem_div_q* weight->shape[2] * weight->shape[3]) ; j += 1) {
      for (size_t k = 0; k < IMC_DIM; k += 1) {
        write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM+j*IMC_DIM+k);
        write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+k*IMC_DIM);
        //write_reg(DMA_ROWIDX, sp_weight_write_base+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM+j*IMC_DIM);
        //write_reg(DRAM2SP_W_BURST, (uintptr_t)dmem_weight_address+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM);
        //asm volatile("wfi");
        //write_reg(IMC_IE, 0);
        //printf("DRAM2SP_W_BURST: first=%d 3rd=%d last-1=%d last=%d\n", dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM + IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+2*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-2)*IMC_DIM+ IMC_DIM-1], dmem_weight_address[i*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+j*IMC_DIM*IMC_DIM+(IMC_DIM-1)*IMC_DIM+ IMC_DIM-1]);
      }
    }
    write_reg(DMA_ROWIDX, sp_weight_write_base+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM+i);
    write_reg(DRAM2SP_W, (uintptr_t)dmem_weight_address+output_div_q*amem_div_q* weight->shape[2] * weight->shape[3]*IMC_DIM*IMC_DIM+i*IMC_DIM);
  }

  //TODO: input should alreay in AMEM. 
 
  /*
  for (size_t i = 0; i < batch_size; i += 1) {
    for (size_t j = 0; j < y->shape[2]; j += 1) {
      for (size_t k = 0; k < y->shape[3]; k += 1) {
        for (size_t l = 0; l < amem_div_q; l += 1) {
          write_reg(DMA_ROWIDX, i*y->shape[2]*y->shape[3]*amem_div_q + j* y->shape[3]*amem_div_q + k*amem_div_q + l);
          write_reg(DRAM2SP_IN, i*y->shape[2]*y->shape[3]*amem_div_q + j* y->shape[3]*amem_div_q + k*amem_div_q + l);
          write_reg(SP_IN2AMEM, i*y->shape[2]*y->shape[3]*amem_div_q + j* y->shape[3]*amem_div_q + k*amem_div_q + l);
        }
      }
    }
  }*/

  write_reg(SP_W_WR_BASE, sp_weight_write_base);
  write_reg(L2CACHE_RD_BASE, l2cache_read_base);
  write_reg(INPUT_ITR, amem_div_q);
  write_reg(OUTPUT_ITR, output_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);
  write_reg(WEIGHT_COL, weight->shape[2]);
  write_reg(WEIGHT_ROW, weight->shape[3]);


  if (bias) {
    write_reg(BIAS_ENABLE, 1);
  }


  for (int i = 0; i < batch_size; i += 1) {
    int8_t *x_data = x_batch_data;
    int8_t *y_data = y_batch_data;

    write_reg(DMA_ROWIDX, amem_write_base+i*output_div_q* y->shape[2] * y->shape[3]);
    write_reg(AMEM_RD_BASE, amem_read_base+i*amem_div_q* weight->shape[2] * weight->shape[3]);

    write_reg(CONV_LAYER_A, 1);
    //write_reg(DMA_ROWIDX, 0);

    //asm volatile("wfi");

    //just for debug prints
    //while (read_reg(IMC_REG_STATUS) == 0) { printf("waiting\n");}
    printf("output read start\n");

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


/* ======================================================================================================== */
/*                                           Non-linear                                                     */
/* ======================================================================================================== */

/**
 * nn_relu2d_i6
 *
 * @brief Applies the ReLU activation function to a 2D tensor.
 *
 * y[i][j] = max(x[i][j], 0)
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_relu2d_i6(Tensor2D_I6 *y, const Tensor2D_I6 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  for (size_t i = 0; i < n; i += 1) {
    y_data[i] = x_data[i] > 0 ? x_data[i] : 0;
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
void nn_relu4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1] && x->shape[2] == y->shape[2] && x->shape[3] == y->shape[3], "Cannot perform ReLU on tensors of different shapes");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  int amem_write_base=1024;
  int amem_read_base=0;

  int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);

  write_reg(DMA_ROWIDX, amem_write_base);
  //write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(ReLU_LAYER, 1);
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
 *
 * @param y The result tensor.
 * @param x The input tensor.
 */
void nn_avgpool4d_i6(Tensor4D_I6 *y, const Tensor4D_I6 *x, int pool_size, int stride) {
  nn_assert(x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1], "Cannot perform Pooling on tensors of different channels");

  size_t n = y->shape[0] * y->shape[1];
  int8_t *x_data = x->data;
  int8_t *y_data = y->data;

  int amem_write_base=1024;
  int amem_read_base=0;

  int amem_div_q = x->shape[1] / IMC_DIM;
  amem_div_q = ((x->shape[1] % IMC_DIM) > 0) ? amem_div_q + 1 : amem_div_q;

  write_reg(OUTPUT_ITR, amem_div_q);
  write_reg(OUTPUT_COL, y->shape[2]);
  write_reg(OUTPUT_ROW, y->shape[3]);

  write_reg(DMA_ROWIDX, amem_write_base);
  write_reg(AMEM_RD_BASE, amem_read_base);

  write_reg(AVGPOOL_LAYER, 1);


}


#endif // __NN_Q8_0_H