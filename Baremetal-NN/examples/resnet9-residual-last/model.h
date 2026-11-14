#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor4D_I6 input_1;
  Tensor4D_I6 conv1;
  Tensor4D_I6 conv1_weight;
  Tensor1D_I6 conv1_bias;
  Tensor4D_I6 relu;
  Tensor4D_I6 conv2;
  Tensor4D_I6 conv2_weight;
  Tensor1D_I6 conv2_bias;
  Tensor4D_I6 relu_1;
  Tensor4D_I6 add;
  Tensor4D_I6 output;
} Model;

void model_init(Model* model) {
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 256;
  model->input_1.shape[2] = 4;
  model->input_1.shape[3] = 4;
  model->input_1.data = (int8_t *)malloc(4096.0);
  model->conv1.shape[0] = 1;
  model->conv1.shape[1] = 256;
  model->conv1.shape[2] = 4;
  model->conv1.shape[3] = 4;
  model->conv1.data = (int8_t *)malloc(4096.0);
  model->conv1_weight.shape[0] = 256;
  model->conv1_weight.shape[1] = 256;
  model->conv1_weight.shape[2] = 3;
  model->conv1_weight.shape[3] = 3;
  model->conv1_weight.data = (int8_t *)(model_weight_data + 0);
  model->conv1_bias.shape[0] = 256;
  model->conv1_bias.data = (int8_t *)(model_weight_data + 589824);
  model->relu.shape[0] = 1;
  model->relu.shape[1] = 256;
  model->relu.shape[2] = 4;
  model->relu.shape[3] = 4;
  model->relu.data = (int8_t *)malloc(4096.0);
  model->conv2.shape[0] = 1;
  model->conv2.shape[1] = 256;
  model->conv2.shape[2] = 4;
  model->conv2.shape[3] = 4;
  model->conv2.data = (int8_t *)malloc(4096.0);
  model->conv2_weight.shape[0] = 256;
  model->conv2_weight.shape[1] = 256;
  model->conv2_weight.shape[2] = 3;
  model->conv2_weight.shape[3] = 3;
  model->conv2_weight.data = (int8_t *)(model_weight_data + 590080);
  model->conv2_bias.shape[0] = 256;
  model->conv2_bias.data = (int8_t *)(model_weight_data + 1179904);
  model->relu_1.shape[0] = 1;
  model->relu_1.shape[1] = 256;
  model->relu_1.shape[2] = 4;
  model->relu_1.shape[3] = 4;
  model->relu_1.data = (int8_t *)malloc(4096.0);
  model->add.shape[0] = 1;
  model->add.shape[1] = 256;
  model->add.shape[2] = 4;
  model->add.shape[3] = 4;
  model->add.data = (int8_t *)malloc(4096.0);
  model->output.shape[0] = 1;
  model->output.shape[1] = 256;
  model->output.shape[2] = 4;
  model->output.shape[3] = 4;
  model->output.data = (int8_t *)malloc(4096.0);
}

void model_forward(Model* model) {
  nn_conv2d_i6(&model->conv1, &model->input_1, &model->conv1_weight, &model->conv1_bias, &model_weight_data[0], (0/IMC_DIM), (4096/IMC_DIM), (0/IMC_DIM), (1, 1), (1, 1), (1, 1), 1);
  nn_relu4d_i6(&model->relu, &model->conv1, (4096/IMC_DIM));
  nn_conv2d_i6(&model->conv2, &model->relu, &model->conv2_weight, &model->conv2_bias, &model_weight_data[590080], (590080/IMC_DIM), (8192/IMC_DIM), (4096/IMC_DIM), (1, 1), (1, 1), (1, 1), 1);
  nn_relu4d_i6(&model->relu_1, &model->conv2, (8192/IMC_DIM));
  nn_add4d_i6(&model->add, &model->relu_1, &model->input_1, (8192/IMC_DIM), 0);
  //memcpy(model->output.data, model->add.data, 4096.0);
}

#endif  // __MODEL_H
