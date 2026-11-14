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
  Tensor4D_I6 actor_0;
  Tensor4D_I6 actor_0_weight;
  Tensor1D_I6 actor_0_bias;
  Tensor4D_I6 output;
} Model;

void model_init(Model* model) {
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 256;
  model->input_1.shape[2] = 4;
  model->input_1.shape[3] = 4;
  model->input_1.data = (int8_t *)malloc(4096.0);
  model->actor_0.shape[0] = 1;
  model->actor_0.shape[1] = 256;
  model->actor_0.shape[2] = 4;
  model->actor_0.shape[3] = 4;
  model->actor_0.data = (int8_t *)malloc(4096.0);
  model->actor_0_weight.shape[0] = 256;
  model->actor_0_weight.shape[1] = 256;
  model->actor_0_weight.shape[2] = 3;
  model->actor_0_weight.shape[3] = 3;
  model->actor_0_weight.data = (int8_t *)(model_weight_data + 0);
  model->actor_0_bias.shape[0] = 256;
  model->actor_0_bias.data = (int8_t *)(model_weight_data + 589824);
  model->output.shape[0] = 1;
  model->output.shape[1] = 256;
  model->output.shape[2] = 4;
  model->output.shape[3] = 4;
  model->output.data = (int8_t *)malloc(4096.0);
}

void model_forward(Model* model) {
  nn_conv2d_i6(&model->actor_0, &model->input_1, &model->actor_0_weight, &model->actor_0_bias, &model_weight_data[0], 0, (4096/IMC_DIM), (0/IMC_DIM), 256, (1, 1), (1, 1), (1, 1), 1);
  //memcpy(model->output.data, model->actor_0.data, 4096.0);
}

#endif  // __MODEL_H
