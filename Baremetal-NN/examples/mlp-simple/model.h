#ifndef __MODEL_H
#define __MODEL_H

#include "nn.h"

// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./model.bin", model_weight);
extern uint8_t model_weight_data[];
extern size_t model_weight_start[];
extern size_t model_weight_end[];

typedef struct {
  Tensor2D_I6 input_1;
  Tensor2D_I6 actor_0;
  Tensor2D_I6 actor_0_weight;
  Tensor1D_I6 actor_0_bias;
  Tensor2D_I6 output;
} Model;

void model_init(Model* model) {
  model->input_1.shape[0] = 1;
  model->input_1.shape[1] = 1024;
  model->input_1.data = (int8_t *)malloc(1024.0);
  model->actor_0.shape[0] = 1;
  model->actor_0.shape[1] = 10;
  model->actor_0.data = (int8_t *)malloc(10.0);
  model->actor_0_weight.shape[0] = 10;
  model->actor_0_weight.shape[1] = 1024;
  model->actor_0_weight.data = (int8_t *)(model_weight_data + 0);
  model->actor_0_bias.shape[0] = 10;
  model->actor_0_bias.data = (int8_t *)(model_weight_data + 10240);
  model->output.shape[0] = 1;
  model->output.shape[1] = 10;
  model->output.data = (int8_t *)malloc(10.0);
}

void model_forward(Model* model) {
  nn_linear_i6(&model->actor_0, &model->input_1, &model->actor_0_weight, &model->actor_0_bias);
  memcpy(model->output.data, model->actor_0.data, 10.0);
}

#endif  // __MODEL_H
