#include <stdlib.h>

#include "layer.h"

Layer::Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha):
  neuron_num_(neuron_num)
{
  neurons_ = new Neuron*[neuron_num_];

  for (int i = 0; i < neuron_num_; i++)
    neurons_[i] = new Neuron(input_num, gamma, alpha);
}

Layer::~Layer(void)
{
  if (neurons_)
  {
    for (int i = 0; i < neuron_num_; i++)
      delete neurons_[i];

    delete[] neurons_;
  }
}

float* Layer::getLayerOutput(void)
{
  float* output = new float[neuron_num_];  // FIXME: Use smart pointer

  for (int i = 0; i < neuron_num_; i++)
    output[i] = neurons_[i]->getOutput();

  return output;
}

void Layer::propagateInput(const float* input_values)
{
  for (int i = 0; i < neuron_num_; i++)
    neurons_[i]->compute(input_values);
}

float Layer::trainLayer(const float* input_values, const float& next_layer_error)
{
  float expected_output_values[neuron_num_];

  for (int i = 0; i < neuron_num_; i++)
    expected_output_values[i] = next_layer_error + neurons_[i]->getOutput();

  return trainLayer(input_values, expected_output_values);
}

float Layer::trainLayer(const float* input_values, const float* expected_output_values)
{
  float error_sum = 0;

  for (int i = 0; i < neuron_num_; i++)
    error_sum += neurons_[i]->fitWeights(input_values, expected_output_values[i]);

  return error_sum;
}
