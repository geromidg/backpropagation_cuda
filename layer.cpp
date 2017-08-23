#include <stdlib.h>

#include "layer.h"

Layer::Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha):
  input_num(input_num),
  neuron_num(neuron_num)
{
  neurons = new Neuron*[neuron_num];

  for (int i = 0; i < neuron_num;i++)
    neurons[i] = new Neuron(input_num, gamma, alpha);
}

Layer::~Layer(void)
{
  if (neurons)
  {
    for(int i = 0; i < neuron_num; i++)
      delete neurons[i];

    delete[] neurons;
  }
}

float* Layer::getLayerOutput(void)
{
  float* output = new float[neuron_num];  // FIXME: Use smart pointer

  for (int i = 0; i < neuron_num; i++)
    output[i] = neurons[i]->getOutput();

  return output;
}

void Layer::propagateInput(const float* input_values)
{
  for (int i = 0; i < neuron_num; i++)
    neurons[i]->compute(input_values);
}

float Layer::trainLayer(const float* input_values, const float& next_layer_error)
{
  float expected_output_values[neuron_num];

  for (int i = 0; i < neuron_num; i++)
    expected_output_values[i] = next_layer_error + neurons[i]->getOutput();

  return trainLayer(input_values, expected_output_values);
}

float Layer::trainLayer(const float* input_values, const float* expected_output_values)
{
  float error_sum = 0;

  for (int i = 0; i < neuron_num; i++)
    error_sum += neurons[i]->fitWeights(input_values, expected_output_values[i]);

  return error_sum;
}
