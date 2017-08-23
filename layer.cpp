#include <stdlib.h>

#include "layer.h"

Layer::Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha):
  input_num(input_num),
  neuron_num(neuron_num)
{
  neurons = new Neuron*[neuron_num];

  for (int i = 0; i < neuron_num;i++)
    neurons[i] = new Neuron(input_num, gamma, alpha);

  layerinput = new float[input_num];
}

Layer::~Layer(void)
{
  if (neurons)
  {
    for(int i = 0; i < neuron_num; i++)
      delete neurons[i];

    delete[] neurons;
  }

  if (layerinput)
    delete[] layerinput;
}

void Layer::calculateNeuronOutputs()
{
  for (int i = 0; i < neuron_num; i++)
    neurons[i]->computeOutput(layerinput);
}

float Layer::trainLayer(const float& next_layer_error)
{
  float expected_output_values[neuron_num];

  for (int i = 0; i < neuron_num; i++)
    expected_output_values[i] = next_layer_error + neurons[i]->output;

  return trainLayer(expected_output_values);
}

float Layer::trainLayer(const float* expected_output_values)
{
  float error_sum = 0;

  for (int i = 0; i < neuron_num; i++)
    error_sum += neurons[i]->fitWeights(layerinput, expected_output_values[i]);

  return error_sum;
}
