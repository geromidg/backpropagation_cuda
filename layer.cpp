#include <stdlib.h>

#include "layer.h"

Layer::Layer(const int& input_num,const int& neuron_num, const float& gamma, const float& alpha):
  input_num(input_num),
  neuron_num(neuron_num),
  gamma(gamma),
  alpha(alpha)
{
  neurons = new Neuron*[neuron_num];

  for (int i = 0; i < neuron_num;i++)
    neurons[i] = new Neuron(input_num);

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
    neurons[i]->calculateOutput(layerinput);
}

// FIXME: rename
float Layer::computeNewWeights(const float& next_layer_error)
{
  float output, error;
  float error_sum;

  for (int i = 0; i < neuron_num; i++)
  {
    output = neurons[i]->output;
    error = output * (1 - output) * next_layer_error;

    neurons[i]->bias += gamma * error;

    for (int j = 0; j < input_num; j++)
    {
      neurons[i]->deltas[j] = gamma * error * layerinput[j] + neurons[i]->deltas[j] * alpha;
      neurons[i]->weights[j] += neurons[i]->deltas[j];

      error_sum += neurons[i]->weights[j] * error;
    }
  }

  return error_sum;
}

float Layer::computeNewWeights(const float* desiredoutput)
{
  float output, error;
  float error_sum;

  for (int i = 0; i < neuron_num; i++)
  {
    output = neurons[i]->output;
    error = output * (1 - output) * (desiredoutput[i] - output);

    neurons[i]->bias += gamma * error;

    for (int j = 0; j < input_num; j++)
    {
      neurons[i]->deltas[j] = gamma * error * layerinput[j] + neurons[i]->deltas[j] * alpha;
      neurons[i]->weights[j] += neurons[i]->deltas[j];

      error_sum += neurons[i]->weights[j] * error;
    }
  }

  return error_sum;
}
