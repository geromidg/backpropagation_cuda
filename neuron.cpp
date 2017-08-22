#include <stdlib.h>
#include <math.h>

#include "neuron.h"

Neuron::Neuron(const int& input_num):
  input_num_(input_num),
  output(0)
{
  weights = new float[input_num];

  for (int i = 0; i < input_num; i++)
    weights[i] = float(rand()) / float(RAND_MAX);

  bias = float(rand()) / float(RAND_MAX);

  deltas = new float[input_num]();
}

Neuron::~Neuron(void)
{
  if (weights)
    delete[] weights;
  
  if (deltas)
    delete[] deltas;
}

void Neuron::calculateOutput(const float* inputs)
{
  float sum=0;

  for (int i = 0; i < input_num_; i++)
    sum += weights[i] * inputs[i];

  sum += bias;

  output = activationFunction(sum);
}

float Neuron::activationFunction(const float& weighted_sum)
{
  return (1 / (1 + exp(-weighted_sum)));  // sigmoid function
}
