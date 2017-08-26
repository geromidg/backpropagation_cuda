#include <stdlib.h>
#include <math.h>

#include "neuron.h"

Neuron::Neuron(const int& input_num, const float& gamma, const float& alpha):
  input_num_(input_num),
  output_(0),
  gamma_(gamma),
  alpha_(alpha)
{
  weights_ = new float[input_num_];
  deltas_ = new float[input_num_]();

  for (int i = 0; i < input_num_; i++)
    weights_[i] = 0;

  bias_ = 0;
}

Neuron::~Neuron(void)
{
  if (weights_)
    delete[] weights_;
  
  if (deltas_)
    delete[] deltas_;
}

float Neuron::activationFunction(const float& weighted_sum)
{
  return (1 / (1 + exp(-weighted_sum)));  // sigmoid function
}

float Neuron::getOutput(void)
{
  return output_;
}

void Neuron::process(const float* inputs)
{
  float weighted_sum = 0;

  for (int i = 0; i < input_num_; i++)
    weighted_sum += weights_[i] * inputs[i];

  weighted_sum += bias_;

  output_ = activationFunction(weighted_sum);
}
