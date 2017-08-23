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
    weights_[i] = float(rand()) / float(RAND_MAX);

  bias_ = float(rand()) / float(RAND_MAX);
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

void Neuron::compute(const float* inputs)
{
  float weighted_sum = 0;

  for (int i = 0; i < input_num_; i++)
    weighted_sum += weights_[i] * inputs[i];

  weighted_sum += bias_;

  output_ = activationFunction(weighted_sum);
}

float Neuron::fitWeights(const float* input_values, const float& expected_output)
{
  float error_sum = 0;
  float error = output_ * (1 - output_) * (expected_output - output_);

  bias_ += gamma_ * error;

  for (int i = 0; i < input_num_; i++)
  {
    deltas_[i] *= alpha_;
    deltas_[i] += gamma_ * error * input_values[i];

    weights_[i] += deltas_[i];

    error_sum += error * weights_[i];
  }

  return error_sum;
}
