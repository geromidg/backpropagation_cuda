#include <stdlib.h>
#include <math.h>

#include "neuron.h"

Neuron::Neuron(const int& input_num, const float& gamma, const float& alpha):
  input_num(input_num),
  output(0),
  gamma(gamma),
  alpha(alpha)
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

void Neuron::computeOutput(const float* inputs)
{
  float weighted_sum = 0;

  for (int i = 0; i < input_num; i++)
    weighted_sum += weights[i] * inputs[i];

  weighted_sum += bias;

  output = activationFunction(weighted_sum);
}

float Neuron::activationFunction(const float& weighted_sum)
{
  return (1 / (1 + exp(-weighted_sum)));  // sigmoid function
}

float Neuron::fitWeights(const float* input_values, const float& expected_output)
{
  float error_sum = 0;
  float error = output * (1 - output) * (expected_output - output);

  bias += gamma * error;

  for (int i = 0; i < input_num; i++)
  {
    deltas[i] = gamma * error * input_values[i] + alpha * deltas[i];
    weights[i] += deltas[i];

    error_sum += error * weights[i];
  }

  return error_sum;
}
