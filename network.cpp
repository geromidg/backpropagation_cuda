#include <stdlib.h>

#include "network.h"

Network::Network(const int& total_layer_num, const int* configuration, const float& gamma, const float& alpha):
  layer_num_(total_layer_num - 1)
{
  layers_ = new Layer*[layer_num_];

  for (int i = 0; i < layer_num_; i++)
    layers_[i] = new Layer(configuration[i], configuration[i + 1], gamma, alpha);
}

Network::~Network(void)
{
  if (layers_)
  {
    for (int i = 0; i < layer_num_; i++)
      delete layers_[i];

    delete[] layers_;
  }
}

float* Network::getNetworkOutput(void)
{
  return layers_[layer_num_ - 1]->getLayerOutput();
}

void Network::propagate(const float *input)
{
  layers_[0]->processNeurons(input);

  for (int i = 1; i < layer_num_; i++)
    layers_[i]->processNeurons(layers_[i - 1]->getLayerOutput());
}

void Network::train(const float *input, const float *expected_output)
{
  propagate(input);

  float layer_error = layers_[layer_num_ - 1]->fitNeurons(input, expected_output);

  for (int i = (layer_num_ - 2); i >= 0; i--)
    layer_error = layers_[i]->fitNeurons(input, layer_error);
}
