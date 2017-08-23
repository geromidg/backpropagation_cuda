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

void Network::propagateInputValues(const float *input)
{
  layers_[0]->propagateInput(input);

  for (int i = 1; i < layer_num_; i++)
    layers_[i]->propagateInput(layers_[i - 1]->getLayerOutput());
}

void Network::trainNetwork(const float *expected_output, const float *input)
{
  propagateInputValues(input);

  float layer_error = layers_[layer_num_ - 1]->trainLayer(input, expected_output);
  
  for (int i = (layer_num_ - 2); i >= 0; i--)
    layer_error = layers_[i]->trainLayer(input, layer_error);
}
