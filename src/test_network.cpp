#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"

#include "test_network.h"

TestNetwork::TestNetwork(const int& network_layer_num, 
    const int* network_configuration, const int& network_epochs,
    const float& network_gamma, const float& network_alpha,
    const int& dataset_size, float** dataset_input, float** dataset_output,
    const int& thread_num):
  network_epochs_(network_epochs),
  dataset_size_(dataset_size),
  dataset_input_(dataset_input),
  dataset_output_(dataset_output)
{
  network_ = new Network(network_layer_num, network_configuration,
    network_gamma, network_alpha, thread_num);
}

TestNetwork::~TestNetwork(void)
{
}

void TestNetwork::train(void)
{
  tic();

  for (int i = 0; i < network_epochs_; i++)
    for (int j = 0; j < dataset_size_; j++)
      network_->train(dataset_input_[j], dataset_output_[j]);

  toc("\nTraining time: %.3f s\n");
}

// TODO: Add support for datasets with more than 1 outputs
void TestNetwork::validate(void)
{
  float accuracy = 0;

  for (int i = 0; i < dataset_size_; i++)
  {
    network_->propagate(dataset_input_[i]);
    accuracy += std::abs(network_->getNetworkOutput()[0] - dataset_output_[i][0]);
  }

  printf("Accuracy: %.1f%% \n", 100 * (1 - accuracy / dataset_size_));
}

void TestNetwork::run(void)
{
  train();
  validate();
}
