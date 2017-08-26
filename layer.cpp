#include <stdlib.h>

#include "training.cuh"

#include "layer.h"

Layer::Layer(const int& input_num, const int& neuron_num,
    const float& gamma, const float& alpha,
    const int& thread_num):
  neuron_num_(neuron_num),
  thread_num_(thread_num)
{
  neurons_ = new Neuron*[neuron_num_];

  for (int i = 0; i < neuron_num_; i++)
    neurons_[i] = new Neuron(input_num, gamma, alpha);

  if (neuron_num_ < thread_num_)
    thread_num_ = neuron_num_;

  thread_num_--;  // Account for master thread
  threads_ = new std::thread[thread_num_];
}

Layer::~Layer(void)
{
  if (neurons_)
  {
    for (int i = 0; i < neuron_num_; i++)
      delete neurons_[i];

    delete[] neurons_;
  }

  if (threads_)
    delete [] threads_;
}

float* Layer::getLayerOutput(void)
{
  float* output = new float[neuron_num_];  // FIXME: Use smart pointer

  for (int i = 0; i < neuron_num_; i++)
    output[i] = neurons_[i]->getOutput();

  return output;
}

void Layer::processNeurons(const float* input)
{
  for (int i = 0; i < neuron_num_; i++)
    neurons_[i]->process(input);
}

float Layer::trainLayer(const float* input, const float& next_layer_error)
{
  float expected_output[neuron_num_];

  for (int i = 0; i < neuron_num_; i++)
    expected_output[i] = next_layer_error + neurons_[i]->getOutput();

  return trainLayer(input, expected_output);
}

float Layer::trainLayer(const float* input, const float* expected_output)
{
  float chunk = (float)neuron_num_ / (thread_num_ + 1);

  shared_error_sum_ = 0;

  for (int i = 0; i < thread_num_; i++)
    threads_[i] = std::thread(parallelTraining, this, thread_num_ + 1,
      ((i + 1) * chunk), ((i + 2) * chunk), input, expected_output);

  // Use master thread in the computations as well
  parallelTraining(this, thread_num_ + 1, 0, chunk, input, expected_output);

  for (int i = 0; i < thread_num_; i++)
    threads_[i].join();

  return shared_error_sum_;
}
