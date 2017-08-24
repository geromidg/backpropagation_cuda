#include <stdlib.h>

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

float Layer::fitNeurons(const float* input, const float& next_layer_error)
{
  float expected_output[neuron_num_];

  for (int i = 0; i < neuron_num_; i++)
    expected_output[i] = next_layer_error + neurons_[i]->getOutput();

  return fitNeurons(input, expected_output);
}

// TODO: Do part of the work as main thread!
float Layer::fitNeurons(const float* input, const float* expected_output)
{
  shared_error_sum_ = 0;

  for (int i = 0; i < thread_num_; ++i)
  {
    int start_block = (i * neuron_num_) / thread_num_;
    int end_block = ((i + 1) * neuron_num_) / thread_num_;

    threads_[i] = std::thread(&Layer::fitNeuronsThreaded, this,
      start_block, end_block, input, expected_output);
  }

  for (int i = 0; i < thread_num_; ++i)
    threads_[i].join();

  return shared_error_sum_;
}

void Layer::fitNeuronsThreaded(const int& start_block, const int& end_block,
  const float* input, const float* expected_output)
{
  float error_sum = 0;

  for (int i = start_block; i < end_block; i++)
    error_sum += neurons_[i]->fit(input, expected_output[i]);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    shared_error_sum_ += error_sum;
  }
}
