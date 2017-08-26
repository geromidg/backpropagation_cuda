#include <cuda_runtime.h>

#include "training.cuh"

static void updateNeuronErrorsLaunch(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* expected_output);
static void updateNeuronDeltasLaunch(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* input);
static float sumAllErrors(const int& start_chunk, const int& end_chunk, Neuron** neurons);

__global__ void updateNeuronErrors(const int neuron_num, const float* expected_output,
  const float* outputs, const float* gammas, float* errors, float* biases)
{
  float output;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < neuron_num; i += stride)
  {
    output = outputs[i];

    errors[i] = output * (1 - output) * (expected_output[i] - output);
    biases[i] += gammas[i] * errors[i];
  }
}

__global__ void updateNeuronDeltas(const int input_num, float* weights, float* deltas,
  const float error, const float* input, const float alpha, const float gamma)
{
  float delta;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < input_num; i += stride)
  {
    delta = deltas[i] * alpha + gamma * error * input[i];

    deltas[i] = delta;
    weights[i] += delta;
  }
}

void updateNeuronErrorsLaunch(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* expected_output)
{
  int gridSize;
  int minGridSize = 0;
  int blockSize = 0;
  int neuron_num = end_chunk - start_chunk;

  float* expected_output_d;
  float* outputs_d;
  float* errors_d;
  float* biases_d;
  float* gammas_d;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateNeuronErrors, 0, 0);
  gridSize = (neuron_num + blockSize - 1) / blockSize;

  cudaMalloc(&expected_output_d, neuron_num * sizeof(float));
  cudaMemcpy(expected_output_d, expected_output, neuron_num * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&outputs_d, neuron_num * sizeof(float));
  cudaMalloc(&errors_d, neuron_num * sizeof(float));
  cudaMalloc(&biases_d, neuron_num * sizeof(float));
  cudaMalloc(&gammas_d, neuron_num * sizeof(float));

  for (int i = 0; i < neuron_num; i++)
  {
    cudaMemcpy(&(outputs_d[i]), &(layer->neurons_[start_chunk + i]->output_), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gammas_d[i]), &(layer->neurons_[start_chunk + i]->gamma_), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(errors_d[i]), &(layer->neurons_[start_chunk + i]->error_), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(biases_d[i]), &(layer->neurons_[start_chunk + i]->bias_), sizeof(float), cudaMemcpyHostToDevice);
  }

  updateNeuronErrors<<<gridSize, blockSize>>>(neuron_num, expected_output_d,
    outputs_d, gammas_d, errors_d, biases_d);

  for (int i = 0; i < neuron_num; i++)
  {
    cudaMemcpy(&(layer->neurons_[start_chunk + i]->error_), &(errors_d[i]), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(layer->neurons_[start_chunk + i]->bias_), &(biases_d[i]), sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(outputs_d);
  cudaFree(gammas_d);
  cudaFree(errors_d);
  cudaFree(biases_d);
  cudaFree(expected_output_d);
}

void updateNeuronDeltasLaunch(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* input)
{
  int gridSize;
  int minGridSize = 0;
  int blockSize = 0;
  int input_num = layer->neurons_[0]->input_num_;

  float* input_d;
  float* weights_d;
  float* deltas_d;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateNeuronDeltas, 0, 0);
  gridSize = (input_num + blockSize - 1) / blockSize;

  cudaMalloc(&input_d, input_num * sizeof(float));
  cudaMemcpy(input_d, input, input_num * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = start_chunk; i < end_chunk; i++)
  {
    cudaMalloc(&weights_d, input_num * sizeof(float));
    cudaMalloc(&deltas_d, input_num * sizeof(float));

    cudaMemcpy(weights_d, layer->neurons_[i]->weights_, input_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deltas_d, layer->neurons_[i]->deltas_, input_num * sizeof(float), cudaMemcpyHostToDevice);

    updateNeuronDeltas<<<gridSize, blockSize>>>(input_num, weights_d, deltas_d,
      layer->neurons_[i]->error_, input_d, layer->neurons_[i]->alpha_, layer->neurons_[i]->gamma_);

    cudaMemcpy(layer->neurons_[i]->weights_, weights_d, input_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(layer->neurons_[i]->deltas_, deltas_d, input_num * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(weights_d);
    cudaFree(deltas_d);
  }

  cudaFree(input_d);
}

// TODO: Use OMP to sum
float sumAllErrors(const int& start_chunk, const int& end_chunk, Neuron** neurons)
{
  int input_num = neurons[0]->input_num_;
  float sum;

  for (int i = start_chunk; i < end_chunk; i++)
    for (int j = 0; j < input_num; j++)
      sum += neurons[i]->error_ * neurons[i]->weights_[j];

  return sum;
}

void parallelTraining(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* input, const float* expected_output)
{
  float error_sum;

  updateNeuronErrorsLaunch(layer, start_chunk, end_chunk, expected_output);
  updateNeuronDeltasLaunch(layer, start_chunk, end_chunk, input);

  cudaStreamSynchronize(0);  // XXX: cudaDeviceSynchronize() ?

  error_sum = sumAllErrors(start_chunk, end_chunk, layer->neurons_);

  {
    std::lock_guard<std::mutex> lock(layer->mutex_);
    layer->shared_error_sum_ += error_sum;
  }
}