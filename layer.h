#ifndef LAYER_H
#define LAYER_H

#include <thread>
#include <mutex>

#include "neuron.h"

class Layer
{
  public:
    Layer(){};
    Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha);
    virtual ~Layer(void);

    float* getLayerOutput(void);  // FIXME: Make return value const

    void processNeurons(const float* input);
    float fitNeurons(const float* input, const float& next_layer_error);
    float fitNeurons(const float* input, const float* expected_output);

  private:
    void fitNeuronsThreaded(const int& start_block, const int& end_block,
      const float* input, const float* expected_output);

    int neuron_num_;
    Neuron** neurons_;

    int threads_num_;
    std::thread* threads_;
    std::mutex mutex_;
    float shared_error_sum_;
};

#endif  // LAYER_H
