#ifndef LAYER_H
#define LAYER_H

#include <thread>
#include <mutex>

#include "neuron.h"

class Layer
{
  public:
    Layer(){};
    Layer(const int& input_num, const int& neuron_num,
      const float& gamma, const float& alpha,
      const int& thread_num);
    virtual ~Layer(void);

    float* getLayerOutput(void);

    void processNeurons(const float* input);
    float trainLayer(const float* input, const float& next_layer_error);
    float trainLayer(const float* input, const float* expected_output);
    
    Neuron** neurons_;
    std::mutex mutex_;
    float shared_error_sum_;

  private:
    int neuron_num_;

    int thread_num_;
    std::thread* threads_;
};

#endif  // LAYER_H
