#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

class Network
{
  public:
    Network(const int& total_layer_num, const int* configuration,
      const float& gamma, const float& alpha,
      const int& thread_num);
    virtual ~Network(void);

    float* getNetworkOutput(void);

    void propagate(const float *input);
    void train(const float *input, const float *expected_output);
        
  private:
    int layer_num_;

    Layer** layers_;
};

#endif  // NETWORK_H
