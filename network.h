#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

class Network
{
  public:
    Network(const int& total_layer_num, const int* configuration, const float& gamma, const float& alpha);
    virtual ~Network(void);

    float* getNetworkOutput(void);  // FIXME: Make return value const
    void propagateInputValues(const float *input);

    void trainNetwork(const float *expected_output, const float *input);
        
  private:
    int layer_num_;

    Layer** layers_;
};

#endif  // NETWORK_H
