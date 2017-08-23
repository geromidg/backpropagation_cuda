#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer
{
  public:
    Layer(){};
    Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha);
    virtual ~Layer(void);

    float* getLayerOutput(void);  // FIXME: Make return value const
    void propagateInput(const float* input_values);

    float trainLayer(const float* input_values, const float& next_layer_error);
    float trainLayer(const float* input_values, const float* expected_output);

  private:
    int neuron_num_;

    Neuron** neurons_;
};

#endif  // LAYER_H
