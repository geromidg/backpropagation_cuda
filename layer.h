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

    void processNeurons(const float* input);
    float fitNeurons(const float* input, const float& next_layer_error);
    float fitNeurons(const float* input, const float* expected_output);

  private:
    int neuron_num_;

    Neuron** neurons_;
};

#endif  // LAYER_H
