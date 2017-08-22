#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer
{
  public:
    Layer(){};
    Layer(const int& input_num, const int& neuron_num, const float& gamma, const float& alpha);
    virtual ~Layer(void);

    void calculateNeuronOutputs();
    float computeNewWeights(const float& next_layer_error);
    float computeNewWeights(const float* desiredoutput);

    int input_num;
    int neuron_num;

    Neuron** neurons;

    float gamma;  // learning rate
    float alpha;  // learning momentum

    float* layerinput;  // FIXME!
};

#endif  // LAYER_H
