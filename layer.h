#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer
{
  public:
    Layer(){};
    Layer(const int& input_num, const int& neuron_num);
    virtual ~Layer(void);

    void create(int inputsize, int _neuron_num);
    void calculate();

    int input_num;
    int neuron_num;

    Neuron** neurons;

    float* layerinput;
};

#endif  // LAYER_H
