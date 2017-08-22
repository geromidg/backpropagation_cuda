#include <stdlib.h>

#include "layer.h"

Layer::Layer(const int& input_num, const int& neuron_num):
  input_num(input_num),
  neuron_num(neuron_num)
{
  neurons = new Neuron*[neuron_num];

  for (int i = 0; i < neuron_num;i++)
    neurons[i] = new Neuron(input_num);

  layerinput = new float[input_num];
}

Layer::~Layer(void)
{
  if (neurons)
  {
    for(int i = 0; i < neuron_num; i++)
      delete neurons[i];

    delete[] neurons;
  }

  if (layerinput)
    delete[] layerinput;
}

void Layer::calculate()
{
  for (int i = 0; i < neuron_num; i++)
    neurons[i]->calculateOutput(layerinput);
}
