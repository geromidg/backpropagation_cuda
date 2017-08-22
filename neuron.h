#ifndef NEURON_H
#define NEURON_H

class Neuron
{
  public:
    Neuron(const int& input_num);
    virtual ~Neuron(void);

    void calculateOutput(const float* inputs);

    float* weights;
    float bias;
    float output;

    float* deltas;

  private:
    float activationFunction(const float& weighted_sum);

    int input_num_;

  private:

};

#endif  // NEURON_H
