#ifndef NEURON_H
#define NEURON_H

class Neuron
{
  public:
    Neuron(const int& input_num, const float& gamma, const float& alpha);
    virtual ~Neuron(void);

    void computeOutput(const float* inputs);
    float fitWeights(const float* input_values, const float& expected_output);

    float* weights;
    float bias;
    float output;

    float* deltas;

    float gamma;  // learning rate
    float alpha;  // learning momentum

  private:
    float activationFunction(const float& weighted_sum);

    int input_num;

  private:

};

#endif  // NEURON_H
