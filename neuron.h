#ifndef NEURON_H
#define NEURON_H

class Neuron
{
  public:
    Neuron(const int& input_num, const float& gamma, const float& alpha);
    virtual ~Neuron(void);

    float getOutput(void);

    void process(const float* inputs);

  private:
    float activationFunction(const float& weighted_sum);

  public:
    int input_num_;

    float* weights_;
    float bias_;
    float output_;

    float error_;
    float* deltas_;
    float gamma_;  // learning rate
    float alpha_;  // learning momentum
};

#endif  // NEURON_H
