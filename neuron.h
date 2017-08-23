#ifndef NEURON_H
#define NEURON_H

class Neuron
{
  public:
    Neuron(const int& input_num, const float& gamma, const float& alpha);
    virtual ~Neuron(void);

    float getOutput(void);

    void process(const float* inputs);
    float fit(const float* input, const float& expected_output);

  private:
    float activationFunction(const float& weighted_sum);

    int input_num_;

    float* weights_;
    float bias_;
    float output_;

    float* deltas_;
    float gamma_;  // learning rate
    float alpha_;  // learning momentum
};

#endif  // NEURON_H
