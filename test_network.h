#ifndef TEST_NETWORK_H
#define TEST_NETWORK_H

#include "network.h"

class TestNetwork
{
  public:
    TestNetwork(const int& network_layer_num,
      const int* network_configuration, const int& network_epochs,
      const float& network_gamma, const float& network_alpha,
      const int& dataset_size, float** dataset_input, float** dataset_output);
    virtual ~TestNetwork(void);

    void run(void);

  private:
    void train(void);
    void validate(void);

    Network* network_;

    int network_epochs_;
    int dataset_size_;
    float** dataset_input_;
    float** dataset_output_;
};

#endif  // TEST_NETWORK_H
