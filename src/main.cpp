#include <stdlib.h>

#include "test_network.h"

#include "network_configuration.h"
#include "dataset_logic_gate.h"

static void create_dataset(float**& dataset_input, float**& dataset_output);
static void delete_dataset(float**& dataset_input, float**& dataset_output);

void create_dataset(float**& dataset_input, float**& dataset_output)
{
  float macro_input[DATASET_SIZE][DATASET_INPUT_SIZE] = DATASET_INPUT;
  float macro_output[DATASET_SIZE][DATASET_OUTPUT_SIZE] = DATASET_OUTPUT;

  dataset_input = new float*[DATASET_SIZE];
  dataset_output = new float*[DATASET_SIZE];

  for (int i = 0; i < DATASET_SIZE; ++i)
  {
    dataset_input[i] = new float[DATASET_INPUT_SIZE];
    dataset_output[i] = new float[DATASET_OUTPUT_SIZE];

    for (int j = 0; j < DATASET_INPUT_SIZE; ++j)
      dataset_input[i][j] = macro_input[i][j];

    for (int j = 0; j < DATASET_OUTPUT_SIZE; ++j)
      dataset_output[i][j] = macro_output[i][j];
  }
}

void delete_dataset(float**& dataset_input, float**& dataset_output)
{
  if (dataset_input)
  {
    for (int i = 0; i < DATASET_SIZE; i++)
      delete dataset_input[i];

    delete[] dataset_input;
  }
  
  if (dataset_output)
  {
    for (int i = 0; i < DATASET_SIZE; i++)
      delete dataset_output[i];

    delete[] dataset_output;
  }
}

int main(int argc, char** argv)
{
  int network_configuration[NETWORK_LAYER_NUM] = NETWORK_CONFIGURATION;
  float** dataset_input;
  float** dataset_output;

  create_dataset(dataset_input, dataset_output);

  TestNetwork test_network = TestNetwork(NETWORK_LAYER_NUM, network_configuration,
    NETWORK_EPOCHS, NETWORK_GAMMA, NETWORK_ALPHA,
    DATASET_SIZE, dataset_input, dataset_output,
    THREAD_NUM);
  test_network.run();

  delete_dataset(dataset_input, dataset_output);

  return 0;
}
