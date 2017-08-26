#ifndef TRAINING_H
#define TRAINING_H

#include "layer.h"

void parallelTraining(Layer* layer, const int& start_chunk, const int& end_chunk,
  const float* input, const float* expected_output);

#endif  // TRAINING_H
