#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"

static timeval elapsed_time;

void tic(void)
{
  gettimeofday(&elapsed_time, NULL);
}

void toc(const char* message)
{
  timeval current_time;

  gettimeofday(&current_time, NULL);
  float diff = (current_time.tv_usec - elapsed_time.tv_usec) / 1.0e6 + (current_time.tv_sec - elapsed_time.tv_sec);

  printf(message, diff);
}
