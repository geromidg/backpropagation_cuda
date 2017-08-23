#ifndef DATASET_LOGIC_GATE_H
#define DATASET_LOGIC_GATE_H

// Dataset for XOR->AND->OR function

#define DATASET_SIZE (16)
#define DATASET_INPUT_SIZE (4)
#define DATASET_OUTPUT_SIZE (1)

#define DATASET_INPUT { \
{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1}, \
{0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}, \
{1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1}, \
{1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}}

#define DATASET_OUTPUT { \
{0}, {0}, {0}, {0}, \
{0}, {1}, {1}, {0}, \
{1}, {1}, {1}, {1}, \
{1}, {1}, {1}, {1}}

#endif  // DATASET_LOGIC_GATE_H
