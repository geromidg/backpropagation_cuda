TARGET = parallel_neural_training

CC = nvcc
CC += -std=c++11
C_FLAGS = -Wall -O3 
CUDA_FLAGS = --default-stream per-thread -x cu

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
OBJECTS += $(patsubst %.cu, %.o, $(wildcard *.cu))
HEADERS = $(wildcard *.h *.cuh)

%.o: %.cpp $(HEADERS)
	$(CC) --compiler-options $(C_FLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(CC) $(CUDA_FLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
