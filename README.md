Parallel Neural Training
========================

This is an application that trains, runs and validates a **neural network on GPU**, given a dataset.<br>
The training of the network is done using the **backpropagation algorithm**.<br>
The parallelization is done using a mix of CUDA, Pthreads and OMP.<br>

The program runs on a machine with CUDA 7+ installed.<br>
To execute it, run:<br>
`$ make`<br>
`$ ./parallel_neural_training`