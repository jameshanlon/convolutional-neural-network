# Convolutional neural network

This repository contains a simple C++ implementation of a convolutional neural
network. It is based on the explanation and examples provided in the Neural Networks
and Deep Learning online book (http://neuralnetworksanddeeplearning.com/).

Requirements:
 - CMake
 - Boost
 - Threading Building Blocks

Example steps to build and run, from the repository source directory:
```
$ mkdir Release
$ cd Release
$ cmake .. -DCMAKE_BUILD_TYPE=Release
...
$ make
$ ../get-mnist.sh
$ ./nn
...
```
