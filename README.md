# Convolutional neural network

This repository contains a simple C++ implementation of a convolutional neural
network. It is based on the explanation and examples provided in the
[Neural Networks and Deep Learning online book](http://neuralnetworksanddeeplearning.com/).

Build requirements:
 - CMake
 - Boost
 - Threading Building Blocks

Example steps to build and run, from the repository source directory:
```
$ mkdir Release
$ cd Release
$ cmake .. -DCMAKE_BUILD_TYPE=Release
...
$ make -j8
$ ../get-mnist.sh
$ ./conv2
...
```

Features implemented:

- Stochastic gradient descent.
- Quadratic and cross entropy cost functions.
- Sigmoid and rectified-linear activation functions.
- Weight initialisation from normally-distributed random numbers.
- Minibatching.
- Regularisation.
- Fully-connected and soft-max layers.
- Convolutional and max-pooling layers.
- Convolutional feature maps.
- Use of a third data set for validation.
- Reporting of cost and accuracy during training.

There are three main source files:

- ``Network.hpp``, which contains the implementation of the network and each
  layer.
- ``Params.hpp``, a small wrapper class to encapsulate various hyperparameters.
- ``Data.hpp``, a class that loads the MNIST image data and creates data
  structures for consumption by the network.

There are three example programs:

- ``fc.cpp``, a network with a single fully-connected layer.
- ``conv1.cpp``, a network with one convolutional and one max-pooling layer.
- ``conv2.cpp``, a network with a stack of two convolutional and max-pooling
  layers.
