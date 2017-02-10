# A convolutional neural network from scratch

This repository contains a simple C++ implementation of a convolutional neural
network. It is based on the explanation and examples provided in the
[Neural Networks and Deep Learning online book](http://neuralnetworksanddeeplearning.com/).
There are more details about the code and workings of convolutional networks on
[my webiste](http://www.jwhanlon.com/a-convolutional-neural-network-from-scratch.html).

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
=============================
Parameters
-----------------------------
Num threads       8
Num epochs        60
Minibatch size    10
Learning rate     0.03
Lambda            0.1
Seed              1486724639
Training images   60000
Testing images    10000
Validation images 0
Monitor interval  1000
=============================
Reading labels: train-labels-idx1-ubyte
Reading labels: t10k-labels-idx1-ubyte
Reading images: train-images-idx3-ubyte
Reading images: t10k-images-idx3-ubyte
Creating the network
Running...
Accuracy on test data: 975 / 10000
Accuracy on test data: 3625 / 10000
Accuracy on test data: 7285 / 10000
Accuracy on test data: 7839 / 10000
Accuracy on test data: 8029 / 10000
Accuracy on test data: 8303 / 10000
...
```

There are three main source files:

- ``Network.hpp``, which contains the implementation of the network and each
  layer.
- ``Params.hpp``, a small wrapper class to encapsulate various hyperparameters.
- ``Data.hpp``, a class that loads the MNIST image data and creates data
  structures for consumption by the network.

There are four example programs:

- ``fc.cpp``, a network with a single fully-connected layer.
- ``conv1.cpp``, a network with one convolutional and one max-pooling layer.
- ``conv2.cpp``, a network with a stack of two convolutional and max-pooling
  layers.
- ``conv3.cpp``, a network with a stack of four convolutional and a max-pooling
  layer.

Features implemented:

- Stochastic gradient descent.
- Quadratic and cross entropy cost functions.
- Sigmoid and rectified-linear activation functions.
- Minibatching.
- Regularisation.
- Fully-connected and soft-max layers.
- Convolutional and max-pooling layers.
- Convolutional feature maps.

Possible features that could be added:

- Padding in the convolutional layer to maintain the input size.
- Dropout to help prevent overfitting.
- ...
