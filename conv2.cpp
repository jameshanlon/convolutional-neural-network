#include <chrono>
#include <iostream>
#include "tbb/tbb.h"
#include "Data.hpp"
#include "Params.hpp"
#include "Network.hpp"

int main(void) {
  tbb::task_scheduler_init init;
  constexpr unsigned mbSize = 10;
  Params params;
  params.numEpochs = 60;
  params.learningRate = 0.03f;
  params.lambda = 0.1f;
  params.seed = std::time(nullptr);
  params.numValidationImages = 0;
  params.numTrainingImages = 60000;
  params.numTestImages = 10000;
  params.monitorTrainingAccuracy = true;
  params.dump(mbSize, init.default_num_threads());
  // Read the MNIST data.
  Data data(params);
  // Create the network.
  std::cout << "Creating the network\n";
  constexpr unsigned conv1FMs = 8;
  constexpr unsigned conv2FMs = 8;
  constexpr unsigned fcSize = 100;
  Network<mbSize, 28, 28, 10, fcSize,
          CrossEntropyCost::compute,
          CrossEntropyCost::delta> network(params, {
      new ConvLayer<mbSize, 5, 5, 1, 28, 28, 1, conv1FMs,
                    ReLU::compute, ReLU::deriv>(params),
      new MaxPoolLayer<mbSize, 2, 2, 24, 24, conv1FMs>(),
      new ConvLayer<mbSize, 5, 5, conv1FMs, 12, 12, conv1FMs, conv2FMs,
                    ReLU::compute, ReLU::deriv>(params),
      new MaxPoolLayer<mbSize, 2, 2, 8, 8, conv2FMs>(),
      new FullyConnectedLayer<mbSize, fcSize, 4*4*conv2FMs,
                              Sigmoid::compute,
                              Sigmoid::deriv>(params)});
  // Run it.
  std::cout << "Running...\n";
  network.SGD(data);
  return 0;
}
