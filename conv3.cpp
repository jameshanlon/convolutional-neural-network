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
  Network<mbSize, 28, 28, 10, 4*4*10,
          CrossEntropyCost::compute,
          CrossEntropyCost::delta> network(params, {
      new ConvLayer<mbSize, 5, 5, 1, 28, 28, 1, 2,
                    ReLU::compute, ReLU::deriv>(params),
      new ConvLayer<mbSize, 5, 5, 2, 24, 24, 2, 2,
                    ReLU::compute, ReLU::deriv>(params),
      new ConvLayer<mbSize, 5, 5, 2, 20, 20, 2, 2,
                    ReLU::compute, ReLU::deriv>(params),
      new MaxPoolLayer<mbSize, 2, 2, 16, 16, 2>(),
      new ConvLayer<mbSize, 5, 5, 2, 8, 8, 2, 10,
                    Sigmoid::compute,
                    Sigmoid::deriv>(params)});
  // Run it.
  std::cout << "Running...\n";
  network.SGD(data);
  return 0;
}
