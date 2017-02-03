#include <iostream>
#include "tbb/tbb.h"
#include "Data.hpp"
#include "Params.hpp"
#include "Network.hpp"

int main(void) {
  tbb::task_scheduler_init init;
  std::cout << "Num threads: " << init.default_num_threads() << "\n";
  constexpr unsigned mbSize = 10;
  Params params;
  params.numEpochs = 60;
  params.learningRate = 0.03f;
  params.lambda = 0.1f;
  params.numValidationImages = 0;
  params.numTrainingImages = 60000;
  params.numTestImages = 10000;
  params.monitorTrainingAccuracy = true;
  params.dump(mbSize);
  // Read the MNIST data.
  Data data(params);
  // Create the network.
  std::cout << "Creating the network\n";
  Network<mbSize, 28, 28, 10, 100,
          CrossEntropyCost::compute,
          CrossEntropyCost::delta> network(params, {
      new ConvLayer<mbSize, 5, 5, 1, 28, 28, 1, 20,
                    ReLU::compute, ReLU::deriv>(params),
      new MaxPoolLayer<mbSize, 2, 2, 24, 24, 20>(),
      new FullyConnectedLayer<mbSize, 100, 12*12*20,
                              Sigmoid::compute,
                              Sigmoid::deriv>(params)});
  // Run it.
  std::cout << "Running...\n";
  network.SGD(data);
  return 0;
}
