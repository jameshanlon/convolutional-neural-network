#include <iostream>
#include <chrono>
#include "tbb/tbb.h"
#include "Data.hpp"
#include "Params.hpp"
#include "Network.hpp"

int main(void) {
  tbb::task_scheduler_init init;
  std::cout << "Num threads: " << init.default_num_threads() << "\n";
  constexpr unsigned mbSize = 10;
  Params params;
  params.numEpochs = 200;
  params.learningRate = 0.5f;
  params.lambda = 0.1f;
  params.seed = std::time(nullptr);
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
      new FullyConnectedLayer<mbSize, 100, 28 * 28,
                              Sigmoid::compute,
                              Sigmoid::deriv>(params)});
  // Run it.
  std::cout << "Running...\n";
  network.SGD(data);
  return 0;
}
