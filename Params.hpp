#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <iostream>

struct Params {
  unsigned  numEpochs;
  float     learningRate;
  float     lambda;
  unsigned  seed = 1;
  unsigned  numValidationImages;
  unsigned  numTrainingImages;
  unsigned  numTestImages;
  bool      monitorEvaluationAccuracy = false;
  bool      monitorEvaluationCost     = false;
  bool      monitorTrainingAccuracy   = false;
  bool      monitorTrainingCost       = false;
  void dump(unsigned mbSize /* mbSize is a template param */,
            unsigned numThreads /* returned by TBB object */) {
    std::cout << "=============================\n";
    std::cout << "Parameters\n";
    std::cout << "-----------------------------\n";
    std::cout << "Num threads       " << numThreads << "\n";
    std::cout << "Num epochs        " << numEpochs << "\n";
    std::cout << "Minibatch size    " << mbSize << "\n";
    std::cout << "Learning rate     " << learningRate << "\n";
    std::cout << "Lambda            " << lambda << "\n";
    std::cout << "Seed              " << seed << "\n";
    std::cout << "Training images   " << numTrainingImages << "\n";
    std::cout << "Testing images    " << numTestImages << "\n";
    std::cout << "Validation images " << numValidationImages << "\n";
    std::cout << "=============================\n";
  }
};

#endif
