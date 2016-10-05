#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>
#include <random>
#include <vector>

using Image = std::vector<float>;

const unsigned numEpochs = 100;
const unsigned mbSize = 10;
const float learningRate = 1.0f;
const float lambda = 5.0f;
const unsigned validationSize = 1000;
const bool monitorEvaluationAccuracy = false;
const bool monitorEvaluationCost = false;
const bool monitorTrainingAccuracy = true;
const bool monitorTrainingCost = true;

static void readLabels(const char *filename,
                       std::vector<uint8_t> &labels) {
  std::ifstream file;
  file.open(filename, std::ios::binary | std::ios::in);
  if (!file.good()) {
    std::cout << "Error opening file " << filename << '\n';
    std::exit(1);
  }
  int32_t magicNumber, numItems;
  file.read(reinterpret_cast<char*>(&magicNumber), 4);
  file.read(reinterpret_cast<char*>(&numItems), 4);
  magicNumber = __builtin_bswap32(magicNumber);
  numItems = __builtin_bswap32(numItems);
  std::cout << "Magic number: " << magicNumber << "\n";
  std::cout << "Num items:    " << numItems << "\n";
  for (unsigned i = 0; i < numItems; ++i) {
    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);
    labels.push_back(label);
  }
  file.close();
}

static void readImages(const char *filename,
                       std::vector<Image> &images) {
  std::ifstream file;
  file.open(filename, std::ios::binary | std::ios::in);
  if (!file.good()) {
    std::cout << "Error opening file " << filename << '\n';
    std::exit(1);
  }
  uint32_t magicNumber, numImages, numRows, numCols;
  file.read(reinterpret_cast<char*>(&magicNumber), 4);
  file.read(reinterpret_cast<char*>(&numImages), 4);
  file.read(reinterpret_cast<char*>(&numRows), 4);
  file.read(reinterpret_cast<char*>(&numCols), 4);
  magicNumber = __builtin_bswap32(magicNumber);
  numImages = __builtin_bswap32(numImages);
  numRows = __builtin_bswap32(numRows);
  numCols = __builtin_bswap32(numCols);
  std::cout << "Magic number: " << magicNumber << "\n";
  std::cout << "Num images:   " << numImages << "\n";
  std::cout << "Num rows:     " << numRows << "\n";
  std::cout << "Num cols:     " << numCols << "\n";
  assert(numRows == 28 && numCols == 28 && "unexpected image size");
  for (unsigned i = 0; i < numImages; ++i) {
    Image image(numRows*numCols);
    for (unsigned j = 0; j < numRows; ++j) {
      for (unsigned k = 0; k < numCols; ++k) {
        uint8_t pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        // Scale the pixel value to between 0 (white) and 1 (black).
        float value = static_cast<float>(pixel) / 255.0;
        image[(j*numRows)+k] = value;
      }
    }
    images.push_back(image);
  }
  file.close();
}

static float sigmoid(float z) {
  return 1.0f / (1.0f + std::exp(-z));
}

/// Derivative of the sigmoid function.
static float sigmoidDerivative(float z) {
  return sigmoid(z) * (1.0f - sigmoid(z));
}

struct QuadraticCost {
  static float compute(float activation, float label) {
    return 0.5f * std::pow(std::abs(activation - label), 2);
  }
  static float delta(float z, float activation, float label) {
    return (activation - label) * sigmoidDerivative(z);
  }
};

struct CrossEntropyCost {
  static float compute(float activation, float label) {
    return (-label * std::log(activation))
            - ((1.0f - label) * std::log(1.0f - activation));
  }
  static float delta(float z, float activation, float label) {
    return activation - label;
  }
};

template<
  int mbSize,
  float (*costFn)(float, float),
  float (*costDelta)(float, float, float)>
class Neuron;

template<
  int mbSize,
  float (*costFn)(float, float),
  float (*costDelta)(float, float, float)>
class Layer {
  using NeuronT = Neuron<mbSize, costFn, costDelta>;
public:
  virtual unsigned size() = 0;
  virtual NeuronT& getNeuron(unsigned index) = 0;
};

template<
  int mbSize,
  float (*costFn)(float, float),
  float (*costDelta)(float, float, float)>
class Neuron {
  using LayerT = Layer<mbSize, costFn, costDelta>;
  LayerT *inputs;
  LayerT *outputs;
  std::vector<float> weights;
  float bias;
  unsigned index;
  // Per batch.
  float weightedInputs[mbSize];
  float activations[mbSize];
  float errors[mbSize];

public:
  Neuron(unsigned index) : index(index) {}

  void initialiseDefaultWeights() {
    // Initialise all weights with random values from normal distribution with
    // mean 0 and stdandard deviation 1, divided by the square root of the
    // number of input connections.
    static std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0, 1.0);
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float weight = distribution(generator) / std::sqrt(inputs->size());
      weights.push_back(weight);
    }
    bias = distribution(generator);
  }

  void initialiseLargeWeights() {
    // Just using random numbers as above.
    static std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0, 1.0f);
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weights.push_back(distribution(generator));
    }
    bias = distribution(generator);
  }

  /// Compute the output error (only the output neurons).
  void computeOutputError(uint8_t label, unsigned mbIndex) {
    float y = label == index ? 1.0f : 0.0f;
    float error = costDelta(weightedInputs[mbIndex], activations[mbIndex], y);
    errors[mbIndex] = error;
  }

  /// Compute the output cost (only the output neurons).
  float computeOutputCost(uint8_t label, unsigned mbIndex) {
    return costFn(activations[mbIndex], label);
  }

  float sumSquaredWeights() {
    float result = 0.0f;
    for (auto weight : weights) {
      result += std::pow(weight, 2.0f);
    }
    return result;
  }

  void forwardBatch(unsigned mbIndex) {
    float weightedInput = 0.0f;
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weightedInput += inputs->getNeuron(i).getOutput(mbIndex) * weights[i];
    }
    weightedInput += bias;
    weightedInputs[mbIndex] = weightedInput;
    activations[mbIndex] = sigmoid(weightedInput);
  }

  void backwardBatch(unsigned mbIndex) {
    float error = 0.0f;
    for (unsigned i = 0; i < outputs->size(); ++i) {
      auto &neuron = outputs->getNeuron(i);
      error += neuron.getWeight(index) * neuron.getError(mbIndex);
    }
    error *= sigmoidDerivative(weightedInputs[mbIndex]);
    errors[mbIndex] = error;
  }

  void endBatch(float learningRate, float lambda, unsigned numTrainingImages) {
    // For each weight.
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float weightDelta = 0.0f;
      // For each batch element, average input activation x error (rate of
      // change of cost w.r.t. weight) and multiply by learning rate.
      for (unsigned j = 0; j < mbSize; ++j) {
        weightDelta += inputs->getNeuron(i).getOutput(j) * errors[j];
      }
      weightDelta *= learningRate / mbSize;
      weights[i] *= 1.0f - (learningRate * (lambda / numTrainingImages));
      weights[i] -= weightDelta;
    }
    // For each batch element, average the errors (error is equal to rate of
    // change of cost w.r.t. bias) and multiply by learning rate.
    float biasDelta = 0.0f;
    for (unsigned j = 0; j < mbSize; ++j) {
      biasDelta += errors[j];
    }
    biasDelta *= learningRate / mbSize;
    bias -= biasDelta;
  }

  void setInputs(LayerT *inputs) { this->inputs = inputs;  }
  void setOutputs(LayerT *outputs) { this->outputs = outputs; }
  void setOutput(unsigned i, float value) { activations[i] = value; }
  unsigned numWeights() { return weights.size(); }
  float getOutput(unsigned mbIndex) { return activations[mbIndex]; }
  float getError(unsigned mbIndex) { return errors[mbIndex]; }
  float getWeight(unsigned i) { return weights.at(i); }
};

template<
  int mbSize,
  float (*costFn)(float, float),
  float (*costDelta)(float, float, float)>
class FullyConnectedLayer : public Layer<mbSize, costFn, costDelta> {
  using LayerT = Layer<mbSize, costFn, costDelta>;
  using NeuronT = Neuron<mbSize, costFn, costDelta>;
  std::vector<NeuronT> neurons;

public:
  FullyConnectedLayer(unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
      neurons.push_back(NeuronT(i));
    }
  };

  /// Set the input image (for the input layer only).
  void setImage(Image &image, unsigned mbIndex) {
    assert(image.size() == neurons.size() && "Invalid layer size for input");
    for (unsigned i = 0; i < neurons.size(); ++i) {
      neurons[i].setOutput(mbIndex, image[i]);
    }
  }

  /// Determine the index of the highest output activation. Inference only.
  unsigned readOutput() {
    unsigned result = 0;
    float max = 0.0f;
    for (unsigned i = 0; i < neurons.size(); ++i) {
      float output = neurons[i].getOutput(0);
      if (output > max) {
        result = i;
        max = output;
      }
    }
    return result;
  }

  float sumSquaredWeights() {
    float result = 0.0f;
    for (auto &neuron : neurons) {
      result += neuron.sumSquaredWeights();
    }
    return result;
  }

  void setInputs(LayerT *layer) {
    for (auto &neuron : neurons) {
      neuron.setInputs(layer);
    }
  }

  void setOutputs(LayerT *layer) {
    for (auto &neuron : neurons) {
      neuron.setOutputs(layer);
    }
  }

  void initialiseDefaultWeights() {
    for (auto &neuron : neurons) {
      neuron.initialiseDefaultWeights();
    }
  }

  void computeOutputError(uint8_t label, unsigned mbIndex) {
    for (auto &neuron : neurons) {
      neuron.computeOutputError(label, mbIndex);
    }
  }

  float computeOutputCost(uint8_t label, unsigned mbIndex) {
    float outputCost = 0.0f;
    for (auto &neuron : neurons) {
      neuron.computeOutputCost(label, mbIndex);
    }
    return outputCost;
  }

  void feedForward(Image &image, unsigned mbIndex) {
    for (auto &neuron : neurons) {
      neuron.forwardBatch(mbIndex);
    }
  }

  void backPropogate(unsigned mbIndex) {
    for (auto &neuron : neurons) {
      neuron.backwardBatch(mbIndex);
    }
  }

  void endBatch(float learningRate, float lambda, unsigned numTrainingImages) {
    for (auto &neuron : neurons) {
      neuron.endBatch(learningRate, lambda, numTrainingImages);
    }
  }

  NeuronT &getNeuron(unsigned index) override {
    return neurons[index];
  }

  unsigned size() override { return neurons.size(); }
};

template<
  int mbSize,
  float (*costFn)(float, float),
  float (*costDelta)(float, float, float)
>
class Network {
  using FullyConnectedLayerT = FullyConnectedLayer<mbSize, costFn, costDelta>;
  std::vector<FullyConnectedLayerT> layers;

  /// Set the activations of the input neurons with an image.
  void setImage(Image &image, unsigned mbIndex) {
    layers.front().setImage(image, mbIndex);
  }

  /// Determine the index of the highest output activation.
  unsigned readOutput() {
    return layers.back().readOutput();
  }

  /// The forward pass.
  void feedForward(Image &image, unsigned mbIndex) {
    for (unsigned i = 1; i < layers.size(); ++i) {
      layers[i].feedForward(image, mbIndex);
    }
  }

  /// The backward pass.
  void backPropogate(Image &image, uint8_t label, unsigned mbIndex) {
    // Set input.
    setImage(image, mbIndex);
    // Feed forward.
    feedForward(image, mbIndex);
    // Compute output error in last layer.
    layers.back().computeOutputError(label, mbIndex);
    // Backpropagate the error.
    for (unsigned i = layers.size() - 2; i > 0; --i) {
      layers[i].backPropogate(mbIndex);
    }
  }

  void updateMiniBatch(std::vector<Image>::iterator trainingImagesIt,
                       std::vector<uint8_t>::iterator trainingLabelsIt,
                       float learningRate,
                       float lambda,
                       unsigned numTrainingImages) {
    // For each training image and label, backPropogateogate.
    for (unsigned i = 0; i < mbSize; ++i) {
      backPropogate(*(trainingImagesIt + i), *(trainingLabelsIt + i), i);
    }
    // Gradient descent: for every neuron, compute the new weights and biases.
    for (unsigned i = layers.size() - 1; i > 0; --i) {
      layers[i].endBatch(learningRate, lambda, numTrainingImages);
    }
  }

public:
  Network(const std::vector<unsigned> sizes) {
    // Create the input layer.
    layers.push_back(FullyConnectedLayerT(sizes[0]));
    // For each remaining layer.
    for (unsigned i = 1; i < sizes.size(); ++i) {
      layers.push_back(FullyConnectedLayerT(sizes[i]));
    }
    // Set neuron inputs.
    for (unsigned i = 1; i < layers.size(); ++i) {
      layers[i].setInputs(&layers[i - 1]);
      layers[i].initialiseDefaultWeights();
    }
    // Set neuron outputs.
    for (unsigned i = 0; i < layers.size() - 1; ++i) {
      layers[i].setOutputs(&layers[i + 1]);
    }
  }

  /// Evaluate the test set and return the number of correct classifications.
  unsigned evaluateAccuracy(std::vector<Image> &testImages,
                            std::vector<uint8_t> &testLabels) {
    unsigned result = 0;
    for (unsigned i = 0; i < testImages.size(); ++i) {
      //std::cout << "\rTest image " << i;
      setImage(testImages[i], 0);
      feedForward(testImages[i], 0);
      if (readOutput() == testLabels[i]) {
        ++result;
      }
    }
    //std::cout << '\n';
    return result;
  }

  float sumSquareWeights() {
    float result = 0.0f;
    for (unsigned i = 1; i < layers.size(); ++i) {
      result += layers[i].sumSquaredWeights();
    }
    return result;
  }

  /// Calculate the total cost for a dataset.
  float evaluateTotalCost(std::vector<Image> &images,
                          std::vector<uint8_t> &labels,
                          float lambda) {
    float cost = 0.0f;
    for (unsigned i = 0; i < images.size(); ++i) {
      setImage(images[i], 0);
      feedForward(images[i], 0);
      cost += layers.back().computeOutputCost(labels[i], 0) / images.size();
      // Add the regularisation term.
      cost += 0.5f * (lambda / images.size()) * sumSquareWeights();
    }
    return cost;
  }

  void SGD(std::vector<Image> &trainingImages,
           std::vector<uint8_t> &trainingLabels,
           std::vector<Image> &validationImages,
           std::vector<uint8_t> &validationLabels,
           std::vector<Image> &testImages,
           std::vector<uint8_t> &testLabels,
           unsigned numEpochs,
           float learningRate,
           float lambda) {
    // For each epoch.
    for (unsigned epoch = 0; epoch < numEpochs; ++epoch) {
      // Identically randomly shuffle the training images and labels.
      unsigned seed = std::time(nullptr);
      std::shuffle(trainingLabels.begin(), trainingLabels.end(),
                   std::default_random_engine(seed));
      std::shuffle(trainingImages.begin(), trainingImages.end(),
                   std::default_random_engine(seed));
      // For each mini batch.
      for (unsigned i = 0, end = trainingImages.size(); i < end; i += mbSize) {
        //std::cout << "\rUpdate minibatch: " << i << " / " << end;
        updateMiniBatch(trainingImages.begin() + i,
                        trainingLabels.begin() + i,
                        learningRate,
                        lambda,
                        trainingImages.size());
      }
      //std::cout << '\n';
      // Evaluate the test set.
      std::cout << "Epoch " << epoch << " complete.\n";
      if (monitorEvaluationAccuracy) {
        unsigned result = evaluateAccuracy(validationImages, validationLabels);
        std::cout << "Accuracy on evaluation data: "
                  << result << " / " << validationImages.size() << '\n';
      }
      if (monitorEvaluationCost) {
        float cost = evaluateTotalCost(validationImages, validationLabels,
                                       lambda);
        std::cout << "Cost on evaluation data: " << cost << "\n";
      }
      if (monitorTrainingAccuracy) {
        unsigned result = evaluateAccuracy(testImages, testLabels);
        std::cout << "Accuracy on training data: "
                  << result << " / " << testImages.size() << '\n';
      }
      if (monitorTrainingCost) {
        float cost = evaluateTotalCost(testImages, testLabels, lambda);
        std::cout << "Cost on test data: " << cost << "\n";
      }
    }
  }
};

int main(int argc, char **argv) {

  // Read the MNIST data.
  std::vector<uint8_t> trainingLabels;
  std::vector<uint8_t> testLabels;
  std::vector<Image> trainingImages;
  std::vector<Image> testImages;

  std::cout << "Reading labels\n";
  readLabels("train-labels-idx1-ubyte", trainingLabels);
  readLabels("t10k-labels-idx1-ubyte", testLabels);

  std::cout << "Reading images\n";
  readImages("train-images-idx3-ubyte", trainingImages);
  readImages("t10k-images-idx3-ubyte", testImages);

  // Take images from the training set for validation.
  std::vector<uint8_t> validationLabels(trainingLabels.end() - validationSize,
                                        trainingLabels.end());
  std::vector<Image> validationImages(trainingImages.end() - validationSize,
                                      trainingImages.end());
  trainingLabels.erase(trainingLabels.end() - validationSize,
                       trainingLabels.end());
  trainingImages.erase(trainingImages.end() - validationSize,
                       trainingImages.end());

  std::cout << "Creating the network\n";
  Network<mbSize, CrossEntropyCost::compute, CrossEntropyCost::delta>
    network({28*28, 100, 10});
  std::cout << "Running...\n";
  network.SGD(trainingImages,
              trainingLabels,
              validationImages,
              validationLabels,
              testImages,
              testLabels,
              numEpochs,
              learningRate,
              lambda);
  return 0;
}
