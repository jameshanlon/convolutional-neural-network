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

// Globals and constants.
const unsigned numEpochs = 100;
const unsigned mbSize = 10;
const float learningRate = 1.0f;
const float lambda = 5.0f;
const unsigned validationSize = 1000;
const bool monitorEvaluationAccuracy = false;
const bool monitorEvaluationCost = false;
const bool monitorTrainingAccuracy = true;
const bool monitorTrainingCost = true;
float (*costFn)(float, float) = CrossEntropyCost::compute;
float (*costDelta)(float, float, float) = CrossEntropyCost::delta;

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

class Neuron;

class Layer {
protected:
  std::vector<Neuron> neurons;
public:
  virtual void initialiseDefaultWeights() = 0;
  virtual void feedForward(Image &image, unsigned mbIndex) = 0;
  virtual void backPropogate(unsigned mbIndex) = 0;
  virtual void endBatch(unsigned numTrainingImages) = 0;
  virtual void computeOutputError(uint8_t label, unsigned mbIndex) = 0;
  virtual float computeOutputCost(uint8_t label, unsigned mbIndex) = 0;
  virtual float sumSquaredWeights() = 0;
  virtual void setInputs(Layer *layer) = 0;
  virtual void setOutputs(Layer *layer) = 0;
  virtual unsigned readOutput() = 0;
  Neuron &getNeuron(unsigned index) { return neurons.at(index); }
  unsigned size() { return neurons.size(); }
};

class Neuron {
  Layer *inputs;
  Layer *outputs;
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
    std::normal_distribution<float> distribution(0, 1.0f);
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float weight = distribution(generator) / std::sqrt(inputs->size());
      weights.push_back(weight);
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

  void endBatch(unsigned numTrainingImages) {
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

  void setInputs(Layer *inputs) { this->inputs = inputs; }
  void setOutputs(Layer *outputs) { this->outputs = outputs; }
  void setOutput(unsigned i, float value) { activations[i] = value; }
  unsigned numWeights() { return weights.size(); }
  float getOutput(unsigned mbIndex) { return activations[mbIndex]; }
  float getError(unsigned mbIndex) { return errors[mbIndex]; }
  float getWeight(unsigned i) { return weights.at(i); }
};

class InputLayer : public Layer {
public:
  InputLayer(unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
      neurons.push_back(Neuron(i));
    }
  }
  void setImage(Image &image, unsigned mbIndex) {
    assert(image.size() == neurons.size() && "Invalid layer size for input");
    for (unsigned i = 0; i < neurons.size(); ++i) {
      neurons[i].setOutput(mbIndex, image[i]);
    }
  }
  void initialiseDefaultWeights() {
    assert(0 && "invalid call");
  }
  void feedForward(Image &image, unsigned mbIndex) {
    assert(0 && "invalid call");
  }
  void backPropogate(unsigned mbIndex) {
    assert(0 && "invalid call");
  }
  void endBatch(unsigned numTrainingImages) {
    assert(0 && "invalid call");
  }
  void computeOutputError(uint8_t label, unsigned mbIndex) {
    assert(0 && "invalid call");
  }
  float computeOutputCost(uint8_t label, unsigned mbIndex) {
    assert(0 && "invalid call");
  }
  float sumSquaredWeights() {
    assert(0 && "invalid call");
  }
  void setInputs(Layer *layer) {
    assert(0 && "invalid call");
  }
  void setOutputs(Layer *layer) {
    assert(0 && "invalid call");
  }
  unsigned readOutput() {
    assert(0 && "invalid call");
  }
};

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
      neurons.push_back(Neuron(i));
    }
  }

  /// Determine the index of the highest output activation.
  unsigned readOutput() override {
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

  float sumSquaredWeights() override {
    float result = 0.0f;
    for (auto &neuron : neurons) {
      result += neuron.sumSquaredWeights();
    }
    return result;
  }

  void setInputs(Layer *layer) override {
    for (auto &neuron : neurons) {
      neuron.setInputs(layer);
    }
  }

  void setOutputs(Layer *layer) override {
    for (auto &neuron : neurons) {
      neuron.setOutputs(layer);
    }
  }

  void initialiseDefaultWeights() override {
    for (auto &neuron : neurons) {
      neuron.initialiseDefaultWeights();
    }
  }

  void feedForward(Image &image, unsigned mbIndex) override {
    for (auto &neuron : neurons) {
      neuron.forwardBatch(mbIndex);
    }
  }

  void backPropogate(unsigned mbIndex) override {
    for (auto &neuron : neurons) {
      neuron.backwardBatch(mbIndex);
    }
  }

  void endBatch(unsigned numTrainingImages) override {
    for (auto &neuron : neurons) {
      neuron.endBatch(numTrainingImages);
    }
  }

  void computeOutputError(uint8_t label, unsigned mbIndex) override {
    for (auto &neuron : neurons) {
      neuron.computeOutputError(label, mbIndex);
    }
  }

  float computeOutputCost(uint8_t label, unsigned mbIndex) override {
    float outputCost = 0.0f;
    for (auto &neuron : neurons) {
      neuron.computeOutputCost(label, mbIndex);
    }
    return outputCost;
  }
};

//template<int kernelWidth, int kernelHeight>
//class ConvLayer : public Layer {
//  float weights[kernelWidth][kernelHeight];
//  Layer *inputs;
//  Layer *outputs;
//
//public:
//  ConvLayer(unsigned size) {}
//
//  float sumSquaredWeights() {
//    float result = 0.0f;
//    for (unsigned i = 0; i < kernelWidth; ++i) {
//      for (unsigned j = 0; j < kernelHeight; ++j) {
//        result += std::pow(weights[i][j]);
//      }
//    }
//    return result;
//  }
//
//  void initialiseDefaultWeights() {
//    static std::default_random_engine generator(std::time(nullptr));
//    std::normal_distribution<float> distribution(0, 1.0);
//    for (unsigned i = 0; i < kernelWidth; ++i) {
//      for (unsigned j = 0; j < kernelHeight; ++j) {
//        weights[i][j] = distribution(generator) / std::sqrt(inputs->size());
//      }
//    }
//  }
//
//  void feedForward(Image &image, unsigned mbIndex) {
//  }
//
//  void backPropogate(unsigned mbIndex) {
//  }
//
//  void endBatch(unsigned numTrainingImages) {
//  }
//
//  void computeOutputError(uint8_t label, unsigned mbIndex) {
//  }
//
//  float computeOutputCost(uint8_t label, unsigned mbIndex) {
//    return 0.0f;
//  }
//
//  void setInputs(Layer *layer) { inputs = layer; }
//  void setOutputs(Layer *layer) { outputs = layer; }
//};
//
//class MaxPoolLayer : public Layer {
//public:
//  MaxPoolLayer() {}
//};

class Network {
  InputLayer inputLayer;
  std::vector<Layer*> layers;

  /// The forward pass.
  void feedForward(Image &image, unsigned mbIndex) {
    for (auto layer : layers) {
      layer->feedForward(image, mbIndex);
    }
  }

  /// The backward pass.
  void backPropogate(Image &image, uint8_t label, unsigned mbIndex) {
    // Set input.
    inputLayer.setImage(image, mbIndex);
    // Feed forward.
    feedForward(image, mbIndex);
    // Compute output error in last layer.
    layers.back()->computeOutputError(label, mbIndex);
    // Backpropagate the error.
    for (int i = layers.size() - 2; i >= 0; --i) {
      layers[i]->backPropogate(mbIndex);
    }
  }

  void updateMiniBatch(std::vector<Image>::iterator trainingImagesIt,
                       std::vector<uint8_t>::iterator trainingLabelsIt,
                       unsigned numTrainingImages) {
    // For each training image and label, back propogate.
    for (unsigned i = 0; i < mbSize; ++i) {
      backPropogate(*(trainingImagesIt + i), *(trainingLabelsIt + i), i);
    }
    // Gradient descent: for every neuron, compute the new weights and biases.
    for (int i = layers.size() - 1; i >= 0; --i) {
      layers[i]->endBatch(numTrainingImages);
    }
  }

  float sumSquareWeights() {
    float result = 0.0f;
    for (auto &layer : layers) {
      result += layer->sumSquaredWeights();
    }
    return result;
  }

  /// Calculate the total cost for a dataset.
  float evaluateTotalCost(std::vector<Image> &images,
                          std::vector<uint8_t> &labels) {
    float cost = 0.0f;
    for (unsigned i = 0; i < images.size(); ++i) {
      inputLayer.setImage(images[i], 0);
      feedForward(images[i], 0);
      cost += layers.back()->computeOutputCost(labels[i], 0) / images.size();
      // Add the regularisation term.
      cost += 0.5f * (lambda / images.size()) * sumSquareWeights();
    }
    return cost;
  }

public:
  Network(unsigned inputSize, std::vector<Layer*> layers) :
      inputLayer(inputSize),
      layers(layers) {
    // Set neuron inputs.
    layers[0]->setInputs(&inputLayer);
    layers[0]->initialiseDefaultWeights();
    for (unsigned i = 1; i < layers.size(); ++i) {
      layers[i]->setInputs(layers[i - 1]);
      layers[i]->initialiseDefaultWeights();
    }
    // Set neuron outputs.
    for (unsigned i = 0; i < layers.size() - 1; ++i) {
      layers[i]->setOutputs(layers[i + 1]);
    }
  }

  /// Evaluate the test set and return the number of correct classifications.
  unsigned evaluateAccuracy(std::vector<Image> &testImages,
                            std::vector<uint8_t> &testLabels) {
    unsigned result = 0;
    for (unsigned i = 0; i < testImages.size(); ++i) {
      inputLayer.setImage(testImages[i], 0);
      feedForward(testImages[i], 0);
      if (layers.back()->readOutput() == testLabels[i]) {
        ++result;
      }
    }
    return result;
  }

  void SGD(std::vector<Image> &trainingImages,
           std::vector<uint8_t> &trainingLabels,
           std::vector<Image> &validationImages,
           std::vector<uint8_t> &validationLabels,
           std::vector<Image> &testImages,
           std::vector<uint8_t> &testLabels) {
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
        float cost = evaluateTotalCost(validationImages, validationLabels);
        std::cout << "Cost on evaluation data: " << cost << "\n";
      }
      if (monitorTrainingAccuracy) {
        unsigned result = evaluateAccuracy(testImages, testLabels);
        std::cout << "Accuracy on training data: "
                  << result << " / " << testImages.size() << '\n';
      }
      if (monitorTrainingCost) {
        float cost = evaluateTotalCost(testImages, testLabels);
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
  Network network(28 * 28, {
      new FullyConnectedLayer(100),
      new FullyConnectedLayer(10),
    });
  std::cout << "Running...\n";
  network.SGD(trainingImages,
              trainingLabels,
              validationImages,
              validationLabels,
              testImages,
              testLabels);
  return 0;
}
