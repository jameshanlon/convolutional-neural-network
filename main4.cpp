#include <boost/multi_array.hpp>
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

#define DO_NOT_USE assert(0 && "invalid call");

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

/// Globals and constants.
const unsigned numEpochs = 1000;
const unsigned mbSize = 10;
const float learningRate = 1.0f;
const float lambda = 5.0f;
const unsigned validationSize = 0;//1000;
const bool monitorEvaluationAccuracy = false;
const bool monitorTrainingAccuracy = true;
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
  uint32_t magicNumber, numItems;
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

struct Neuron {
  /// Each neuron in the network stores an activation and an error.
  unsigned index, x, y;
  float weightedInputs[mbSize];
  float activations[mbSize];
  float errors[mbSize];
  Neuron(unsigned index) : index(index) {}
  Neuron(unsigned x, unsigned y) : x(x), y(y) {}
};

struct Layer {
  virtual void initialiseDefaultWeights() = 0;
  virtual void feedForward(unsigned mbIndex) = 0;
  virtual void calcBwdError(unsigned mbIndex) = 0;
  virtual void backPropogate(unsigned mbIndex) = 0;
  virtual void endBatch(unsigned numTrainingImages) = 0;
  virtual void computeOutputError(uint8_t label, unsigned mbIndex) = 0;
  virtual void setInputs(Layer *layer) = 0;
  virtual void setOutputs(Layer *layer) = 0;
  virtual unsigned readOutput() = 0;
  virtual float getBwdError(unsigned index, unsigned mbIndex) = 0;
  virtual float getBwdError(unsigned x, unsigned y, unsigned mbIndex) = 0;
  virtual Neuron &getNeuron(unsigned index) = 0;
  virtual Neuron &getNeuron(unsigned x, unsigned y) = 0;
  virtual unsigned getNumDims() = 0;
  virtual unsigned getDimension(unsigned i) = 0;
  virtual unsigned size() = 0;
};

///===--------------------------------------------------------------------===///
/// Input layer.
///===--------------------------------------------------------------------===///
class InputLayer : public Layer {
  boost::multi_array<Neuron*, 2> neurons;
public:
  InputLayer(unsigned imageX, unsigned imageY) :
      neurons(boost::extents[imageX][imageY]) {
    for (unsigned x = 0; x < imageX; ++x) {
      for (unsigned y = 0; y < imageY; ++y) {
        neurons[x][y] = new Neuron(x, y);
      }
    }
  }
  void setImage(Image &image, unsigned mbIndex) {
    assert(image.size() == neurons.num_elements() && "invalid image size");
    unsigned imageX = neurons.shape()[0];
    for (unsigned i = 0; i < image.size(); ++i) {
      neurons[i % imageX][i / imageX]->activations[mbIndex] = image[i];
    }
  }
  void initialiseDefaultWeights() override {
    DO_NOT_USE;
  }
  virtual void calcBwdError(unsigned) override {
    DO_NOT_USE;
  }
  void feedForward(unsigned) override {
    DO_NOT_USE;
  }
  void backPropogate(unsigned) override {
    DO_NOT_USE;
  }
  void endBatch(unsigned) override {
    DO_NOT_USE;
  }
  void computeOutputError(uint8_t, unsigned) override {
    DO_NOT_USE;
  }
  void setInputs(Layer*) override {
    DO_NOT_USE;
  }
  void setOutputs(Layer*) override {
    DO_NOT_USE;
  }
  unsigned readOutput() override {
    DO_NOT_USE;
    return 0;
  }
  float getBwdError(unsigned, unsigned) override {
    DO_NOT_USE;
    return 0.0f;
  }
  float getBwdError(unsigned, unsigned, unsigned) override {
    DO_NOT_USE;
    return 0.0f;
  }
  Neuron &getNeuron(unsigned i) override {
    assert(i < neurons.num_elements() && "Neuron index out of range.");
    unsigned imageX = neurons.shape()[0];
    return *neurons[i % imageX][i / imageX];
  }
  Neuron &getNeuron(unsigned x, unsigned y) override { return *neurons[x][y]; }
  unsigned getNumDims() override { return 2; }
  unsigned getDimension(unsigned i) override { return neurons.shape()[i]; }
  unsigned size() override { return neurons.num_elements(); }
};

///===--------------------------------------------------------------------===///
/// Fully-connected neuron.
///===--------------------------------------------------------------------===///
class FullyConnectedNeuron : public Neuron {
  Layer *inputs;
  Layer *outputs;
  std::vector<float> weights;
  float bias;

public:
  FullyConnectedNeuron(unsigned index) :
    Neuron(index), inputs(nullptr), outputs(nullptr) {}

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

  void feedForward(unsigned mbIndex) {
    float weightedInput = 0.0f;
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weightedInput += inputs->getNeuron(i).activations[mbIndex] * weights[i];
    }
    weightedInput += bias;
    weightedInputs[mbIndex] = weightedInput;
    activations[mbIndex] = sigmoid(weightedInput);
  }

  void backPropogate(unsigned mbIndex) {
    // Get the weight-error sum component from the next layer, then multiply by
    // the sigmoid derivative to get the error for this neuron.
    float error = outputs->getBwdError(index, mbIndex);
    error *= sigmoidDerivative(weightedInputs[mbIndex]);
    errors[mbIndex] = error;
    //std::cout << "error for "<<index<<": "<<error<<"\n";
  }

  void endBatch(unsigned numTrainingImages) {
    // For each weight.
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float weightDelta = 0.0f;
      // For each batch element, average input activation x error (rate of
      // change of cost w.r.t. weight) and multiply by learning rate.
      // Note that FC layers can only be followed by FC layers.
      for (unsigned j = 0; j < mbSize; ++j) {
        weightDelta += inputs->getNeuron(i).activations[j] * errors[j];
      }
      weightDelta *= learningRate / mbSize;
      weights[i] *= 1.0f - (learningRate * (lambda / numTrainingImages));
      weights[i] -= weightDelta;
//      std::cout<<"Weight: "<<weights[i]<<"\n";
    }
    // For each batch element, average the errors (error is equal to rate of
    // change of cost w.r.t. bias) and multiply by learning rate.
    float biasDelta = 0.0f;
    for (unsigned j = 0; j < mbSize; ++j) {
      biasDelta += errors[j];
    }
    biasDelta *= learningRate / mbSize;
    bias -= biasDelta;
//    std::cout<<"Bias: "<<bias<<"\n";
  }

  /// Compute the output error (only the output neurons).
  void computeOutputError(uint8_t label, unsigned mbIndex) {
    float y = label == index ? 1.0f : 0.0f;
    float error = costDelta(weightedInputs[mbIndex], activations[mbIndex], y);
    errors[mbIndex] = error;
  }

  void setInputs(Layer *inputs) { this->inputs = inputs; }
  void setOutputs(Layer *outputs) { this->outputs = outputs; }
  unsigned numWeights() { return weights.size(); }
  float getWeight(unsigned i) { return weights.at(i); }
};

///===--------------------------------------------------------------------===///
/// Fully-connected layer.
///===--------------------------------------------------------------------===///
class FullyConnectedLayer : public Layer {
  Layer *inputs;
  Layer *outputs;
  std::vector<FullyConnectedNeuron> neurons;
  boost::multi_array<float, 2> bwdErrors;

public:
  FullyConnectedLayer(unsigned size, unsigned prevSize) :
      bwdErrors(boost::extents[prevSize][mbSize]) {
    for (unsigned i = 0; i < size; ++i) {
      neurons.push_back(FullyConnectedNeuron(i));
    }
  }

  /// Determine the index of the highest output activation.
  unsigned readOutput() override {
    unsigned result = 0;
    float max = 0.0f;
    for (unsigned i = 0; i < neurons.size(); ++i) {
      float output = neurons[i].activations[0];
      if (output > max) {
        result = i;
        max = output;
      }
    }
    return result;
  }

  void setInputs(Layer *layer) override {
    inputs = layer;
    for (auto &neuron : neurons) {
      neuron.setInputs(layer);
    }
  }

  void setOutputs(Layer *layer) override {
    outputs = layer;
    for (auto &neuron : neurons) {
      neuron.setOutputs(layer);
    }
  }

  void initialiseDefaultWeights() override {
    for (auto &neuron : neurons) {
      neuron.initialiseDefaultWeights();
    }
  }

  void feedForward(unsigned mbIndex) override {
    for (auto &neuron : neurons) {
      neuron.feedForward(mbIndex);
    }
  }

  /// Calculate the l+1 component of the error for each neuron in prev layer.
  void calcBwdError(unsigned mbIndex) override {
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float error = 0.0f;
      for (auto &neuron : neurons) {
        error += neuron.getWeight(i) * neuron.errors[mbIndex];
      }
      bwdErrors[i][mbIndex] = error;
    }
  }

  /// Update errors from next layer.
  void backPropogate(unsigned mbIndex) override {
    for (auto &neuron : neurons) {
      neuron.backPropogate(mbIndex);
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

  float getBwdError(unsigned index, unsigned mbIndex) override {
    return bwdErrors[index][mbIndex];
  }
  float getBwdError(unsigned, unsigned, unsigned) override {
    DO_NOT_USE;
    return 0.0f;
  }

  Neuron &getNeuron(unsigned index) override { return neurons.at(index); }
  Neuron &getNeuron(unsigned, unsigned) override {
    DO_NOT_USE;
    return *new Neuron(0);
  }
  unsigned getNumDims() override { return 1; }
  unsigned getDimension(unsigned i) override {
    assert(i == 0 && "Layer is 1D");
    return neurons.size();
  }
  unsigned size() override { return neurons.size(); }
};

///===--------------------------------------------------------------------===///
/// Convolutional neuron.
///===--------------------------------------------------------------------===///
class ConvNeuron : public Neuron {
  Layer *inputs;
  Layer *outputs;
  unsigned dimX;
public:
  ConvNeuron(unsigned x, unsigned y, unsigned dimX) : Neuron(x, y), dimX(dimX) {}
  void feedForward(boost::multi_array_ref<float, 2> &weights, unsigned mbIndex) {
    // Convolve using each weight.
    float weightedInput = 0.0f;
    for (unsigned a = 0; a < weights.shape()[0]; ++a) {
      for (unsigned b = 0; b < weights.shape()[1]; ++b) {
        float input = inputs->getNeuron(x + a, y + b).activations[mbIndex];
        weightedInput += input * weights[a][b];
      }
    }
    // Apply non linerarity.
    weightedInputs[mbIndex] = weightedInput;
    activations[mbIndex] = sigmoid(weightedInput);
  }
  void backPropogate(unsigned x, unsigned y, unsigned mbIndex) {
    float error = outputs->getNumDims() == 1
              ? outputs->getBwdError((dimX * y) + x, mbIndex)
              : outputs->getBwdError(x, y, mbIndex);
    error *= sigmoidDerivative(weightedInputs[mbIndex]);
    errors[mbIndex] = error;
  }
  void setInputs(Layer *inputs) { this->inputs = inputs; }
  void setOutputs(Layer *outputs) { this->outputs = outputs; }
};

///===--------------------------------------------------------------------===///
/// Convolutional layer
///
/// kernelX is num cols
/// kernelY is num rows
/// neuron(x, y) is row y, col x
/// weights(a, b) is row b, col a
/// TODO: weights, neurons and biases per feature map.
///===--------------------------------------------------------------------===///
class ConvLayer : public Layer {
  Layer *inputs;
  Layer *outputs;
  unsigned kernelX;
  unsigned kernelY;
  unsigned inputX;
  unsigned inputY;
  boost::multi_array<ConvNeuron*, 2> neurons;
  boost::multi_array<float, 2> weights;
  boost::multi_array<float, 3> bwdErrors;

public:
  ConvLayer(unsigned kernelX, unsigned kernelY,
            unsigned inputX, unsigned inputY) :
      inputs(nullptr), outputs(nullptr),
      inputX(inputX), inputY(inputY),
      neurons(boost::extents[inputX - kernelX + 1][inputY - kernelY + 1]),
      weights(boost::extents[kernelX][kernelY]),
      bwdErrors(boost::extents[inputX][inputY][mbSize]) {
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y] = new ConvNeuron(x, y, neurons.shape()[0]);
      }
    }
  }

  void initialiseDefaultWeights() override {
    static std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0, 1.0f);
    for (unsigned a = 0; a < weights.shape()[0]; ++a) {
      for (unsigned b = 0; b < weights.shape()[1]; ++b) {
        weights[a][b] = distribution(generator) / std::sqrt(inputs->size());
      }
    }
  }

  void feedForward(unsigned mbIndex) override {
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y]->feedForward(weights, mbIndex);
      }
    }
  }

  void calcBwdError(unsigned mbIndex) override {
    DO_NOT_USE;
  }

  void backPropogate(unsigned mbIndex) override {
    // Update errors from next layer.
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y]->backPropogate(x, y, mbIndex);
      }
    }
  }

  void endBatch(unsigned numTrainingImages) override {
    // Calculate delta for each weight and update.
    for (unsigned a = 0; a < weights.shape()[0]; ++a) {
      for (unsigned b = 0; b < weights.shape()[1]; ++b) {
        float weightDelta = 0.0f;
        // For each item of the minibatch.
        for (unsigned mb = 0; mb < mbSize; ++mb) {
          // For each neuron.
          for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
            for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
              weightDelta += inputs->getNeuron(x + a, y + b).activations[mb]
                                 * neurons[x][y]->errors[mb];
            }
          }
        }
        weightDelta *= learningRate / mbSize;
        weights[a][b] -= weightDelta;
//        std::cout << "wd: "<<weightDelta<<"\n";
      }
    }
//    std::cout << "Weights:\n";
//    for (unsigned a = 0; a < weights.shape()[0]; ++a) {
//      for (unsigned b = 0; b < weights.shape()[1]; ++b) {
//        std::cout << weights[a][b] << " ";
//      }
//      std::cout << "\n";
//    }
  }

  float getBwdError(unsigned x, unsigned y, unsigned mbIndex) override {
    return bwdErrors[x][y][mbIndex];
  }

  void setInputs(Layer *layer) override {
    assert(layer->size() == inputX * inputY && "Invalid input layer size");
    inputs = layer;
    std::for_each(neurons.data(), neurons.data() + neurons.num_elements(),
                  [layer](ConvNeuron *n){ n->setInputs(layer); });
  }

  void setOutputs(Layer *layer) override {
    outputs = layer;
    std::for_each(neurons.data(), neurons.data() + neurons.num_elements(),
                  [layer](ConvNeuron *n){ n->setOutputs(layer); });
  }

  float getBwdError(unsigned, unsigned) override {
    DO_NOT_USE; // No FC layers preceed conv layers.
    return 0.0f;
  }

  void computeOutputError(uint8_t, unsigned) override {
    DO_NOT_USE;
  }

  unsigned readOutput() override {
    DO_NOT_USE;
    return 0;
  }

  Neuron &getNeuron(unsigned i) override {
    assert(i < neurons.num_elements() && "Neuron index out of range");
    unsigned imageX = neurons.shape()[0];
    return *neurons[i % imageX][i / imageX];
  }

  Neuron &getNeuron(unsigned x, unsigned y) override { return *neurons[x][y]; }
  unsigned getNumDims() override { return 2; }
  unsigned getDimension(unsigned i) override { return neurons.shape()[i]; }
  unsigned size() override { return neurons.num_elements(); }
};

///===--------------------------------------------------------------------===///
/// The network.
///===--------------------------------------------------------------------===///
class Network {
  InputLayer inputLayer;
  std::vector<Layer*> layers;

public:
  Network(unsigned inputX, unsigned inputY, std::vector<Layer*> layers) :
      inputLayer(inputX, inputY), layers(layers) {
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

  /// The forward pass.
  void feedForward(unsigned mbIndex) {
    for (auto layer : layers) {
      layer->feedForward(mbIndex);
    }
  }

  /// The backward pass.
  void backPropogate(Image &image, uint8_t label, unsigned mbIndex) {
    // Set input.
    inputLayer.setImage(image, mbIndex);
    // Feed forward.
    feedForward(mbIndex);
    // Compute output error in last layer.
    layers.back()->computeOutputError(label, mbIndex);
    layers.back()->calcBwdError(mbIndex);
    // Backpropagate the error and calculate component for next layer.
    for (int i = layers.size() - 2; i > 0; --i) {
      layers[i]->backPropogate(mbIndex);
      layers[i]->calcBwdError(mbIndex);
    }
    layers[0]->backPropogate(mbIndex);
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

  /// Evaluate the test set and return the number of correct classifications.
  unsigned evaluateAccuracy(std::vector<Image> &testImages,
                            std::vector<uint8_t> &testLabels) {
    unsigned result = 0;
    for (unsigned i = 0; i < testImages.size(); ++i) {
      inputLayer.setImage(testImages[i], 0);
      feedForward(0);
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
        std::cout << "\rUpdate minibatch: " << i << " / " << end;
        updateMiniBatch(trainingImages.begin() + i,
                        trainingLabels.begin() + i,
                        trainingImages.size());
      }
      std::cout << '\n';
      // Evaluate the test set.
      std::cout << "Epoch " << epoch << " complete.\n";
      if (monitorEvaluationAccuracy) {
        unsigned result = evaluateAccuracy(validationImages, validationLabels);
        std::cout << "Accuracy on evaluation data: "
                  << result << " / " << validationImages.size() << '\n';
      }
      if (monitorTrainingAccuracy) {
        unsigned result = evaluateAccuracy(testImages, testLabels);
        std::cout << "Accuracy on test data: "
                  << result << " / " << testImages.size() << '\n';
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
  // Labels.
  std::cout << "Reading labels\n";
  readLabels("train-labels-idx1-ubyte", trainingLabels);
  readLabels("t10k-labels-idx1-ubyte", testLabels);
  //Images.
  std::cout << "Reading images\n";
  readImages("train-images-idx3-ubyte", trainingImages);
  readImages("t10k-images-idx3-ubyte", testImages);
  // Reduce number of training images and use them for test (for debugging).
  trainingLabels.erase(trainingLabels.begin() + 1000, trainingLabels.end());
  trainingImages.erase(trainingImages.begin() + 1000, trainingImages.end());
  testLabels.erase(testLabels.begin() + 1000, testLabels.end());
  testImages.erase(testImages.begin() + 1000, testImages.end());
  // Take images from the training set for validation.
  std::vector<uint8_t> validationLabels(trainingLabels.end() - validationSize,
                                        trainingLabels.end());
  std::vector<Image> validationImages(trainingImages.end() - validationSize,
                                      trainingImages.end());
  trainingLabels.erase(trainingLabels.end() - validationSize,
                       trainingLabels.end());
  trainingImages.erase(trainingImages.end() - validationSize,
                       trainingImages.end());
  // Create the network.
  std::cout << "Creating the network\n";
//  auto FC1 = new FullyConnectedLayer(100, 28 * 28);
//  auto FC2 = new FullyConnectedLayer(10, FC1->size());
//  Network network(28, 28, { FC1, FC2 });
  auto Conv1 = new ConvLayer(5, 5, 28, 28);
  auto FC1 = new FullyConnectedLayer(100, Conv1->size());
  auto FC2 = new FullyConnectedLayer(10, FC1->size());
  Network network(28, 28, { Conv1, FC1, FC2 });
  // Run it.
  std::cout << "Running...\n";
  network.SGD(trainingImages,
              trainingLabels,
              validationImages,
              validationLabels,
              testImages,
              testLabels);
  return 0;
}
