#include <boost/multi_array.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <limits>
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
const unsigned imageHeight = 28;
const unsigned imageWidth = 28;
const unsigned numEpochs = 1000;
const unsigned mbSize = 10;
const float learningRate = 1.0f;
const float lambda = 5.0f;
const unsigned validationSize = 0;//1000;
const unsigned numTrainingImages = 10000;
const unsigned numTestImages = 10000;
const bool monitorEvaluationAccuracy = false;
const bool monitorEvaluationCost = false;
const bool monitorTrainingAccuracy = true;
const bool monitorTrainingCost = false;
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
  assert(numRows == imageHeight && numCols == imageWidth &&
         "unexpected image size");
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
  /// Each neuron in the network can be indexed by a one- or two-dimensional
  /// coordinate, and stores a weighted input, an activation and an error.
  unsigned index, x, y;
  float weightedInputs[mbSize];
  float activations[mbSize];
  float errors[mbSize];
  Neuron(unsigned index) : index(index) {}
  Neuron(unsigned x, unsigned y) : x(x), y(y) {}
};

struct Layer {
  virtual void initialiseDefaultWeights() = 0;
  virtual void feedForward(unsigned mb) = 0;
  virtual void calcBwdError(unsigned mb) = 0;
  virtual void backPropogate(unsigned mb) = 0;
  virtual void endBatch(unsigned numTrainingImages) = 0;
  virtual void computeOutputError(uint8_t label, unsigned mb) = 0;
  virtual float computeOutputCost(uint8_t label, unsigned mb) = 0;
  virtual float sumSquaredWeights() = 0;
  virtual void setInputs(Layer *layer) = 0;
  virtual void setOutputs(Layer *layer) = 0;
  virtual unsigned readOutput() = 0;
  virtual float getBwdError(unsigned index, unsigned mb) = 0;
  virtual float getBwdError(unsigned x, unsigned y, unsigned mb) = 0;
  virtual Neuron &getNeuron(unsigned index) = 0;
  virtual Neuron &getNeuron(unsigned x, unsigned y) = 0;
  virtual unsigned getNumDims() = 0;
  virtual unsigned getDim(unsigned i) = 0;
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
  void setImage(Image &image, unsigned mb) {
    assert(image.size() == neurons.num_elements() && "invalid image size");
    unsigned imageX = neurons.shape()[0];
    for (unsigned i = 0; i < image.size(); ++i) {
      neurons[i % imageX][i / imageX]->activations[mb] = image[i];
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
  float computeOutputCost(uint8_t, unsigned) override {
    DO_NOT_USE;
    return 0.0f;
  }
  float sumSquaredWeights() override {
    DO_NOT_USE;
    return 0.0f;
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
  unsigned getDim(unsigned i) override { return neurons.shape()[i]; }
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

  void feedForward(unsigned mb) {
    float weightedInput = 0.0f;
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weightedInput += inputs->getNeuron(i).activations[mb] * weights[i];
    }
    weightedInput += bias;
    weightedInputs[mb] = weightedInput;
    activations[mb] = sigmoid(weightedInput);
  }

  void backPropogate(unsigned mb) {
    // Get the weight-error sum component from the next layer, then multiply by
    // the sigmoid derivative to get the error for this neuron.
    float error = outputs->getBwdError(index, mb);
    error *= sigmoidDerivative(weightedInputs[mb]);
    errors[mb] = error;
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
  void computeOutputError(uint8_t label, unsigned mb) {
    float y = label == index ? 1.0f : 0.0f;
    float error = costDelta(weightedInputs[mb], activations[mb], y);
    errors[mb] = error;
  }

  /// Compute the output cost (only the output neurons).
  float computeOutputCost(uint8_t label, unsigned mb) {
    return costFn(activations[mb], label);
  }

  float sumSquaredWeights() {
    float result = 0.0f;
    for (auto weight : weights) {
      result += std::pow(weight, 2.0f);
    }
    return result;
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

  float sumSquaredWeights() override {
    float result = 0.0f;
    for (auto &neuron : neurons) {
      result += neuron.sumSquaredWeights();
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

  void feedForward(unsigned mb) override {
    for (auto &neuron : neurons) {
      neuron.feedForward(mb);
    }
  }

  /// Calculate the l+1 component of the error for each neuron in prev layer.
  void calcBwdError(unsigned mb) override {
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float error = 0.0f;
      for (auto &neuron : neurons) {
        error += neuron.getWeight(i) * neuron.errors[mb];
      }
      bwdErrors[i][mb] = error;
    }
  }

  /// Update errors from next layer.
  void backPropogate(unsigned mb) override {
    for (auto &neuron : neurons) {
      neuron.backPropogate(mb);
    }
  }

  void endBatch(unsigned numTrainingImages) override {
    for (auto &neuron : neurons) {
      neuron.endBatch(numTrainingImages);
    }
  }

  void computeOutputError(uint8_t label, unsigned mb) override {
    for (auto &neuron : neurons) {
      neuron.computeOutputError(label, mb);
    }
  }

  float computeOutputCost(uint8_t label, unsigned mb) override {
    float outputCost = 0.0f;
    for (auto &neuron : neurons) {
      neuron.computeOutputCost(label, mb);
    }
    return outputCost;
  }

  float getBwdError(unsigned index, unsigned mb) override {
    return bwdErrors[index][mb];
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
  unsigned getDim(unsigned i) override {
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
  void feedForward(boost::multi_array_ref<float, 2> &weights, float bias,
                   unsigned mb) {
    // Convolve using each weight.
    float weightedInput = 0.0f;
    for (unsigned a = 0; a < weights.shape()[0]; ++a) {
      for (unsigned b = 0; b < weights.shape()[1]; ++b) {
        float input = inputs->getNeuron(x + a, y + b).activations[mb];
        weightedInput += input * weights[a][b];
      }
    }
    // Add bias and apply non linerarity.
    weightedInput += bias;
    weightedInputs[mb] = weightedInput;
    activations[mb] = sigmoid(weightedInput);
//    std::cout<<"activation["<<x<<","<<y<<"] = "<<activations[mb]<<"\n";
  }
  void backPropogate(unsigned x, unsigned y, unsigned mb) {
//    assert(outputs->getNumDims() == 2 && "Conv must be followed by max pool");
//    float error = outputs->getBwdError(x, y, mb);
    float error = outputs->getNumDims() == 1
              ? outputs->getBwdError((dimX * y) + x, mb)
              : outputs->getBwdError(x, y, mb);
    error *= sigmoidDerivative(weightedInputs[mb]);
    errors[mb] = error;
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
  float bias;
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
    bias = distribution(generator);
  }

  void feedForward(unsigned mb) override {
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y]->feedForward(weights, bias, mb);
      }
    }
  }

  void calcBwdError(unsigned mb) override {
    // Calculate the l+1 component of the error for each neuron in prev layer.
    for (unsigned x = 0; x < inputX; ++x) {
      for (unsigned y = 0; y < inputY; ++y) {
        float error = 0.0f;
        for (unsigned a = 0; a < kernelX; ++a) {
          for (unsigned b = 0; b < kernelY; ++b) {
            if (a <= x &&
                b <= y &&
                x - a < neurons.shape()[0] &&
                y - b < neurons.shape()[1]) {
              error += weights[a][b] * neurons[x - a][y - b]->errors[mb];
            }
          }
        }
        bwdErrors[x][y][mb] = error;
      }
    }
  }

  void backPropogate(unsigned mb) override {
    // Update errors from next layer.
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y]->backPropogate(x, y, mb);
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
        weights[a][b] *= 1.0f - (learningRate * (lambda / numTrainingImages));
        weights[a][b] -= weightDelta;
      }
    }
    // Calculate bias delta and update it.
    float biasDelta = 0.0f;
    // For each item of the minibatch.
    for (unsigned mb = 0; mb < mbSize; ++mb) {
      // For each neuron.
      for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
        for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
          biasDelta += neurons[x][y]->errors[mb];
        }
      }
    }
    biasDelta *= learningRate / mbSize;
    bias -= biasDelta;
  }

  float getBwdError(unsigned x, unsigned y, unsigned mb) override {
    return bwdErrors[x][y][mb];
  }

  float sumSquaredWeights() override {
    float result = 0.0f;
    for (unsigned a = 0; a < kernelX; ++a) {
      for (unsigned b = 0; b < kernelY; ++b) {
        result += std::pow(weights[a][b], 2.0f);
      }
    }
    return result;
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

  float computeOutputCost(uint8_t, unsigned) override {
    DO_NOT_USE;
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
  unsigned getNumDims() override { return neurons.num_dimensions(); }
  unsigned getDim(unsigned i) override { return neurons.shape()[i]; }
  unsigned size() override { return neurons.num_elements(); }
};

///===--------------------------------------------------------------------===///
/// Max pool layer
///===--------------------------------------------------------------------===///
class MaxPoolLayer : public Layer {
  Layer *inputs;
  Layer *outputs;
  unsigned poolX;
  unsigned poolY;
  boost::multi_array<Neuron*, 2> neurons;

public:
  MaxPoolLayer(unsigned poolX, unsigned poolY,
               unsigned inputX, unsigned inputY) :
      inputs(nullptr), outputs(nullptr),
      poolX(poolX), poolY(poolY),
      neurons(boost::extents[inputX / poolX][inputY / poolY]) {
    assert(inputX % poolX == 0 && "Dimension x mismatch with pooling");
    assert(inputY % poolY == 0 && "Dimension y mismatch with pooling");
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        neurons[x][y] = new Neuron(x, y);
      }
    }
  }

  void initialiseDefaultWeights() override { /* Skip */ }

  void feedForward(unsigned mb) override {
    // For each neuron.
    for (unsigned x = 0; x < neurons.shape()[0]; ++x) {
      for (unsigned y = 0; y < neurons.shape()[1]; ++y) {
        // Take maximum activation over pool area.
        float weightedInput = std::numeric_limits<float>::min();
        for (unsigned a = 0; a < poolX; ++a) {
          for (unsigned b = 0; b < poolY; ++b) {
            unsigned nX = (x * poolX) + a;
            unsigned nY = (y * poolY) + b;
            float input = inputs->getNeuron(nX, nY).activations[mb];
            float max = std::max(weightedInput, input);
            neurons[x][y]->activations[mb] = max;
          }
        }
      }
    }
  }

  void calcBwdError(unsigned) override { /* Skip */ }
  void backPropogate(unsigned) override { /* Skip */ }

  float getBwdError(unsigned x, unsigned y, unsigned mb) override {
    // Forward the backwards error component from the next layer.
    unsigned nx = x / poolX;
    unsigned ny = y / poolY;
    return outputs->getNumDims() == 1
              ? outputs->getBwdError((neurons.shape()[0] * ny) + nx, mb)
              : outputs->getBwdError(nx, ny, mb);
  }

  void endBatch(unsigned) override { /* Skip */ }
  float sumSquaredWeights() override { /* Skip */ return 0.0f; }

  float getBwdError(unsigned, unsigned) override {
    DO_NOT_USE; // No FC layers preceed max-pooling layers.
    return 0.0f;
  }

  void computeOutputError(uint8_t, unsigned) override {
    DO_NOT_USE;
  }

  float computeOutputCost(uint8_t, unsigned) override {
    DO_NOT_USE;
    return 0.0f;
  }

  unsigned readOutput() override {
    DO_NOT_USE;
    return 0;
  }

  void setInputs(Layer *layer) override {
    assert(layer->size() == poolX * poolY * neurons.num_elements() &&
           "invalid input layer size");
    inputs = layer;
  }

  void setOutputs(Layer *layer) override { outputs = layer; }

  Neuron &getNeuron(unsigned i) override {
    assert(i < neurons.num_elements() && "Neuron index out of range.");
    unsigned imageX = neurons.shape()[0];
    return *neurons[i % imageX][i / imageX];
  }

  Neuron &getNeuron(unsigned x, unsigned y) override { return *neurons[x][y]; }

  unsigned getNumDims() override { return neurons.num_dimensions(); }
  unsigned getDim(unsigned i) override { return neurons.shape()[i]; }
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
  void feedForward(unsigned mb) {
    for (auto layer : layers) {
      layer->feedForward(mb);
    }
  }

  /// The backward pass.
  void backPropogate(Image &image, uint8_t label, unsigned mb) {
    // Set input.
    inputLayer.setImage(image, mb);
    // Feed forward.
    feedForward(mb);
    // Compute output error in last layer.
    layers.back()->computeOutputError(label, mb);
    layers.back()->calcBwdError(mb);
    // Backpropagate the error and calculate component for next layer.
    for (int i = layers.size() - 2; i > 0; --i) {
      layers[i]->backPropogate(mb);
      layers[i]->calcBwdError(mb);
    }
    layers[0]->backPropogate(mb);
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
      feedForward(0);
      cost += layers.back()->computeOutputCost(labels[i], 0) / images.size();
      // Add the regularisation term.
      cost += 0.5f * (lambda / images.size()) * sumSquareWeights();
    }
    return cost;
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
      if (monitorEvaluationCost) {
        float cost = evaluateTotalCost(validationImages, validationLabels);
        std::cout << "Cost on evaluation data: " << cost << "\n";
      }
      if (monitorTrainingAccuracy) {
        unsigned result = evaluateAccuracy(testImages, testLabels);
        std::cout << "Accuracy on test data: "
                  << result << " / " << testImages.size() << '\n';
      }
      if (monitorTrainingCost) {
        float cost = evaluateTotalCost(testImages, testLabels);
        std::cout << "Cost on test data: " << cost << "\n";
      }
    }
  }

  void setInput(Image &image, unsigned mb) {
    inputLayer.setImage(image, mb);
  }
  unsigned readOutput() {
    return layers.back()->readOutput();
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
  trainingLabels.erase(trainingLabels.begin() + numTrainingImages,
                       trainingLabels.end());
  trainingImages.erase(trainingImages.begin() + numTrainingImages,
                       trainingImages.end());
  testLabels.erase(testLabels.begin() + numTestImages, testLabels.end());
  testImages.erase(testImages.begin() + numTestImages, testImages.end());
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

//  auto Conv1 = new ConvLayer(5, 5, imageHeight, imageWidth);
//  //auto MaxPool1 = new MaxPoolLayer(2, 2, Conv1->getDim(0), Conv1->getDim(1));
//  auto Conv2 = new ConvLayer(5, 5, Conv1->getDim(0), Conv1->getDim(1));
//  //auto MaxPool2 = new MaxPoolLayer(2, 2, Conv2->getDim(0), Conv2->getDim(1));
//  auto FC1 = new FullyConnectedLayer(100, Conv2->size());
//  auto FC2 = new FullyConnectedLayer(10, FC1->size());
//  Network network(imageHeight, imageWidth, {
//            Conv1, Conv2, FC1, FC2 });

  auto Conv1 = new ConvLayer(5, 5, imageHeight, imageWidth);
  auto Pool1 = new MaxPoolLayer(2, 2, Conv1->getDim(0), Conv1->getDim(1));
  auto FC1 = new FullyConnectedLayer(100, Pool1->size());
  auto FC2 = new FullyConnectedLayer(10, FC1->size());
  Network network(imageHeight, imageWidth, {
            Conv1, Pool1, FC1, FC2 });
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
