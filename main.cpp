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

const unsigned numEpochs = 30;
const unsigned mbSize = 10;
const float learningRate = 3.0;

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

template<int mbSize>
class Neuron {
  std::vector<Neuron<mbSize>> *inputs;
  std::vector<Neuron<mbSize>> *outputs;
  std::vector<float> weights;
  float bias;
  unsigned index;
  // Per batch.
  float weightedInputs[mbSize];
  float activations[mbSize];
  float errors[mbSize];

  float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
  }

  /// Derivative of the sigmoid function.
  float sigmoidDerivative(float z) {
    return sigmoid(z) * (1.0f - sigmoid(z));
  }

public:
  Neuron(unsigned index) : index(index) {}

  void initialise() {
    // Initialise all weights and biases with random values from
    // normal distribution with mean 0 and stdandard deviation 1.
    static std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0, 1.0);
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weights.push_back(distribution(generator));
    }
    bias = distribution(generator);
  }

  /// Compute the output error (only the output neurons).
  void computeOutputError(uint8_t label, unsigned mbIndex) {
    float y = label == index ? 1.0f : 0.0f;
    float error = (activations[mbIndex] - y)
                     * sigmoidDerivative(weightedInputs[mbIndex]);
    errors[mbIndex] = error;
  }

  void forwardBatch(unsigned mbIndex) {
    float weightedInput = 0.0f;
    for (unsigned i = 0; i < inputs->size(); ++i) {
      weightedInput += (*inputs)[i].getOutput(mbIndex) * weights[i];
    }
    weightedInput += bias;
    weightedInputs[mbIndex] = weightedInput;
    activations[mbIndex] = sigmoid(weightedInput);
  }

  void backwardBatch(unsigned mbIndex) {
    float error = 0.0f;
    for (Neuron<mbSize> &neuron : *outputs) {
      error += neuron.getWeight(index) * neuron.getError(mbIndex);
    }
    error *= sigmoidDerivative(weightedInputs[mbIndex]);
    errors[mbIndex] = error;
  }

  void endBatch(float learningRate) {
    // For each weight.
    for (unsigned i = 0; i < inputs->size(); ++i) {
      float weightDelta = 0.0f;
      // For each batch element, average input activation x error (rate of
      // change of cost w.r.t. weight) and multiply by learning rate.
      for (unsigned j = 0; j < mbSize; ++j) {
        weightDelta += inputs->at(i).getOutput(j) * errors[j];
      }
      weightDelta *= learningRate / static_cast<float>(mbSize);
      weights[i] -= weightDelta;
    }
    // For each batch element, average the errors (error is equal to rate of
    // change of cost w.r.t. bias) and multiply by learning rate.
    float biasDelta = 0;
    for (unsigned j = 0; j < mbSize; ++j) {
      biasDelta += errors[j];
    }
    biasDelta *= learningRate / static_cast<float>(mbSize);
    bias -= biasDelta;
  }

  void setInputs(std::vector<Neuron> *inputs) {
    this->inputs = inputs;
  }
  void setOutputs(std::vector<Neuron> *outputs) {
    this->outputs = outputs;
  }
  void setOutput(unsigned i, float value) {
    activations[i] = value;
  }
  /// Get output of batch element mbIndex.
  float getOutput(unsigned mbIndex) { return activations[mbIndex]; }
  /// Get error of batch element i.
  float getError(unsigned mbIndex) { return errors[mbIndex]; }
  /// Get weight of connection i.
  float getWeight(unsigned i) { return weights.at(i); }
};

template<int mbSize>
class Network {
  std::vector<Neuron<mbSize>> inputLayer;
  std::vector<std::vector<Neuron<mbSize>>> layers;

  /// Set the activations of the input neurons with an image.
  void setInput(Image &image, unsigned mbIndex) {
    assert(image.size() == layers[0].size() && "invalid input layer");
    for (unsigned i = 0; i < layers[0].size(); ++i) {
      layers[0][i].setOutput(mbIndex, image[i]);
    }
  }

  /// Determine the index of the highest output activation.
  unsigned readOutput() {
    unsigned result = 0;
    float max = 0.0f;
    for (unsigned i = 0; i < layers.back().size(); ++i) {
      float output = layers.back()[i].getOutput(0);
      if (output > max) {
        result = i;
        max = output;
      }
    }
    return result;
  }

  void backprop(Image &image, uint8_t label, unsigned mbIndex) {
    // Set input.
    setInput(image, mbIndex);
    // Feed forward.
    for (unsigned i = 1; i < layers.size(); ++i) {
      for (Neuron<mbSize> &neuron : layers[i]) {
        neuron.forwardBatch(mbIndex);
      }
    }
    // Compute output error (last layer).
    for (Neuron<mbSize> &neuron : layers.back()) {
      neuron.computeOutputError(label, mbIndex);
    }
    // Backpropagate the error.
    for (unsigned i = layers.size() - 2; i > 0; --i) {
      for (unsigned j = 0; j < layers[i].size(); ++j) {
        layers[i][j].backwardBatch(mbIndex);
      }
    }
  }

  void updateMiniBatch(std::vector<Image>::iterator trainingImagesIt,
                       std::vector<uint8_t>::iterator trainingLabelsIt,
                       float learningRate) {
    // For each training image and label, backpropogate.
    for (unsigned i = 0; i < mbSize; ++i) {
      backprop(*(trainingImagesIt + i), *(trainingLabelsIt + i), i);
    }
    // Gradient descent: for every neuron, compute the new weights and biases.
    for (unsigned i = layers.size() - 1; i > 0; --i) {
      for (Neuron<mbSize> &neuron : layers[i]) {
        neuron.endBatch(learningRate);
      }
    }
  }

public:
  Network(const std::vector<unsigned> sizes) {
    // Create the input layer.
    auto layer = std::vector<Neuron<mbSize>>();
    for (unsigned i = 0; i < sizes[0]; ++i) {
      layer.push_back(Neuron<mbSize>(i));
    }
    layers.push_back(layer);
    // For each remaining layer.
    for (unsigned i = 1; i < sizes.size(); ++i) {
      auto layer = std::vector<Neuron<mbSize>>();
      // For each neuron.
      for (unsigned j = 0; j < sizes[i]; ++j) {
        layer.push_back(Neuron<mbSize>(j));
      }
      layers.push_back(layer);
    }
    // Set neuron inputs.
    for (unsigned i = 1; i < layers.size(); ++i) {
      for (unsigned j = 0; j < layers[i].size(); ++j) {
        layers[i][j].setInputs(&layers[i-1]);
        layers[i][j].initialise();
      }
    }
    // Set neuron outputs.
    for (unsigned i = 0; i < layers.size() - 1; ++i) {
      for (unsigned j = 0; j < layers[i].size(); ++j) {
        layers[i][j].setOutputs(&layers[i+1]);
      }
    }
  }

  /// Evaluate the test set and return the number of correct classifications.
  unsigned evaluate(std::vector<Image> &testImages,
                    std::vector<uint8_t> &testLabels) {
    unsigned result = 0;
    for (unsigned i = 0; i < testImages.size(); ++i) {
      //std::cout << "\rTest image " << i;
      setInput(testImages[i], 0);
      // Calculate the activations for each neuron, for each layer in sequence.
      for (unsigned j = 1; j < layers.size(); ++j) {
        for (Neuron<mbSize> &neuron : layers[j]) {
          neuron.forwardBatch(0);
        }
      }
      // Read the output.
      if (readOutput() == testLabels[i]) {
        ++result;
      }
    }
    //std::cout << '\n';
    return result;
  }

  void SGD(std::vector<Image> &trainingImages,
           std::vector<uint8_t> &trainingLabels,
           std::vector<Image> &testImages,
           std::vector<uint8_t> &testLabels,
           unsigned numEpochs,
           float learningRate) {
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
                        learningRate);
      }
      //std::cout << '\n';
      // Evaluate the test set.
      unsigned result = evaluate(testImages, testLabels);
      std::cout << "Epoch " << epoch << ": "
                << result << " / " << testImages.size() << '\n';
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

  std::cout << "Creating the network\n";
  Network<mbSize> network({28*28, 30, 10});
  std::cout << "Running...\n";
  network.SGD(trainingImages,
              trainingLabels,
              testImages,
              testLabels,
              numEpochs,
              learningRate);
  return 0;
}
