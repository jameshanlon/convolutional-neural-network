#ifndef _DATA_H_
#define _DATA_H_

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "Params.hpp"

using Image = std::vector<float>;

class Data {

  static constexpr unsigned imageHeight = 28;
  static constexpr unsigned imageWidth = 28;

  std::vector<uint8_t> trainingLabels;
  std::vector<uint8_t> testLabels;
  std::vector<uint8_t> validationLabels;
  std::vector<Image>   trainingImages;
  std::vector<Image>   testImages;
  std::vector<Image>   validationImages;

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

public:
  Data(Params params) {
    // Labels.
    std::cout << "Reading labels\n";
    readLabels("train-labels-idx1-ubyte", trainingLabels);
    readLabels("t10k-labels-idx1-ubyte", testLabels);
    //Images.
    std::cout << "Reading images\n";
    readImages("train-images-idx3-ubyte", trainingImages);
    readImages("t10k-images-idx3-ubyte", testImages);
    // Reduce number of training images and use them for test (for debugging).
    trainingLabels.erase(trainingLabels.begin() + params.numTrainingImages,
                         trainingLabels.end());
    trainingImages.erase(trainingImages.begin() + params.numTrainingImages,
                         trainingImages.end());
    testLabels.erase(testLabels.begin() + params.numTestImages,
                     testLabels.end());
    testImages.erase(testImages.begin() + params.numTestImages,
                     testImages.end());
    // Take images from the training set for validation.
    validationLabels.assign(trainingLabels.end() - params.numValidationImages,
                            trainingLabels.end());
    validationImages.assign(trainingImages.end() - params.numValidationImages,
                            trainingImages.end());
    trainingLabels.erase(trainingLabels.end() - params.numValidationImages,
                         trainingLabels.end());
    trainingImages.erase(trainingImages.end() - params.numValidationImages,
                         trainingImages.end());
  }
  std::vector<Image>   &getTrainingImages()   { return trainingImages; }
  std::vector<uint8_t> &getTrainingLabels()   { return trainingLabels; }
  std::vector<Image>   &getValidationImages() { return validationImages; }
  std::vector<uint8_t> &getValidationLabels() { return validationLabels; }
  std::vector<Image>   &getTestImages()       { return testImages; }
  std::vector<uint8_t> &getTestLabels()       { return testLabels; }
};

#endif
