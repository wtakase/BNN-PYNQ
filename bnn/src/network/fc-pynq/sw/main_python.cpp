/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file main_python.c
 *
 * Host code for BNN, overlay FC-Pynq, to manage parameter loading, 
 * classification (training) of single and multiple images
 * 
 *
 *****************************************************************************/
#include <iostream>
#include <string.h>
#include "foldedmv-offload-fc.h"


extern "C" double *train(const char *path, unsigned int imageNum, float *usecPerImage)
{
  std::string trainLabelPath(path);
  trainLabelPath.append("/train-labels-idx1-ubyte");
  std::string trainImagePath(path);
  trainImagePath.append("/train-images-idx3-ubyte");
  std::vector<tiny_cnn::label_t> trainLabels;
  std::vector<tiny_cnn::vec_t> trainImages;
  tiny_cnn::parse_mnist_labels(trainLabelPath, &trainLabels);
  tiny_cnn::parse_mnist_images(trainImagePath, &trainImages, 0.0, 1.0, 0, 0);

  std::string testLabelPath(path);
  testLabelPath.append("/t10k-labels-idx1-ubyte");
  std::string testImagePath(path);
  testImagePath.append("/t10k-images-idx3-ubyte");
  std::vector<tiny_cnn::label_t> testLabels;
  std::vector<tiny_cnn::vec_t> testImages;
  tiny_cnn::parse_mnist_labels(testLabelPath, &testLabels);
  tiny_cnn::parse_mnist_images(testImagePath, &testImages, 0.0, 1.0, 0, 0);

  bnn_fc::FoldedMVInit("fc-pynq");
  std::vector<double> accuracyResult;
  float usecPerImage_int;
  accuracyResult = bnn_fc::trainMNIST<0>(trainImages, trainLabels, testImages, testLabels, imageNum, usecPerImage_int);
  double *result = new double[12000];
  std::copy(accuracyResult.begin(), accuracyResult.end(), result);
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return result;
}

extern "C" void free_results(double *result)
{
  delete[] result;
}

extern "C" void deinit() {
  bnn_fc::FoldedMVDeinit();
}
