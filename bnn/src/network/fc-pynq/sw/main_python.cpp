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

std::vector<tiny_cnn::label_t> trainLabels;
std::vector<tiny_cnn::vec_t> trainImages;

extern "C" void load_images(const char *path)
{
  std::string trainLabelPath(path);
  trainLabelPath.append("/train-labels-idx1-ubyte");
  std::string trainImagePath(path);
  trainImagePath.append("/train-images-idx3-ubyte");
  tiny_cnn::parse_mnist_labels(trainLabelPath, &trainLabels);
  tiny_cnn::parse_mnist_images(trainImagePath, &trainImages, 0.0, 1.0, 0, 0);
}

extern "C" float *train(unsigned int imageNum, float *usecPerImage)
{
  bnn_fc::FoldedMVInit("fc-pynq");
  std::vector<float> wBResult;
  float usecPerImage_int;
  wBResult = bnn_fc::trainMNIST<0>(trainImages, trainLabels, imageNum, usecPerImage_int);
  float *result = new float[W_B_SIZE];
  std::copy(wBResult.begin(), wBResult.end(), result);
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return result;
}

extern "C" void free_results(float *result)
{
  delete result;
  result = 0;
}

extern "C" void free_images()
{
  std::vector<tiny_cnn::label_t>().swap(trainLabels);
  std::vector<tiny_cnn::vec_t>().swap(trainImages);
}

extern "C" void deinit() {
  bnn_fc::FoldedMVDeinit();
}
