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
 * @file foldedmv-offload-fc.h
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/

#pragma once
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "tiny_cnn/tiny_cnn.h"
#include "DlUtil.hpp"
#include "DlAffine.hpp"
#include "DlSoftmaxWithLoss.hpp"

namespace bnn_fc
{

#ifndef VIRTUAL
#define INPUT_BUF_ENTRIES       3840000
#define OUTPUT_BUF_ENTRIES      160000
#else
#define INPUT_BUF_ENTRIES	8192
#define OUTPUT_BUF_ENTRIES	1024
#endif
#define FOLDEDMV_INPUT_PADCHAR  0

void FoldedMVInit(const char *attachName);

void FoldedMVDeinit();

std::string getBNNRoot();

template<unsigned int dummy>
void SetParam(ExtMemWord *param, unsigned int rowNum, unsigned int colNum, double weightInitStd)
{
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution<> dist(0.0, weightInitStd);
  for (int i = 0; i < rowNum; ++i) {
    for (int j = 0; j < colNum; ++j) {
      if (weightInitStd == 0) {
        param[i * colNum + j] = 0.0;
      } else {
        param[i * colNum + j] = (ExtMemWord)dist(engine);
      }
    }
  }
}

} // namespace bnn_fc

#if defined(OFFLOAD) && defined(RAWHLS)
#include "bnn-library-fc.h"

namespace bnn_fc
{
void BlackBoxJam(ExtMemWord *in, ExtMemWord *out);

extern ExtMemWord *bufIn, *bufOut;

template<unsigned int dummy>
std::vector<float> trainMNIST(std::vector<tiny_cnn::vec_t> &trainImages, std::vector<tiny_cnn::label_t> &trainLabels, const unsigned int imageNum, float &usecPerImage) {

  const unsigned int imageSize = trainImages[0].size();

  // allocate host-side buffers for packed input and outputs
  unsigned int imagesSize = BATCH_SIZE * imageSize;
  unsigned int labelsSize = BATCH_SIZE * 1;
  unsigned int packedInSize = W_B_SIZE + imagesSize + labelsSize;
  unsigned int packedOutSize = W_B_SIZE;

  if (INPUT_BUF_ENTRIES < packedInSize) {
    throw "Not enough space in accelBufIn";
  }
  if (OUTPUT_BUF_ENTRIES < packedOutSize) {
    throw "Not enough space in accelBufOut";
  }

  // NOTE(wtakase): Need comment out to prevent
  // 'application performed illegal memory access and is being terminated'
  //ExtMemWord *packedIn = new ExtMemWord[packedInSize];
  //ExtMemWord *packedOut = new ExtMemWord[packedOutSize];
  ExtMemWord packedIn[packedInSize];
  ExtMemWord packedOut[packedOutSize];

  // initialize weights and biases and pack them
  unsigned int inOffset = 0;
  bnn_fc::SetParam<0>(&packedIn[inOffset], INPUT_SIZE, HIDDEN1_SIZE, WEIGHT_INIT_STD);
  inOffset += W1_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], 1, HIDDEN1_SIZE, 0.0);
  inOffset += B1_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], HIDDEN1_SIZE, OUTPUT_SIZE, WEIGHT_INIT_STD);
  inOffset += W2_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], 1, OUTPUT_SIZE, 0.0);
  inOffset += B2_SIZE;

  unsigned int countLoopNum = 1;
  unsigned int count = BATCH_SIZE;
  if (imageNum > BATCH_SIZE) {
    countLoopNum = imageNum / BATCH_SIZE;
    if (imageNum % BATCH_SIZE > 0) {
      countLoopNum++;
    }
  }

  std::vector<float> result;
  for (unsigned int countLoop = 0; countLoop < countLoopNum; countLoop++) {
    unsigned int countOffset = countLoop * count;
    // pack images and labels
    inOffset = W_B_SIZE;
    for (unsigned int i = 0; i < count; i++) {
      for (unsigned int j = 0; j < imageSize + 1; j++) {
        if (j < imageSize) {
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(trainImages[countOffset + i][j]);
        } else {
#if defined(HLSFIXED) && !defined(HLSNOSHIFT)
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(static_cast<ShiftMemWord>(trainLabels[countOffset + i]) >> 4);
#else
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(trainLabels[countOffset + i]);
#endif
        }
      }
    }

    //auto t1 = std::chrono::high_resolution_clock::now();
    // call the accelerator in compute mode
    bnn_fc::BlackBoxJam((ExtMemWord *)packedIn, (ExtMemWord *)packedOut);
    //auto t2 = std::chrono::high_resolution_clock::now();

    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //usecPerImage = (float)duration / (count);

    //std::cout << "1-batch (" << count << " images) training took " << duration << " microseconds, " << usecPerImage << " usec per image, " << 1000000.0 / usecPerImage << " images per second" << ", Test accuracy: " << accuracy << std::endl;

    // get trained weights and biases
    for (unsigned int i = 0; i < W_B_SIZE; i++) {
      packedIn[i] = packedOut[i];
    }
  }

  // put trained weights and biases
  for (unsigned int i = 0; i < W_B_SIZE; i++) {
    result.push_back(static_cast<float>(packedOut[i]));
  }

  // NOTE(wtakase): Need comment out to prevent
  // 'application performed illegal memory access and is being terminated'
  //delete packedIn;
  //delete packedOut;
  //packedIn = 0;
  //packedOut = 0;
  return (result);
}

} // namespace bnn_fc

#elif defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

namespace bnn_fc
{

extern DonutDriver * thePlatform;
extern void *accelBufIn, *accelBufOut;
extern ExtMemWord *bufIn, *bufOut;

void ExecAccel();

template<unsigned int dummy>
std::vector<float> trainMNIST(std::vector<tiny_cnn::vec_t> &trainImages, std::vector<tiny_cnn::label_t> &trainLabels, const unsigned int imageNum, float &usecPerImage) {

  const unsigned int imageSize = trainImages[0].size();

  // allocate host-side buffers for packed input and outputs
  unsigned int imagesSize = BATCH_SIZE * imageSize;
  unsigned int labelsSize = BATCH_SIZE * 1;
  unsigned int packedInSize = W_B_SIZE + imagesSize + labelsSize;
  unsigned int packedOutSize = W_B_SIZE;

  if (INPUT_BUF_ENTRIES < packedInSize) {
    throw "Not enough space in accelBufIn";
  }
  if (OUTPUT_BUF_ENTRIES < packedOutSize) {
    throw "Not enough space in accelBufOut";
  }

  // NOTE(wtakase): Need comment out to prevent
  // 'application performed illegal memory access and is being terminated'
  //ExtMemWord *packedIn = new ExtMemWord[packedInSize];
  //ExtMemWord *packedOut = new ExtMemWord[packedOutSize];
  ExtMemWord packedIn[packedInSize];
  ExtMemWord packedOut[packedOutSize];

  // initialize weights and biases and pack them
  unsigned int inOffset = 0;
  bnn_fc::SetParam<0>(&packedIn[inOffset], INPUT_SIZE, HIDDEN1_SIZE, WEIGHT_INIT_STD);
  inOffset += W1_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], 1, HIDDEN1_SIZE, 0.0);
  inOffset += B1_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], HIDDEN1_SIZE, OUTPUT_SIZE, WEIGHT_INIT_STD);
  inOffset += W2_SIZE;
  bnn_fc::SetParam<0>(&packedIn[inOffset], 1, OUTPUT_SIZE, 0.0);
  inOffset += B2_SIZE;

  unsigned int countLoopNum = 1;
  unsigned int count = BATCH_SIZE;
  if (imageNum > BATCH_SIZE) {
    countLoopNum = imageNum / BATCH_SIZE;
    if (imageNum % BATCH_SIZE > 0) {
      countLoopNum++;
    }
  }

  std::vector<float> result;
  for (unsigned int countLoop = 0; countLoop < countLoopNum; countLoop++) {
    unsigned int countOffset = countLoop * count;
    // pack images and labels
    inOffset = W_B_SIZE;
    for (unsigned int i = 0; i < count; i++) {
      for (unsigned int j = 0; j < imageSize + 1; j++) {
        if (j < imageSize) {
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(trainImages[countOffset + i][j]);
        } else {
#if defined(HLSFIXED) && !defined(HLSNOSHIFT)
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(static_cast<ShiftMemWord>(trainLabels[countOffset + i]) >> 4);
#else
          packedIn[inOffset + i * (imageSize + 1) + j] = static_cast<ExtMemWord>(trainLabels[countOffset + i]);
#endif
        }
      }
    }

    // copy inputs to accelerator
    thePlatform->copyBufferHostToAccel((void *)packedIn, accelBufIn, sizeof(ExtMemWord) * packedInSize);

    //auto t1 = std::chrono::high_resolution_clock::now();
    // call the accelerator in compute mode
    ExecAccel();
    //auto t2 = std::chrono::high_resolution_clock::now();

    // copy results back to host
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * packedOutSize);

    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //usecPerImage = (float)duration / (count);

    //std::cout << "1-batch (" << count << " images) training took " << duration << " microseconds, " << usecPerImage << " usec per image, " << 1000000.0 / usecPerImage << " images per second" << ", Test accuracy: " << accuracy << std::endl;

    // get trained weights and biases
    for (unsigned int i = 0; i < W_B_SIZE; i++) {
      packedIn[i] = packedOut[i];
    }
  }

  // put trained weights and biases
  for (unsigned int i = 0; i < W_B_SIZE; i++) {
    result.push_back(static_cast<float>(packedOut[i]));
  }

  // NOTE(wtakase): Need comment out to prevent
  // 'application performed illegal memory access and is being terminated'
  //delete packedIn;
  //delete packedOut;
  //packedIn = 0;
  //packedOut = 0;
  return (result);
}

} // namespace bnn_fc

#endif
