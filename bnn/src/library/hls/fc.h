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
 * @file fc.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file implement the BNN add layer 
 * 
 *
 *****************************************************************************/

#include "DlAffine.hpp"
#include "DlRelu.hpp"
#include "DlSoftmaxWithLoss.hpp"

namespace bnn_fc
{

template<unsigned int SIZE_PER_IMAGE>
void StreamingTrain_Batch(hls::stream<ExtMemWord> &in, hls::stream<ExtMemWord> &out) {
  ExtMemWord w1[W1_SIZE];
  ExtMemWord b1[B1_SIZE];
  ExtMemWord w2[W2_SIZE];
  ExtMemWord b2[B2_SIZE];
  for (unsigned int i = 0; i < W1_SIZE; i++) {
    w1[i] = in.read();
  }
  for (unsigned int i = 0; i < B1_SIZE; i++) {
    b1[i] = in.read();
  }
  for (unsigned int i = 0; i < W2_SIZE; i++) {
    w2[i] = in.read();
  }
  for (unsigned int i = 0; i < B2_SIZE; i++) {
    b2[i] = in.read();
  }

  ExtMemWord xTrain[INPUT_SIZE * BATCH_SIZE];
  ExtMemWord tTrain[OUTPUT_SIZE * BATCH_SIZE];

  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < INPUT_SIZE; j++) {
      xTrain[i * INPUT_SIZE + j] = in.read();
    }
#if defined(HLSFIXED)
    ExtMemWord label = in.read();
    for (unsigned int j = 0; j < OUTPUT_SIZE; j++) {
      ExtMemWord iterLabel = (ExtMemWord)((ShiftMemWord)j >> 4);
      if (label == iterLabel) {
        tTrain[i * OUTPUT_SIZE + j] = 1.0;
      } else {
        tTrain[i * OUTPUT_SIZE + j] = 0.0;
      }
    }
#else
    unsigned int label = (unsigned int)in.read();
    for (unsigned int j = 0; j < OUTPUT_SIZE; j++) {
      if (label == j) {
        tTrain[i * OUTPUT_SIZE + j] = 1.0;
      } else {
        tTrain[i * OUTPUT_SIZE + j] = 0.0;
      }
    }
#endif
  }

  // Create Two-layer network
  DlAffine1 affine1(w1, b1);
  DlRelu1 relu1;
  DlAffine2 affine2(w2, b2);
  DlSoftmaxWithLoss softmaxWithLoss;

  // Train
  affine1.Forward(xTrain);
  relu1.Forward(affine1.out);
  affine2.Forward(relu1.out);
  softmaxWithLoss.Forward(affine2.out, tTrain);
  softmaxWithLoss.Backward();
  affine2.Backward(softmaxWithLoss.dx);
  relu1.Backward(affine2.dx);
  affine1.Backward(relu1.dx);

  // Update parameters
  for (unsigned int i = 0; i < INPUT_SIZE; i++) {
    for (unsigned int j = 0; j < HIDDEN1_SIZE; j++) {
      w1[i * HIDDEN1_SIZE + j] -= affine1.dw[i * HIDDEN1_SIZE + j] * LEARNING_RATE;
      if (i == 0) {
        b1[j] -= affine1.db[j] * LEARNING_RATE;
      }
    }
  }

  for (unsigned int i = 0; i < HIDDEN1_SIZE; i++) {
    for (unsigned int j = 0; j < OUTPUT_SIZE; j++) {
      w2[i * OUTPUT_SIZE + j] -= affine2.dw[i * OUTPUT_SIZE + j] * LEARNING_RATE;
      if (i == 0) {
        b2[j] -= affine2.db[j] * LEARNING_RATE;
      }
    }
  }

  for (unsigned int i = 0; i < W1_SIZE; i++) {
    out.write(w1[i]);
  }
  for (unsigned int i = 0; i < B1_SIZE; i++) {
    out.write(b1[i]);
  }
  for (unsigned int i = 0; i < W2_SIZE; i++) {
    out.write(w2[i]);
  }
  for (unsigned int i = 0; i < B2_SIZE; i++) {
    out.write(b2[i]);
  }
}

} // namespace bnn_fc
