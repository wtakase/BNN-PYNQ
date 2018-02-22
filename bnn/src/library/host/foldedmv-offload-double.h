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
 * @file foldedmv-offload-double.h
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
#include "DlRelu.hpp"
#include "DlSoftmaxWithLoss.hpp"
#include "DlTwoLayerNet.hpp"

namespace bnn_double
{

typedef double ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord) * 8;


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

} // namespace bnn_double

#if defined(OFFLOAD) && defined(RAWHLS)
#include "bnn-library-double.h"

namespace bnn_double
{

void BlackBoxJam(double *in, double *out);

extern ExtMemWord *bufIn, *bufOut;

template<unsigned int dummy>
std::vector<double> addTwoValues(const double in1, const double in2, float &usecPerImage) {
  const unsigned int count = 1;
  std::vector<double> results;
  std::cout << "addTwoValues(): in1: " << in1 << ", in2: " << in2 << std::endl;
  ExtMemWord *packedIns = new ExtMemWord[2];
  ExtMemWord *packedOut = new ExtMemWord[1];
  packedIns[0] = in1;
  packedIns[1] = in2;
  std::cout << "addTwoValues(): packedIns[0]: " << packedIns[0] << ", packedIns[1]: " << packedIns[1] << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((double *)packedIns, (double *)packedOut);
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "addTwoValues(): packedOut[0]: " << packedOut[0] << std::endl;
  results.push_back((double)packedOut[0]);

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  std::cout << "addTwoValues(): Addition took " << duration << " microseconds" << std::endl;
  delete [] packedIns;
  delete [] packedOut;
  return (results);
}

} // namespace bnn_double

#elif defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

namespace bnn_double
{

extern DonutDriver * thePlatform;
extern void *accelBufIn, *accelBufOut;
extern ExtMemWord *bufIn, *bufOut;

void ExecAccel();

template<unsigned int dummy>
std::vector<double> addTwoValues(const double in1, const double in2, float &usecPerImage) {
  const unsigned int count = 1;
  std::vector<double> results;
  std::cout << "addTwoValues(): in1: " << in1 << ", in2: " << in2 << std::endl;
  ExtMemWord *packedIns = new ExtMemWord[2];
  ExtMemWord *packedOut = new ExtMemWord[1];
  packedIns[0] = in1;
  packedIns[1] = in2;
  std::cout << "addTwoValues(): packedIns[0]: " << packedIns[0] << ", packedIns[1]: " << packedIns[1] << std::endl;

  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedIns, accelBufIn, sizeof(ExtMemWord) * count * 2);
  thePlatform->writeJamRegAddr(0x54, count);

  auto t1 = std::chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = std::chrono::high_resolution_clock::now();

  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * 1);

  std::cout << "addTwoValues(): packedOut[0]: " << packedOut[0] << std::endl;
  results.push_back((double)packedOut[0]);

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  std::cout << "addTwoValues(): Addition took " << duration << " microseconds" << std::endl;
  delete [] packedIns;
  delete [] packedOut;
  return (results);
}

} // namespace bnn_double

#endif
