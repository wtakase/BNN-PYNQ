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
 * Host code for BNN, overlay ADD-Pynq, to manage parameter loading, 
 * classification (inference) of single and multiple images
 * 
 *
 *****************************************************************************/
#include <iostream>
#include <string.h>
#include "foldedmv-offload.h"

using namespace std;


extern "C" unsigned int* add(unsigned int in1, unsigned int in2, float *usecPerImage)
{
  FoldedMVInit("add-pynq");
  std::vector<unsigned int> add_result;
  float usecPerImage_int;
  cout << "add(): in1: " << in1 << ", in2: " << in2 << endl;
  add_result = addTwoValues<0>(in1, in2, usecPerImage_int);
  unsigned int *result = new unsigned int [1];
  std::copy(add_result.begin(),add_result.end(), result);
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return result;
}

extern "C" void free_results(unsigned int * result)
{
  delete[] result;
}

extern "C" void deinit() {
  FoldedMVDeinit();
}
