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
 * Host code for BNN, overlay ADD-Double-Pynq, to manage parameter loading, 
 * classification (inference) of single and multiple images
 * 
 *
 *****************************************************************************/
#include <iostream>
#include <string.h>
#include "foldedmv-offload-double.h"


extern "C" double* add_double(double in1, double in2, float *usecPerImage)
{
  bnn_double::FoldedMVInit("add-double-pynq");
  std::vector<double> add_double_result;
  float usecPerImage_int;
  std::cout << "add_double(): in1: " << in1 << ", in2: " << in2 << std::endl;
  add_double_result = bnn_double::addTwoValues<0>(in1, in2, usecPerImage_int);
  double *result = new double[1];
  std::copy(add_double_result.begin(), add_double_result.end(), result);
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
  bnn_double::FoldedMVDeinit();
}
