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
 * @file rawhls-offload-fc.h
 *
 * Library of functions for compatible execution of HLS source code (SW execution)
 * 
 *
 *****************************************************************************/
#if defined(RAWHLS) && defined(OFFLOAD)
#include "foldedmv-offload-fc.h"
#include <string.h>
#include <vector>
#include <iostream>

namespace bnn_fc
{

ExtMemWord *bufIn, *bufOut;

void FoldedMVInit(const char * attachName) {
  if (!bufIn) {
    bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
    if (!bufIn) throw "Failed to allocate host buffer";
  }
  if (!bufOut) {
    bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
    if (!bufOut) throw "Failed to allocate host buffer";
  }
}

void FoldedMVDeinit() {
  // NOTE(wtakase): Not sure why these cause 'double free or corruption'
  //delete bufIn;
  //delete bufOut;
  //bufIn = 0;
  //bufOut = 0;
}

} // namespace bnn_fc

#endif
