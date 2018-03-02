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
 * @file foldedmv-offload-fc.cpp
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/
#include "foldedmv-offload-fc.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

namespace bnn_fc
{

//#define INCLUDE_TRANSFER_TIMES_IN_BENCHMARK

#if defined(INCLUDE_TRANSFER_TIMES_IN_BENCHMARK) || defined(RAWHLS)
#define TRANSFER_EXCL(x) ;
#define TRANSFER_INCL(x) x;
#else
#define TRANSFER_EXCL(x) x;
#define TRANSFER_INCL(x) ;
#endif


std::string getBNNRoot() {
  char * bnnRoot = getenv ("XILINX_BNN_ROOT");
  if(!bnnRoot)
    throw "XILINX_BNN_ROOT must be set";
  return std::string(bnnRoot);
}

} // namespace bnn_fc

#ifdef OFFLOAD

#endif

#if defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

namespace bnn_fc
{

DonutDriver *thePlatform = 0;
void *accelBufIn, *accelBufOut;
ExtMemWord *bufIn, *bufOut;

// register map for FoldedMV:
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of in_r
//        bit 31~0 - in_r[31:0] (Read/Write)
// 0x14 : Data signal of in_r
//        bit 31~0 - in_r[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of out_r
//        bit 31~0 - out_r[31:0] (Read/Write)
// 0x20 : Data signal of out_r
//        bit 31~0 - out_r[63:32] (Read/Write)
// 0x24 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

void ExecAccel() {
  // invoke accelerator and wait for result
  thePlatform->writeJamRegAddr(0x00, 1);
  while((thePlatform->readJamRegAddr(0x00) & 0x2) == 0) usleep(1);
}

void FoldedMVInit(const char * attachName) {
    thePlatform = initPlatform();
    thePlatform->attach(attachName);

    // allocate input/output buffers
    // TODO should be dynamically sized based on the largest I/O
    if (!bufIn) {
        bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
        if (!bufIn) throw "Failed to allocated host buffer";
    }
    if (!bufOut) {
        bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
        if (!bufOut) throw "Failed to allocated host buffer";
    }
    if (!accelBufIn) {
        accelBufIn = thePlatform->allocAccelBuffer(INPUT_BUF_ENTRIES * sizeof(ExtMemWord));
        if (!accelBufIn) throw "Failed to allocate accel buffer";
        accelBufOut = thePlatform->allocAccelBuffer(OUTPUT_BUF_ENTRIES * sizeof(ExtMemWord));
        if (!accelBufOut) throw "Failed to allocate accel buffer";
    }
    // set up I/O buffer addresses for the accelerator
    thePlatform->write64BitJamRegAddr(0x10, (AccelDblReg)accelBufIn);
    thePlatform->write64BitJamRegAddr(0x1c, (AccelDblReg)accelBufOut);
}

void FoldedMVDeinit() {
    // NOTE(wtakase): Not sure why these cause 'double free or corruption'
    //delete bufIn;
    //delete bufOut;
    //bufIn = 0;
    //bufOut = 0;
    if (thePlatform && accelBufIn) thePlatform->deallocAccelBuffer(accelBufIn);
    accelBufIn = 0;
    if (thePlatform && accelBufOut) thePlatform->deallocAccelBuffer(accelBufOut);
    accelBufOut = 0;
    deinitPlatform(thePlatform);
    thePlatform = 0;
}

} // namespace bnn_fc

#endif
