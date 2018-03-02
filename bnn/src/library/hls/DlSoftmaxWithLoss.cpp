#include "DlUtil.hpp"
#include "DlSoftmaxWithLoss.hpp"
#include <algorithm>

#if defined(FPGA)

#include "hls_math.h"

float expWrapper(float in) {
  return hls::exp(in);
}

float logWrapper(float in) {
  return hls::log(in);
}


#else

float expWrapper(float in) {
  return std::exp(in);
}

float logWrapper(float in) {
  return std::log(in);
}

#endif

DlSoftmaxWithLoss::DlSoftmaxWithLoss()
{
}

void DlSoftmaxWithLoss::SoftmaxWithLoss(ExtMemWord x[BATCH_SIZE * SIZE])
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    unsigned int maxIndex = i * SIZE;
    for (unsigned int j = 0; j < SIZE; j++) {
      if (x[i * SIZE + j] > x[maxIndex]) {
        maxIndex = i * SIZE + j;
      }
    }
    ExtMemWord expXSubXmaxSum = 0;
    for (unsigned int j = 0; j < SIZE; j++) {
      float xSubXMaxFloat = x[i * SIZE + j] - x[maxIndex];
      ExtMemWord expXSubXmax = expWrapper(xSubXMaxFloat);
      out[i * SIZE + j] = expXSubXmax;
      expXSubXmaxSum += expXSubXmax;
    }
    for (unsigned int j = 0; j < SIZE; j++) {
      out[i * SIZE + j] /= expXSubXmaxSum;
    }
  }
}

ExtMemWord DlSoftmaxWithLoss::CrossEntropyError()
{
  ExtMemWord sum = 0;
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < SIZE; j++) {
      float outFloat = out[i * SIZE + j];
      outFloat += 1e-7;
      sum += t[i * SIZE + j] * (ExtMemWord)logWrapper(outFloat);
    }
  }
  return -sum / BATCH_SIZE;
}

ExtMemWord DlSoftmaxWithLoss::Forward(ExtMemWord x[BATCH_SIZE * SIZE], ExtMemWord t[BATCH_SIZE * SIZE])
{
  this->t = t;
  DlSoftmaxWithLoss::SoftmaxWithLoss(x);
  return DlSoftmaxWithLoss::CrossEntropyError();
}

void DlSoftmaxWithLoss::Backward()
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < SIZE; j++) {
      mulBox = (MulMemWord)(out[i * SIZE + j] - t[i * SIZE + j]) / (MulMemWord)BATCH_SIZE;
      dx[i * SIZE + j] = (ExtMemWord)mulBox;
    }
  }
}
