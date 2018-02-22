#include "DlUtil.hpp"
#include "DlSoftmaxWithLoss.hpp"
#include <algorithm>

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
      ExtMemWord expXSubXmax = std::exp(x[i * SIZE + j] - x[maxIndex]);
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
      sum += t[i * SIZE + j] * log(out[i * SIZE + j] + 1e-7);
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
      dx[i * SIZE + j] = (out[i * SIZE + j] - t[i * SIZE + j]) / BATCH_SIZE;
    }
  }
}
