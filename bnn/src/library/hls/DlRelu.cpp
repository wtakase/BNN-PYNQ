#include "DlUtil.hpp"
#include "DlRelu.hpp"

DlRelu1::DlRelu1()
{
}

void DlRelu1::Forward(ExtMemWord x[BATCH_SIZE * SIZE])
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < SIZE; j++) {
      if (x[i * SIZE + j] <= 0) {
        out[i * SIZE + j] = 0;
      } else {
        out[i * SIZE + j] = x[i * SIZE + j];
      }
    }
  }
}

void DlRelu1::Backward(ExtMemWord dout[BATCH_SIZE * SIZE])
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < SIZE; j++) {
      if (out[i * SIZE + j] == 0) {
        dx[i * SIZE + j] = 0;
      } else {
        dx[i * SIZE + j] = dout[i * SIZE + j];
      }
    }
  }
}
