#include "DlUtil.hpp"
#include "DlAffine.hpp"

DlAffine1::DlAffine1(ExtMemWord w[W1_SIZE], ExtMemWord b[B1_SIZE])
{
  this->w = w;
  this->b = b;
}

void DlAffine1::Forward(ExtMemWord x[BATCH_SIZE * IN_SIZE])
{
  this->x = x;

  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < IN_SIZE; k++) {
        if (k == 0) {
          out[i * OUT_SIZE + j] = b[j];
        }
        out[i * OUT_SIZE + j] += x[i * IN_SIZE + k] * w[k * OUT_SIZE + j];
      }
    }
  }
}

void DlAffine1::Backward(ExtMemWord dout[BATCH_SIZE * OUT_SIZE])
{
  for (unsigned int i = 0; i < IN_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < BATCH_SIZE; k++) {
        if (k == 0) {
          dw[i * OUT_SIZE + j] = 0;
        }
        dw[i * OUT_SIZE + j] += x[k * IN_SIZE + i] * dout[k * OUT_SIZE + j];
        if (i == 0) {
          if (k == 0) {
            db[j] = 0;
          }
          db[j] += dout[k * OUT_SIZE + j];
        }
      }
    }
  }
}


DlAffine2::DlAffine2(ExtMemWord w[W2_SIZE], ExtMemWord b[B2_SIZE])
{
  this->w = w;
  this->b = b;
}

void DlAffine2::Forward(ExtMemWord x[BATCH_SIZE * IN_SIZE])
{
  this->x = x;

  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < IN_SIZE; k++) {
        if (k == 0) {
          out[i * OUT_SIZE + j] = b[j];
        }
        out[i * OUT_SIZE + j] += x[i * IN_SIZE + k] * w[k * OUT_SIZE + j];
      }
    }
  }
}

void DlAffine2::Backward(ExtMemWord dout[BATCH_SIZE * OUT_SIZE])
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < IN_SIZE; j++) {
      for (unsigned int k = 0; k < OUT_SIZE; k++) {
        if (k == 0) {
          dx[i * IN_SIZE + j] = 0;
        }
        dx[i * IN_SIZE + j] += dout[i * OUT_SIZE + k] * w[j * OUT_SIZE + k];
      }
    }
  }

  for (unsigned int i = 0; i < IN_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < BATCH_SIZE; k++) {
        if (k == 0) {
          dw[i * OUT_SIZE + j] = 0;
        }
        dw[i * OUT_SIZE + j] += x[k * IN_SIZE + i] * dout[k * OUT_SIZE + j];
        if (i == 0) {
          if (k == 0) {
            db[j] = 0;
          }
          db[j] += dout[k * OUT_SIZE + j];
        }
      }
    }
  }
}
