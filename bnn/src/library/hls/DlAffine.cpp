#include "DlUtil.hpp"
#include "DlAffine.hpp"

DlAffine1::DlAffine1(IntMemWord w[W1_SIZE], IntMemWord b[B1_SIZE])
{
  this->w = w;
  this->b = b;
}

void DlAffine1::Forward(IntMemWord x[BATCH_SIZE * IN_SIZE])
{
  this->x = x;

  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < IN_SIZE; k++) {
        if (k == 0) {
          out[i * OUT_SIZE + j] = b[j];
        }
#if defined(HLSFIXED) && !defined(HLSNOCAST)
        mulBox = (MulMemWord)x[i * IN_SIZE + k] * (MulMemWord)w[k * OUT_SIZE + j];
        out[i * OUT_SIZE + j] += (IntMemWord)mulBox;
#else
        out[i * OUT_SIZE + j] += x[i * IN_SIZE + k] * w[k * OUT_SIZE + j];
#endif
      }
    }
  }
}

void DlAffine1::Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE])
{
  for (unsigned int i = 0; i < IN_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < BATCH_SIZE; k++) {
        if (k == 0) {
          dw[i * OUT_SIZE + j] = 0;
        }
#if defined(HLSFIXED) && !defined(HLSNOCAST)
        mulBox = (MulMemWord)x[k * IN_SIZE + i] * (MulMemWord)dout[k * OUT_SIZE + j];
        dw[i * OUT_SIZE + j] += (IntMemWord)mulBox;
#else
        dw[i * OUT_SIZE + j] += x[k * IN_SIZE + i] * dout[k * OUT_SIZE + j];
#endif
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


DlAffine2::DlAffine2(IntMemWord w[W2_SIZE], IntMemWord b[B2_SIZE])
{
  this->w = w;
  this->b = b;
}

void DlAffine2::Forward(IntMemWord x[BATCH_SIZE * IN_SIZE])
{
  this->x = x;

  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < IN_SIZE; k++) {
        if (k == 0) {
          out[i * OUT_SIZE + j] = b[j];
        }
#if defined(HLSFIXED) && !defined(HLSNOCAST)
        mulBox = (MulMemWord)x[i * IN_SIZE + k] * (MulMemWord)w[k * OUT_SIZE + j];
        out[i * OUT_SIZE + j] += (IntMemWord)mulBox;
#else
        out[i * OUT_SIZE + j] += x[i * IN_SIZE + k] * w[k * OUT_SIZE + j];
#endif
      }
    }
  }
}

void DlAffine2::Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE])
{
  for (unsigned int i = 0; i < BATCH_SIZE; i++) {
    for (unsigned int j = 0; j < IN_SIZE; j++) {
      for (unsigned int k = 0; k < OUT_SIZE; k++) {
        if (k == 0) {
          dx[i * IN_SIZE + j] = 0;
        }
#if defined(HLSFIXED) && !defined(HLSNOCAST)
        mulBox = (MulMemWord)dout[i * OUT_SIZE + k] * (MulMemWord)w[j * OUT_SIZE + k];
        dx[i * IN_SIZE + j] += (IntMemWord)mulBox;
#else
        dx[i * IN_SIZE + j] += dout[i * OUT_SIZE + k] * w[j * OUT_SIZE + k];
#endif
      }
    }
  }

  for (unsigned int i = 0; i < IN_SIZE; i++) {
    for (unsigned int j = 0; j < OUT_SIZE; j++) {
      for (unsigned int k = 0; k < BATCH_SIZE; k++) {
        if (k == 0) {
          dw[i * OUT_SIZE + j] = 0;
        }
#if defined(HLSFIXED) && !defined(HLSNOCAST)
        mulBox = (MulMemWord)x[k * IN_SIZE + i] * (MulMemWord)dout[k * OUT_SIZE + j];
        dw[i * OUT_SIZE + j] += (IntMemWord)mulBox;
#else
        dw[i * OUT_SIZE + j] += x[k * IN_SIZE + i] * dout[k * OUT_SIZE + j];
#endif
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
