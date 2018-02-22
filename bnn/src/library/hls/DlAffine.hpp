#ifndef _DL_AFFINE_HPP
#define _DL_AFFINE_HPP

#include "DlUtil.hpp"

using namespace bnn_fc;

class DlAffine1
{
public:
  static const unsigned int IN_SIZE = INPUT_SIZE;
  static const unsigned int OUT_SIZE = HIDDEN1_SIZE;
  ExtMemWord *x;
  ExtMemWord *w;
  ExtMemWord *b;
  ExtMemWord out[BATCH_SIZE * OUT_SIZE];
  ExtMemWord dw[W1_SIZE];
  ExtMemWord db[B1_SIZE];

  DlAffine1(ExtMemWord w[W1_SIZE], ExtMemWord b[B1_SIZE]);

  void Forward(ExtMemWord x[BATCH_SIZE * IN_SIZE]);

  void Backward(ExtMemWord dout[BATCH_SIZE * OUT_SIZE]);
};

class DlAffine2
{
public:
  static const unsigned int IN_SIZE = HIDDEN1_SIZE;
  static const unsigned int OUT_SIZE = OUTPUT_SIZE;
  ExtMemWord *x;
  ExtMemWord *w;
  ExtMemWord *b;
  ExtMemWord out[BATCH_SIZE * OUT_SIZE];
  ExtMemWord dx[BATCH_SIZE * IN_SIZE];
  ExtMemWord dw[W2_SIZE];
  ExtMemWord db[B2_SIZE];

  DlAffine2(ExtMemWord w[W2_SIZE], ExtMemWord b[B2_SIZE]);

  void Forward(ExtMemWord x[BATCH_SIZE * IN_SIZE]);

  void Backward(ExtMemWord dout[BATCH_SIZE * OUT_SIZE]);
};

#endif
