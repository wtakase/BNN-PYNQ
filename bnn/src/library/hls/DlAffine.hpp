#ifndef _DL_AFFINE_HPP
#define _DL_AFFINE_HPP

#include "DlUtil.hpp"

using namespace bnn_fc;

class DlAffine1
{
public:
  static const unsigned int IN_SIZE = INPUT_SIZE;
  static const unsigned int OUT_SIZE = HIDDEN1_SIZE;
  IntMemWord *x;
  IntMemWord *w;
  IntMemWord *b;
  IntMemWord out[BATCH_SIZE * OUT_SIZE];
  IntMemWord dw[W1_SIZE];
  IntMemWord db[B1_SIZE];
  MulMemWord mulBox;

  DlAffine1(IntMemWord w[W1_SIZE], IntMemWord b[B1_SIZE]);

  void Forward(IntMemWord x[BATCH_SIZE * IN_SIZE]);

  void Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE]);
};

class DlAffine2
{
public:
  static const unsigned int IN_SIZE = HIDDEN1_SIZE;
  static const unsigned int OUT_SIZE = OUTPUT_SIZE;
  IntMemWord *x;
  IntMemWord *w;
  IntMemWord *b;
  IntMemWord out[BATCH_SIZE * OUT_SIZE];
  IntMemWord dx[BATCH_SIZE * IN_SIZE];
  IntMemWord dw[W2_SIZE];
  IntMemWord db[B2_SIZE];
  MulMemWord mulBox;

  DlAffine2(IntMemWord w[W2_SIZE], IntMemWord b[B2_SIZE]);

  void Forward(IntMemWord x[BATCH_SIZE * IN_SIZE]);

  void Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE]);
};

#endif
