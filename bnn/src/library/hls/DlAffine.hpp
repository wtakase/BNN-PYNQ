#ifndef _DL_AFFINE_HPP
#define _DL_AFFINE_HPP

#include "DlUtil.hpp"

using namespace bnn_fc;

class DlAffine1
{
public:
  static const unsigned int IN_SIZE = INPUT_SIZE;
  static const unsigned int OUT_SIZE = HIDDEN1_SIZE;
  IntMemWord out[BATCH_SIZE * OUT_SIZE];
  IntMemWord dw[W1_SIZE];
  IntMemWord db[B1_SIZE];
#if defined(HLSFIXED) && !defined(HLSNOCAST)
  MulMemWord mulBox;
#endif
  IntMemWord sumBox1;
  IntMemWord sumBox2;

  DlAffine1();

  void Forward(IntMemWord x[BATCH_SIZE * IN_SIZE], IntMemWord w[W1_SIZE], IntMemWord b[B1_SIZE]);

  void Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE], IntMemWord x[BATCH_SIZE * IN_SIZE]);
};

class DlAffine2
{
public:
  static const unsigned int IN_SIZE = HIDDEN1_SIZE;
  static const unsigned int OUT_SIZE = OUTPUT_SIZE;
  IntMemWord out[BATCH_SIZE * OUT_SIZE];
  IntMemWord dx[BATCH_SIZE * IN_SIZE];
  IntMemWord dw[W2_SIZE];
  IntMemWord db[B2_SIZE];
#if defined(HLSFIXED) && !defined(HLSNOCAST)
  MulMemWord mulBox;
#endif
  IntMemWord sumBox1;
  IntMemWord sumBox2;

  DlAffine2();

  void Forward(IntMemWord x[BATCH_SIZE * IN_SIZE], IntMemWord w[W2_SIZE], IntMemWord b[B2_SIZE]);

  void Backward(IntMemWord dout[BATCH_SIZE * OUT_SIZE], IntMemWord x[BATCH_SIZE * IN_SIZE], IntMemWord w[W2_SIZE]);
};

#endif
