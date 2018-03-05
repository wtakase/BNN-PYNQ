#ifndef _DL_RELU_HPP
#define _DL_RELU_HPP

#include "DlUtil.hpp"

using namespace bnn_fc;

class DlRelu1
{
public:
  static const unsigned int SIZE = HIDDEN1_SIZE;
  IntMemWord out[BATCH_SIZE * SIZE];
  IntMemWord dx[BATCH_SIZE * SIZE];

  DlRelu1();

  void Forward(IntMemWord x[BATCH_SIZE * SIZE]);

  void Backward(IntMemWord dout[BATCH_SIZE * SIZE]);
};

#endif
