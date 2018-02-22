#ifndef _DL_RELU_HPP
#define _DL_RELU_HPP

#include "DlUtil.hpp"

using namespace bnn_fc;

class DlRelu1
{
public:
  static const unsigned int SIZE = HIDDEN1_SIZE;
  ExtMemWord out[BATCH_SIZE * SIZE];
  ExtMemWord dx[BATCH_SIZE * SIZE];

  DlRelu1();

  void Forward(ExtMemWord x[BATCH_SIZE * SIZE]);

  void Backward(ExtMemWord dout[BATCH_SIZE * SIZE]);
};

#endif
