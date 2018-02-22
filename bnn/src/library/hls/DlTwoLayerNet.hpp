#ifndef _DL_TWO_LAYER_NET_HPP
#define _DL_TWO_LAYER_NET_HPP

#include "DlAffine.hpp"
#include "DlRelu.hpp"
#include "DlSoftmaxWithLoss.hpp"

using namespace bnn_fc;

class DlTwoLayerNet
{
public:
  static const unsigned int IN_SIZE = INPUT_SIZE;
  static const unsigned int OUT_SIZE = OUTPUT_SIZE;
  ExtMemWord *w1;
  ExtMemWord *b1;
  ExtMemWord *w2;
  ExtMemWord *b2;
  DlAffine1 *affine1;
  DlRelu1 *relu1;
  DlAffine2 *affine2;
  DlSoftmaxWithLoss *softmaxWithLoss;
  ExtMemWord *x;

  DlTwoLayerNet(ExtMemWord w1[W1_SIZE], ExtMemWord b1[B1_SIZE], ExtMemWord w2[W2_SIZE], ExtMemWord b2[B2_SIZE]);

  void Predict(ExtMemWord x[BATCH_SIZE * IN_SIZE]);

  void Loss(ExtMemWord x[BATCH_SIZE * IN_SIZE], ExtMemWord t[BATCH_SIZE * OUT_SIZE]);

  void Gradient(ExtMemWord x[BATCH_SIZE * IN_SIZE], ExtMemWord t[BATCH_SIZE * OUT_SIZE]);
};

#endif
