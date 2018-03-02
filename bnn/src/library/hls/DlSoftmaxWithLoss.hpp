#ifndef _DL_SOFTMAX_WITH_LOSS_HPP
#define _DL_SOFTMAX_WITH_LOSS_HPP

using namespace bnn_fc;

class DlSoftmaxWithLoss
{
public:
  static const unsigned int SIZE = OUTPUT_SIZE;
  ExtMemWord out[BATCH_SIZE * SIZE];
  ExtMemWord dx[BATCH_SIZE * SIZE];
  ExtMemWord loss;
  ExtMemWord *t;
  MulMemWord mulBox;

  DlSoftmaxWithLoss();

  void SoftmaxWithLoss(ExtMemWord x[BATCH_SIZE * SIZE]);

  ExtMemWord CrossEntropyError();

  ExtMemWord Forward(ExtMemWord x[BATCH_SIZE * SIZE], ExtMemWord t[BATCH_SIZE * SIZE]);

  void Backward();
};

#endif
