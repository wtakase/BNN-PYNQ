#ifndef _DL_SOFTMAX_WITH_LOSS_HPP
#define _DL_SOFTMAX_WITH_LOSS_HPP

using namespace bnn_fc;

class DlSoftmaxWithLoss
{
public:
  static const unsigned int SIZE = OUTPUT_SIZE;
  IntMemWord out[BATCH_SIZE * SIZE];
  IntMemWord dx[BATCH_SIZE * SIZE];
  IntMemWord loss;
  IntMemWord *t;
#if defined(HLSFIXED) && !defined(HLSNOCAST)
  MulMemWord mulBox;
#endif

  DlSoftmaxWithLoss();

  void SoftmaxWithLoss(IntMemWord x[BATCH_SIZE * SIZE]);

  IntMemWord CrossEntropyError();

  IntMemWord Forward(IntMemWord x[BATCH_SIZE * SIZE], IntMemWord t[BATCH_SIZE * SIZE]);

  void Backward();
};

#endif
