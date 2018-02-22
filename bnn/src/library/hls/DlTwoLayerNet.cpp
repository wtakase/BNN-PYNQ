#include "DlUtil.hpp"
#include "DlTwoLayerNet.hpp"

DlTwoLayerNet::DlTwoLayerNet(ExtMemWord w1[W1_SIZE], ExtMemWord b1[B1_SIZE], ExtMemWord w2[W2_SIZE], ExtMemWord b2[B2_SIZE])
{
  this->w1 = w1;
  this->b1 = b1;
  this->w2 = w2;
  this->b2 = b2;

  // Define layers
  DlAffine1 affine1(w1, b1);
  DlRelu1 relu1;
  DlAffine2 affine2(w2, b2);
  DlSoftmaxWithLoss softmaxWithLoss;

  this->affine1 = &affine1;
  this->relu1 = &relu1;
  this->affine2 = &affine2;
  this->softmaxWithLoss = &softmaxWithLoss;
}

void DlTwoLayerNet::Predict(ExtMemWord x[BATCH_SIZE * IN_SIZE])
{
  this->x = x;
  affine1->Forward(x);
  relu1->Forward(affine1->out);
  affine2->Forward(relu1->out);
}

void DlTwoLayerNet::Loss(ExtMemWord x[BATCH_SIZE * IN_SIZE], ExtMemWord t[BATCH_SIZE * OUT_SIZE])
{
  Predict(x);
  softmaxWithLoss->Forward(affine2->out, t);
}

void DlTwoLayerNet::Gradient(ExtMemWord x[BATCH_SIZE * IN_SIZE], ExtMemWord t[BATCH_SIZE * OUT_SIZE])
{
  // Forward
  Loss(x, t);

  // Backward
  softmaxWithLoss->Backward();
  affine2->Backward(softmaxWithLoss->dx);
  relu1->Backward(affine2->dx);
  affine1->Backward(relu1->dx);
}
