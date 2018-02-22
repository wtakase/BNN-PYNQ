#include "DlUtil.hpp"
#include "DlSgd.hpp"

DlSgd::DlSgd(ExtMemWord lr)
{
  this->lr = lr;
}

void DlSgd::Update(DlTwoLayerNet *network)
{
  for (unsigned int i = 0; i < INPUT_SIZE; i++) {
    for (unsigned int j = 0; j < HIDDEN1_SIZE; j++) {
      network->w1[i * HIDDEN1_SIZE + j] -= network->affine1->dw[i * HIDDEN1_SIZE + j] * lr;
      if (i == 0) {
        network->b1[j] -= network->affine1->db[j] * lr;
      }
    }
  }

  for (unsigned int i = 0; i < HIDDEN1_SIZE; i++) {
    for (unsigned int j = 0; j < OUTPUT_SIZE; j++) {
      network->w2[i * OUTPUT_SIZE + j] -= network->affine2->dw[i * OUTPUT_SIZE + j] * lr;
      if (i == 0) {
        network->b2[j] -= network->affine2->db[j] * lr;
      }
    }
  }
}
