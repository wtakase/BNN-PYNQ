#ifndef _DL_SGD_HPP
#define _DL_SGD_HPP

#include "DlUtil.hpp"
#include "DlTwoLayerNet.hpp"

using namespace bnn_fc;

class DlSgd
{
private:
  ExtMemWord lr;

public:
  DlSgd(ExtMemWord lr);

  void Update(DlTwoLayerNet *network);
};

#endif
