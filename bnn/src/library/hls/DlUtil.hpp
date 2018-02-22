#ifndef _DL_UTIL_HPP
#define _DL_UTIL_HPP

#if defined(HLSHALF)

#include "hls_half.h"

namespace bnn_fc
{
//typedef half ExtMemWord;
typedef float ExtMemWord;
const unsigned int bytesPerExtMemWord = sizeof(ExtMemWord);
const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord) * 8;
} // namespace bnn_fc

#else

namespace bnn_fc
{
typedef float ExtMemWord;
const unsigned int bytesPerExtMemWord = sizeof(ExtMemWord);
const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord) * 8;
} // namespace bnn_fc

#endif

static const unsigned int INPUT_SIZE = 784;
static const unsigned int HIDDEN_LAYER_NUM = 1;
static const unsigned int HIDDEN1_SIZE = 25;
static const unsigned int OUTPUT_SIZE = 10;
static const unsigned int BATCH_SIZE = 40;
static const unsigned int MAX_SIZE = INPUT_SIZE;
static const double WEIGHT_INIT_STD = 0.01;
static const bnn_fc::ExtMemWord LEARNING_RATE = 0.01;
static const unsigned int TRAIN_SIZE = 60000;
static const unsigned int TEST_SIZE = 10000;
static const unsigned int STEPS_PER_EPOCH = TRAIN_SIZE / BATCH_SIZE;
static const unsigned int MAX_EPOCHS = 3;
static const unsigned int MAX_ITERATIONS = STEPS_PER_EPOCH * MAX_EPOCHS;
static const unsigned int W1_SIZE = INPUT_SIZE * HIDDEN1_SIZE;
static const unsigned int B1_SIZE = HIDDEN1_SIZE;
static const unsigned int W2_SIZE = HIDDEN1_SIZE * OUTPUT_SIZE;
static const unsigned int B2_SIZE = OUTPUT_SIZE;
static const unsigned int W_B_SIZE = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE;

#endif
