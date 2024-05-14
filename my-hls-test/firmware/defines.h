#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 100
#define N_INPUT_2_1 100
#define N_INPUT_3_1 3
#define OUT_HEIGHT_10 100
#define OUT_WIDTH_10 100
#define N_FILT_10 25
#define OUT_HEIGHT_2 100
#define OUT_WIDTH_2 100
#define N_FILT_2 25
#define OUT_HEIGHT_11 100
#define OUT_WIDTH_11 100
#define N_FILT_11 25
#define OUT_HEIGHT_4 100
#define OUT_WIDTH_4 100
#define N_FILT_4 25
#define OUT_HEIGHT_12 100
#define OUT_WIDTH_12 100
#define N_FILT_12 25
#define OUT_HEIGHT_6 100
#define OUT_WIDTH_6 100
#define N_FILT_6 25
#define OUT_HEIGHT_8 50
#define OUT_WIDTH_8 50
#define N_FILT_8 25
#define OUT_HEIGHT_9 100
#define OUT_WIDTH_9 100
#define N_CHAN_9 25

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 3*1> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer10_t;
typedef ap_fixed<16,6> conv2d_weight_t;
typedef ap_fixed<16,6> conv2d_bias_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer3_t;
typedef ap_fixed<18,8> conv2d_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer11_t;
typedef ap_fixed<16,6> conv2d_1_weight_t;
typedef ap_fixed<16,6> conv2d_1_bias_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer5_t;
typedef ap_fixed<18,8> conv2d_1_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer12_t;
typedef ap_fixed<16,6> conv2d_2_weight_t;
typedef ap_fixed<16,6> conv2d_2_bias_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer7_t;
typedef ap_fixed<18,8> conv2d_2_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer8_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> result_t;

#endif
