#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &conv2d_input,
    hls::stream<result_t> &layer9_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=conv2d_input,layer9_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv2d_weight_t, 75>(w10, "w10.txt");
        nnet::load_weights_from_txt<conv2d_bias_t, 25>(b10, "b10.txt");
        nnet::load_weights_from_txt<conv2d_1_weight_t, 625>(w11, "w11.txt");
        nnet::load_weights_from_txt<conv2d_1_bias_t, 25>(b11, "b11.txt");
        nnet::load_weights_from_txt<conv2d_2_weight_t, 625>(w12, "w12.txt");
        nnet::load_weights_from_txt<conv2d_2_bias_t, 25>(b12, "b12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=10000
    nnet::pointwise_conv_2d_cl<input_t, layer10_t, config10>(conv2d_input, layer10_out, w10, b10); // conv2d

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=10000
    nnet::relu<layer10_t, layer3_t, relu_config3>(layer10_out, layer3_out); // conv2d_relu

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=10000
    nnet::pointwise_conv_2d_cl<layer3_t, layer11_t, config11>(layer3_out, layer11_out, w11, b11); // conv2d_1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=10000
    nnet::relu<layer11_t, layer5_t, relu_config5>(layer11_out, layer5_out); // conv2d_1_relu

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=10000
    nnet::pointwise_conv_2d_cl<layer5_t, layer12_t, config12>(layer5_out, layer12_out, w12, b12); // conv2d_2

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=10000
    nnet::relu<layer12_t, layer7_t, relu_config7>(layer12_out, layer7_out); // conv2d_2_relu

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=2500
    nnet::pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // max_pooling2d

    nnet::resize_nearest<layer8_t, config9>(layer8_out, layer9_out); // up_sampling2d

}
