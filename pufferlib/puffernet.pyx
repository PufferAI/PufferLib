# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=False

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrtf
from cpython.list cimport PyList_GET_ITEM
cimport cython
cimport numpy as cnp
import numpy as np

cdef extern from "puffernet.h":
    void _linear(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim)
    void _relu(float* input, float* output,int size)
    float _sigmoid(float x)
    void _conv2d(float* input, float* weights, float* bias,
        float* output, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride)
    void _lstm(float* input, float* state_h, float* state_c, float* weights_input,
        float* weights_state, float* bias_input, float*bias_state,
        float *buffer, int batch_size, int input_size, int hidden_size)
    void _one_hot(int* input, int* output, int batch_size,
        int input_size, int num_classes)
    void _cat_dim1(float* x, float* y, float* output,
        int batch_size, int x_size, int y_size)
    void _argmax_multidiscrete(float* input, int* output,
        int batch_size, int logit_sizes[], int num_actions)

def puf_linear_layer(cnp.ndarray input, cnp.ndarray weights, cnp.ndarray bias, cnp.ndarray output,
        int batch_size, int input_dim, int output_dim):
    _linear(<float*> input.data, <float*> weights.data, <float*> bias.data,
        <float*> output.data, batch_size, input_dim, output_dim)

def puf_relu(cnp.ndarray input, cnp.ndarray output, int size):
    _relu(<float*> input.data, <float*> output.data, size)

def puf_sigmoid(float x):
    return _sigmoid(x)

def puf_convolution_layer(cnp.ndarray input, cnp.ndarray weights, cnp.ndarray bias,
        cnp.ndarray output, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride):
    _conv2d(<float*> input.data, <float*> weights.data, <float*> bias.data,
        <float*> output.data, batch_size, in_width, in_height, in_channels, out_channels,
        kernel_size, stride)

def puf_lstm(cnp.ndarray input, cnp.ndarray state_h, cnp.ndarray state_c, cnp.ndarray weights_input,
        cnp.ndarray weights_state, cnp.ndarray bias_input, cnp.ndarray bias_state,
        cnp.ndarray buffer, int batch_size, int input_size, int hidden_size):
    _lstm(<float*> input.data, <float*> state_h.data, <float*> state_c.data,
        <float*> weights_input.data, <float*> weights_state.data, <float*> bias_input.data,
        <float*> bias_state.data, <float*> buffer.data, batch_size, input_size, hidden_size)

def puf_one_hot(cnp.ndarray input, cnp.ndarray output, int batch_size, int input_size, int num_classes):
    _one_hot(<int*> input.data, <int*> output.data, batch_size, input_size, num_classes)

def puf_cat_dim1(cnp.ndarray x, cnp.ndarray y, cnp.ndarray output, int batch_size, int x_size, int y_size):
    _cat_dim1(<float*> x.data, <float*> y.data, <float*> output.data, batch_size, x_size, y_size)

def puf_argmax_multidiscrete(cnp.ndarray input, cnp.ndarray output,
        int batch_size, cnp.ndarray logit_sizes, int num_actions):
    _argmax_multidiscrete(<float*> input.data, <int*> output.data,
        batch_size, <int*> logit_sizes.data, num_actions)

