#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

typedef struct {
    void* data;
    size_t capacity;
    size_t used;
} Arena;

Arena* make_allocator(size_t total_size) {
    void* buffer = calloc(1, total_size + sizeof(Arena));
    Arena* allocator = (Arena*)buffer;
    allocator->data = (void*)((char*)buffer + sizeof(Arena));
    allocator->capacity = total_size;
    allocator->used = 0;
    return allocator;
}

void* alloc(Arena* allocator, size_t size) {
    void* ptr = (void*)((char*)allocator->data + allocator->used);
    if (allocator->used + size > allocator->capacity) {
        return NULL;
    }
    allocator->used += size;
    return ptr;
}

// File format is obained by flattening and concatenating all pytorch layers
typedef struct Weights Weights;
struct Weights {
    float* data;
    int size;
    int idx;
};

void _load_weights(const char* filename, float* weights, size_t num_weights) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
    }
    fseek(file, 0, SEEK_END);
    rewind(file);
    size_t read_size = fread(weights, sizeof(float), num_weights, file);
    fclose(file);
    if (read_size != num_weights) {
        perror("Error reading file");
    }
}

Weights* load_weights(const char* filename, size_t num_weights) {
    Weights* weights = calloc(1, sizeof(Weights) + num_weights*sizeof(float));
    weights->data = (float*)(weights + 1);
    _load_weights(filename, weights->data, num_weights);
    weights->size = num_weights;
    weights->idx = 0;
    return weights;
}

float* get_weights(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    assert(weights->idx <= weights->size);
    return data;
}

// PufferNet implementation of PyTorch functions
// These are tested against the PyTorch implementation
void _relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++)
        output[i] = fmaxf(0.0f, input[i]);
}

float _sigmoid(float x);
inline float _sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void _linear(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++)
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            output[b*output_dim + o] = sum + bias[o];
        }
    }
}

void _linear_accumulate(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++)
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            output[b*output_dim + o] += sum + bias[o];
        }
    }
}

void _conv2d(float* input, float* weights, float* bias,
        float* output, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    int h_out = (in_height - kernel_size)/stride + 1;
    int w_out = (in_width - kernel_size)/stride + 1;
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    int out_adr = (
                        b*out_channels*h_out*w_out
                        + oc*h_out*w_out+ 
                        + h*w_out
                        + w
                    );
                    output[out_adr] = bias[oc];
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_adr = (
                                    b*in_channels*in_height*in_width
                                    + ic*in_height*in_width
                                    + (h*stride + kh)*in_width
                                    + (w*stride + kw)
                                );
                                int weight_adr = (
                                    oc*in_channels*kernel_size*kernel_size
                                    + ic*kernel_size*kernel_size
                                    + kh*kernel_size
                                    + kw
                                );
                                output[out_adr] += input[in_adr]*weights[weight_adr];
                            }
                        }
                    }
               }
            }
        }
    }
}

void _conv3d(float* input, float* weights, float* bias,
        float* output, int batch_size, int in_width, int in_height, int in_depth,
        int in_channels, int out_channels, int kernel_size, int stride) {
    int d_out = (in_depth - kernel_size)/stride + 1;
    int h_out = (in_height - kernel_size)/stride + 1;
    int w_out = (in_width - kernel_size)/stride + 1;
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int d = 0; d < d_out; d++) {
                for (int h = 0; h < h_out; h++) {
                    for (int w = 0; w < w_out; w++) {
                        int out_adr = (
                            b*out_channels*d_out*h_out*w_out
                            + oc*d_out*h_out*w_out
                            + d*h_out*w_out
                            + h*w_out
                            + w
                        );
                        output[out_adr] = bias[oc];
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kd = 0; kd < kernel_size; kd++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int in_adr = (
                                            b*in_channels*in_depth*in_height*in_width
                                            + ic*in_depth*in_height*in_width
                                            + (d*stride + kd)*in_height*in_width
                                            + (h*stride + kh)*in_width
                                            + (w*stride + kw)
                                        );
                                        int weight_adr = (
                                            oc*in_channels*kernel_size*kernel_size*kernel_size
                                            + ic*kernel_size*kernel_size*kernel_size
                                            + kd*kernel_size*kernel_size
                                            + kh*kernel_size
                                            + kw
                                        );
                                        output[out_adr] += input[in_adr]*weights[weight_adr];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void _lstm(float* input, float* state_h, float* state_c, float* weights_input,
        float* weights_state, float* bias_input, float*bias_state,
        float *buffer, int batch_size, int input_size, int hidden_size) {
    _linear(input, weights_input, bias_input, buffer, batch_size, input_size, 4*hidden_size);
    _linear_accumulate(state_h, weights_state, bias_state, buffer, batch_size, hidden_size, 4*hidden_size);

    // Activation functions
    for (int b=0; b<batch_size; b++) {
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<2*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = _sigmoid(buffer[buf_adr]);
        }
        for (int i=2*hidden_size; i<3*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = tanh(buffer[buf_adr]);
        }
        for (int i=3*hidden_size; i<4*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = _sigmoid(buffer[buf_adr]);
        }
    }

    // Gates
    for (int b=0; b<batch_size; b++) {
        int hidden_offset = b*hidden_size;
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<hidden_size; i++) {
            state_c[hidden_offset + i] = (
                buffer[b_offset + hidden_size + i] * state_c[hidden_offset + i]
                + buffer[b_offset + i] * buffer[b_offset + 2*hidden_size + i]
            );
            state_h[hidden_offset + i] = (
                buffer[b_offset + 3*hidden_size + i] * tanh(state_c[hidden_offset + i])
            );
        }
    }
}

void _embedding(int* input, float* weights, float* output, int batch_size, int num_embeddings, int embedding_dim) {
    for (int b = 0; b < batch_size; b++) {
        memcpy(output + b*embedding_dim, weights + input[b]*embedding_dim, embedding_dim*sizeof(float));
    }
}

void _one_hot(int* input, int* output, int batch_size, int input_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < input_size; i++) {
            int in_adr = b*input_size + i;
            int out_adr = (
                b*input_size*num_classes
                + i*num_classes
                + input[in_adr]
            );
            output[out_adr] = 1.0f;
        }
    }
}

void _cat_dim1(float* x, float* y, float* output, int batch_size, int x_size, int y_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < x_size; i++) {
            int x_adr = b*x_size + i;
            int out_adr = b*(x_size + y_size) + i;
            output[out_adr] = x[x_adr];
        }
        for (int i = 0; i < y_size; i++) {
            int y_adr = b*y_size + i;
            int out_adr = b*(x_size + y_size) + x_size + i;
            output[out_adr] = y[y_adr];
        }
    }
}

void _argmax_multidiscrete(float* input, int* output, int batch_size, int logit_sizes[], int num_actions) {
    int in_adr = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b*num_actions + a;
            float max_logit = input[in_adr];
            output[out_adr] = 0;
            int num_action_types = logit_sizes[a];
            for (int i = 1; i < num_action_types; i++) {
                float out = input[in_adr + i];
                if (out > max_logit) {
                    max_logit = out;
                    output[out_adr] = i;
                }
            }
            in_adr += num_action_types;
        }
    }
}

void _softmax_multidiscrete(float* input, int* output, int batch_size, int logit_sizes[], int num_actions) {
    int in_adr = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b*num_actions + a;
            float logit_exp_sum = 0;
            int num_action_types = logit_sizes[a];
            for (int i = 0; i < num_action_types; i++) {
                logit_exp_sum += expf(input[in_adr + i]);
            }
            float prob = rand() / (float)RAND_MAX;
            float logit_prob = 0;
            for (int i = 0; i < num_action_types; i++) {
                logit_prob += expf(input[in_adr + i]) / logit_exp_sum;
                if (prob < logit_prob) {
                    output[out_adr] = i;
                    break;
                }
            }
            in_adr += num_action_types;
        }
    }
}

// User API. Provided to help organize layers
typedef struct Linear Linear;
struct Linear {
    float* output;
    float* weights;
    float* bias;
    int batch_size;
    int input_dim;
    int output_dim;
};

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim) {
    size_t buffer_size = batch_size*output_dim*sizeof(float);
    Linear* layer = calloc(1, sizeof(Linear) + buffer_size);
    *layer = (Linear){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, output_dim*input_dim),
        .bias = get_weights(weights, output_dim),
        .batch_size = batch_size,
        .input_dim = input_dim,
        .output_dim = output_dim,
    };
    return layer;
}

void linear(Linear* layer, float* input) {
    _linear(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->input_dim, layer->output_dim);
}

void linear_accumulate(Linear* layer, float* input) {
    _linear_accumulate(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->input_dim, layer->output_dim);
}

typedef struct ReLU ReLU;
struct ReLU {
    float* output;
    int batch_size;
    int input_dim;
};

ReLU* make_relu(int batch_size, int input_dim) {
    size_t buffer_size = batch_size*input_dim*sizeof(float);
    ReLU* layer = calloc(1, sizeof(ReLU) + buffer_size);
    *layer = (ReLU){
        .output = (float*)(layer + 1),
        .batch_size = batch_size,
        .input_dim = input_dim,
    };
    return layer;
}

void relu(ReLU* layer, float* input) {
    _relu(input, layer->output, layer->batch_size*layer->input_dim);
}

typedef struct Conv2D Conv2D;
struct Conv2D {
    float* output;
    float* weights;
    float* bias;
    int batch_size;
    int in_width;
    int in_height;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
};

Conv2D* make_conv2d(Weights* weights, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    size_t buffer_size = batch_size*out_channels*in_height*in_width*sizeof(float);
    int num_weights = out_channels*in_channels*kernel_size*kernel_size;
    Conv2D* layer = calloc(1, sizeof(Conv2D) + buffer_size);
    *layer = (Conv2D){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, num_weights),
        .bias = get_weights(weights, out_channels),
        .batch_size = batch_size,
        .in_width = in_width,
        .in_height = in_height,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .kernel_size = kernel_size,
        .stride = stride,
    };
    return layer;
}

void conv2d(Conv2D* layer, float* input) {
    _conv2d(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->in_width, layer->in_height,
        layer->in_channels, layer->out_channels, layer->kernel_size, layer->stride);
}

typedef struct Conv3D Conv3D;
struct Conv3D {
    float* output;
    float* weights;
    float* bias;
    int batch_size;
    int in_width;
    int in_height;
    int in_depth;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
};

Conv3D* make_conv3d(Weights* weights, int batch_size, int in_width, int in_height, int in_depth,
        int in_channels, int out_channels, int kernel_size, int stride) {
    
    size_t buffer_size = batch_size*out_channels*in_depth*in_height*in_width*sizeof(float);
    int num_weights = out_channels*in_channels*kernel_size*kernel_size*kernel_size;
    Conv3D* layer = calloc(1, sizeof(Conv3D) + buffer_size);
    *layer = (Conv3D){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, num_weights),
        .bias = get_weights(weights, out_channels),
        .batch_size = batch_size,
        .in_width = in_width,
        .in_height = in_height,
        .in_depth = in_depth,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .kernel_size = kernel_size,
        .stride = stride,
    };
    return layer;
}

void conv3d(Conv3D* layer, float* input) {
    _conv3d(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->in_width, layer->in_height, layer->in_depth,
        layer->in_channels, layer->out_channels, layer->kernel_size, layer->stride);
}

typedef struct LSTM LSTM;
struct LSTM {
    float* state_h;
    float* state_c;
    float* weights_input;
    float* weights_state;
    float* bias_input;
    float*bias_state;
    float *buffer;
    int batch_size;
    int input_size;
    int hidden_size;
};

LSTM* make_lstm(Weights* weights, int batch_size, int input_size, int hidden_size) {
    int state_size = batch_size*hidden_size;
    LSTM* layer = calloc(1, sizeof(LSTM) + 6*state_size*sizeof(float));
    float* buffer = (float*)(layer + 1);
    *layer = (LSTM){
        .state_h = buffer,
        .state_c = buffer + state_size,
        .weights_input = get_weights(weights, 4*hidden_size*input_size),
        .weights_state = get_weights(weights, 4*hidden_size*hidden_size),
        .bias_input = get_weights(weights, 4*hidden_size),
        .bias_state = get_weights(weights, 4*hidden_size),
        .buffer = buffer + 2*state_size,
        .batch_size = batch_size,
        .input_size = input_size,
        .hidden_size = hidden_size,

    };
    return layer;
}

void lstm(LSTM* layer, float* input) {
    _lstm(input, layer->state_h, layer->state_c, layer->weights_input,
        layer->weights_state, layer->bias_input, layer->bias_state,
        layer->buffer, layer->batch_size, layer->input_size, layer->hidden_size);
}

typedef struct Embedding Embedding;
struct Embedding {
    float* output;
    float* weights;
    int batch_size;
    int num_embeddings;
    int embedding_dim;
};

Embedding* make_embedding(Weights* weights, int batch_size, int num_embeddings, int embedding_dim) {
    size_t output_size = batch_size*embedding_dim*sizeof(float);
    Embedding* layer = (Embedding*)calloc(1, sizeof(Embedding) + batch_size + output_size);
    *layer = (Embedding){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, num_embeddings*embedding_dim),
        .batch_size = batch_size,
        .num_embeddings = num_embeddings,
        .embedding_dim = embedding_dim,
    };
    return layer;
}

void embedding(Embedding* layer, int* input) {
    _embedding(input, layer->weights, layer->output, layer->batch_size, layer->num_embeddings, layer->embedding_dim);
}

typedef struct OneHot OneHot;
struct OneHot {
    int* output;
    int batch_size;
    int input_size;
    int num_classes;
};

OneHot* make_one_hot(int batch_size, int input_size, int num_classes) {
    size_t buffer_size = batch_size*input_size*num_classes*sizeof(int);
    OneHot* layer = calloc(1, sizeof(OneHot) + buffer_size);
    *layer = (OneHot){
        .output = (int*)(layer + 1),
        .batch_size = batch_size,
        .input_size = input_size,
        .num_classes = num_classes,
    };
    return layer;
}

void one_hot(OneHot* layer, int* input) {
    _one_hot(input, layer->output, layer->batch_size, layer->input_size, layer->num_classes);
}

typedef struct CatDim1 CatDim1;
struct CatDim1 {
    float* output;
    int batch_size;
    int x_size;
    int y_size;
};

CatDim1* make_cat_dim1(int batch_size, int x_size, int y_size) {
    size_t buffer_size = batch_size*(x_size + y_size)*sizeof(float);
    CatDim1* layer = calloc(1, sizeof(CatDim1) + buffer_size);
    *layer = (CatDim1){
        .output = (float*)(layer + 1),
        .batch_size = batch_size,
        .x_size = x_size,
        .y_size = y_size,
    };
    return layer;
}

void cat_dim1(CatDim1* layer, float* x, float* y) {
    _cat_dim1(x, y, layer->output, layer->batch_size, layer->x_size, layer->y_size);
}

typedef struct Multidiscrete Multidiscrete;
struct Multidiscrete {
    int batch_size;
    int logit_sizes[32];
    int num_actions;
};

Multidiscrete* make_multidiscrete(int batch_size, int logit_sizes[], int num_actions) {
    Multidiscrete* layer = calloc(1, sizeof(Multidiscrete));
    layer->batch_size = batch_size;
    layer->num_actions = num_actions;
    memcpy(layer->logit_sizes, logit_sizes, num_actions*sizeof(int));
    return layer;
}

void argmax_multidiscrete(Multidiscrete* layer, float* input, int* output) {
    _argmax_multidiscrete(input, output, layer->batch_size, layer->logit_sizes, layer->num_actions);
}

void softmax_multidiscrete(Multidiscrete* layer, float* input, int* output) {
    _softmax_multidiscrete(input, output, layer->batch_size, layer->logit_sizes, layer->num_actions);
}

// Default models

typedef struct Default Default;
struct Default {
    int num_agents;
    float* obs;
    Linear* encoder;
    ReLU* relu1;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

Default* make_default(Weights* weights, int num_agents, int input_dim, int hidden_dim, int action_dim) {
    Default* net = calloc(1, sizeof(Default));
    net->num_agents = num_agents;
    net->obs = (float*)calloc(num_agents*input_dim, sizeof(float));
    net->encoder = make_linear(weights, num_agents, input_dim, hidden_dim);
    net->relu1 = make_relu(num_agents, hidden_dim);
    net->actor = make_linear(weights, num_agents, hidden_dim, action_dim);
    net->value_fn = make_linear(weights, num_agents, hidden_dim, 1);
    int logit_sizes[1] = {action_dim};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void free_default(Default* net) {
    free(net->obs);
    free(net->encoder);
    free(net->relu1);
    free(net->actor);
    free(net->value_fn);
    free(net->multidiscrete);
    free(net);
}

void forward_default(Default* net, float* observations, int* actions) {
    linear(net->encoder, observations);
    relu(net->relu1, net->encoder->output);
    linear(net->actor, net->relu1->output);
    linear(net->value_fn, net->relu1->output);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

typedef struct LinearLSTM LinearLSTM;
struct LinearLSTM {
    int num_agents;
    float* obs;
    Linear* encoder;
    ReLU* relu1;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

LinearLSTM* make_linearlstm(Weights* weights, int num_agents, int input_dim, int action_dim) {
    LinearLSTM* net = calloc(1, sizeof(LinearLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents*input_dim, sizeof(float));
    net->encoder = make_linear(weights, num_agents, input_dim, 128);
    net->relu1 = make_relu(num_agents, 128);
    net->actor = make_linear(weights, num_agents, 128, action_dim);
    net->value_fn = make_linear(weights, num_agents, 128, 1);
    net->lstm = make_lstm(weights, num_agents, 128, 128);
    int logit_sizes[1] = {action_dim};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void free_linearlstm(LinearLSTM* net) {
    free(net->obs);
    free(net->encoder);
    free(net->relu1);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward_linearlstm(LinearLSTM* net, float* observations, int* actions) {
    linear(net->encoder, observations);
    relu(net->relu1, net->encoder->output);
    lstm(net->lstm, net->relu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}

typedef struct ConvLSTM ConvLSTM; struct ConvLSTM {
    int num_agents;
    float* obs;
    Conv2D* conv1;
    ReLU* relu1;
    Conv2D* conv2;
    ReLU* relu2;
    Linear* linear;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

ConvLSTM* make_convlstm(Weights* weights, int num_agents, int input_dim,
        int input_channels, int cnn_channels, int hidden_dim, int action_dim) {
    ConvLSTM* net = calloc(1, sizeof(ConvLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents*input_dim*input_dim*input_channels, sizeof(float));
    net->conv1 = make_conv2d(weights, num_agents, input_dim,
        input_dim, input_channels, cnn_channels, 5, 3);
    net->relu1 = make_relu(num_agents, hidden_dim*3*3);
    net->conv2 = make_conv2d(weights, num_agents, 3, 3, cnn_channels, cnn_channels, 3, 1);
    net->relu2 = make_relu(num_agents, hidden_dim);
    net->linear = make_linear(weights, num_agents, cnn_channels, hidden_dim);
    net->actor = make_linear(weights, num_agents, hidden_dim, action_dim);
    net->value_fn = make_linear(weights, num_agents, hidden_dim, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_dim, hidden_dim);
    int logit_sizes[1] = {action_dim};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

void free_convlstm(ConvLSTM* net) {
    free(net->obs);
    free(net->conv1);
    free(net->relu1);
    free(net->conv2);
    free(net->relu2);
    free(net->linear);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->multidiscrete);
    free(net);
}

void forward_convlstm(ConvLSTM* net, float* observations, int* actions) {
    conv2d(net->conv1, observations);
    relu(net->relu1, net->conv1->output);
    conv2d(net->conv2, net->relu1->output);
    relu(net->relu2, net->conv2->output);
    linear(net->linear, net->relu2->output);
    lstm(net->lstm, net->linear->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}
