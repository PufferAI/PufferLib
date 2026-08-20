#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>

typedef struct {
    void* data;
    size_t capacity;
    size_t used;
} Arena;

Arena* make_allocator(size_t total_size) {
    void* buffer = (void*)calloc(1, total_size + sizeof(Arena));
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

// File format: flat fp32 tensors.
typedef struct Weights {
    float* data;
    int size;
    int idx;
} Weights;

Weights* load_weights(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    size_t num_weights = (size_t)file_size / sizeof(float);
    // +7 ensures get_weights_aligned never reads past the buffer: the native
    // backend uses 16-byte alignment with bf16 params (2 bytes), so each tensor
    // starts at an 8-float boundary. After the last tensor, up to 7 extra floats
    // may be addressed before the next 8-aligned boundary.
    Weights* weights = (Weights*)calloc(1, sizeof(Weights) + (num_weights + 7)*sizeof(float));
    weights->data = (float*)(weights + 1);
    size_t read_size = fread(weights->data, sizeof(float), num_weights, file);
    fclose(file);
    if (read_size != num_weights) {
        perror("Error reading file");
    }
    weights->size = (int)num_weights + 7;
    weights->idx = 0;
    return weights;
}

float* get_weights(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    assert(weights->idx <= weights->size);
    return data;
}

// Advances index to next 8-float (16-byte) boundary after reading, matching
// the native backend's Allocator which aligns to 16 bytes with bf16 params.
float* get_weights_aligned(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    weights->idx = (weights->idx + 7) & ~7;
    assert(weights->idx <= weights->size);
    return data;
}

// PufferNet implementation of PyTorch functions
// These are tested against the PyTorch implementation
void _relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

float _sigmoid(float x) {
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

double _randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

void _gaussian_sample(float* input, float* log_std, float* output, int batch_size, int num_actions) {
    for (int b = 0; b < batch_size; b++) {
        // +1 skips the value head fused into the decoder output
        int in_adr = b * (num_actions + 1);
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b * num_actions + a;
            float mean = input[in_adr + a];
            float std = expf(log_std[a]);
            output[out_adr] = (float)_randn(mean, std);
        }
    }
}

// hard=1: argmax; hard=0: softmax sample. +1 on in_adr skips fused value head.
// mask is optional (batch * atn_sum); NULL means every logit is legal.
void _multidiscrete(float* input, float* output, int batch_size, int logit_sizes[],
        int num_actions, int hard, const unsigned char* mask) {
    int atn_sum = 0;
    for (int a = 0; a < num_actions; a++) {
        atn_sum += logit_sizes[a];
    }
    for (int b = 0; b < batch_size; b++) {
        int in_adr = b * (atn_sum + 1);
        int mask_adr = b * atn_sum;
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b * num_actions + a;
            int n = logit_sizes[a];
            const unsigned char* head_mask = mask ? mask + mask_adr : NULL;
            output[out_adr] = 0.0f;
            int first = -1;
            for (int i = 0; i < n; i++) {
                if (!head_mask || head_mask[i]) {
                    first = i;
                    break;
                }
            }
            if (first < 0) {
                in_adr += n;
                mask_adr += n;
                continue;
            }
            if (hard) {
                float max_logit = input[in_adr + first];
                output[out_adr] = (float)first;
                for (int i = first + 1; i < n; i++) {
                    if (head_mask && !head_mask[i]) {
                        continue;
                    }
                    float out = input[in_adr + i];
                    if (out > max_logit) {
                        max_logit = out;
                        output[out_adr] = (float)i;
                    }
                }
            } else {
                float max_logit = input[in_adr + first];
                for (int i = first + 1; i < n; i++) {
                    if ((!head_mask || head_mask[i]) &&
                            input[in_adr + i] > max_logit) {
                        max_logit = input[in_adr + i];
                    }
                }
                float logit_exp_sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    if (!head_mask || head_mask[i]) {
                        logit_exp_sum += expf(input[in_adr + i] - max_logit);
                    }
                }
                float prob = rand() / (float)RAND_MAX;
                float logit_prob = 0.0f;
                output[out_adr] = (float)first;
                for (int i = 0; i < n; i++) {
                    if (head_mask && !head_mask[i]) {
                        continue;
                    }
                    logit_prob += expf(input[in_adr + i] - max_logit) / logit_exp_sum;
                    if (prob < logit_prob) {
                        output[out_adr] = (float)i;
                        break;
                    }
                }
            }
            in_adr += n;
            mask_adr += n;
        }
    }
}

void _max_dim1(float* input, float* output, int batch_size, int seq_len, int feature_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            float max_val = input[b*seq_len*feature_dim + f];
            for (int s = 1; s < seq_len; s++) {
                float val = input[b*seq_len*feature_dim + s*feature_dim + f];
                if (val > max_val) {
                    max_val = val;
                }
            }
            output[b*feature_dim + f] = max_val;
        }
    }
}

// User API. Provided to help organize layers
typedef struct Linear {
    float* output;
    float* weights;
    int batch_size;
    int input_dim;
    int output_dim;
} Linear;

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim) {
    size_t buffer_size = batch_size*output_dim*sizeof(float);
    Linear* layer = (Linear*)calloc(1, sizeof(Linear) + buffer_size);
    *layer = (Linear){
        .output = (float*)(layer + 1),
        .weights = get_weights_aligned(weights, output_dim*input_dim),
        .batch_size = batch_size,
        .input_dim = input_dim,
        .output_dim = output_dim,
    };
    return layer;
}

void linear(Linear* layer, float* input) {
    for (int b = 0; b < layer->batch_size; b++) {
        for (int o = 0; o < layer->output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < layer->input_dim; i++)
                sum += input[b*layer->input_dim + i] * layer->weights[o*layer->input_dim + i];
            layer->output[b*layer->output_dim + o] = sum;
        }
    }
}

typedef struct ReLU {
    float* output;
    int batch_size;
    int input_dim;
} ReLU;

ReLU* make_relu(int batch_size, int input_dim) {
    size_t buffer_size = batch_size*input_dim*sizeof(float);
    ReLU* layer = (ReLU*)calloc(1, sizeof(ReLU) + buffer_size);
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

typedef struct MaxDim1 {
    float* output;
    int batch_size;
    int seq_len;
    int feature_dim;
} MaxDim1;

MaxDim1* make_max_dim1(int batch_size, int seq_len, int feature_dim) {
    size_t buffer_size = batch_size*feature_dim*sizeof(float);
    MaxDim1* layer = (MaxDim1*)calloc(1, sizeof(MaxDim1) + buffer_size);
    *layer = (MaxDim1){
        .output = (float*)(layer + 1),
        .batch_size = batch_size,
        .seq_len = seq_len,
        .feature_dim = feature_dim,
    };
    return layer;
}

void max_dim1(MaxDim1* layer, float* input) {
    _max_dim1(input, layer->output, layer->batch_size, layer->seq_len, layer->feature_dim);
}

typedef struct Conv2D {
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
} Conv2D;

Conv2D* make_conv2d(Weights* weights, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    size_t buffer_size = batch_size*out_channels*in_height*in_width*sizeof(float);
    int num_weights = out_channels*in_channels*kernel_size*kernel_size;
    Conv2D* layer = (Conv2D*)calloc(1, sizeof(Conv2D) + buffer_size);
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

typedef struct Embedding {
    float* output;
    float* weights;
    int batch_size;
    int num_embeddings;
    int embedding_dim;
} Embedding;

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

typedef struct OneHot {
    int* output;
    int batch_size;
    int input_size;
    int num_classes;
} OneHot;

OneHot* make_one_hot(int batch_size, int input_size, int num_classes) {
    size_t buffer_size = batch_size*input_size*num_classes*sizeof(int);
    OneHot* layer = (OneHot*)calloc(1, sizeof(OneHot) + buffer_size);
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

typedef struct Multidiscrete {
    int batch_size;
    int logit_sizes[32];
    int num_actions;
} Multidiscrete;

Multidiscrete* make_multidiscrete(int batch_size, int logit_sizes[], int num_actions) {
    Multidiscrete* layer = (Multidiscrete*)calloc(1, sizeof(Multidiscrete));
    layer->batch_size = batch_size;
    layer->num_actions = num_actions;
    memcpy(layer->logit_sizes, logit_sizes, num_actions*sizeof(int));
    return layer;
}

void multidiscrete(Multidiscrete* layer, float* input, float* output, int hard,
        const unsigned char* mask) {
    _multidiscrete(input, output, layer->batch_size, layer->logit_sizes,
        layer->num_actions, hard, mask);
}

// MinGRU: inference-only single-step recurrent layer.
// Matches the fused gate + highway connection in models.cu mingru_gate kernel.
// Each layer has a bias-free projection (hidden -> 3*hidden).
// State layout: (num_layers, batch_size, hidden_size).
typedef struct MinGRU {
    float* state;    // (num_layers, batch_size, hidden_size) - persists across steps
    float* output;   // (batch_size, hidden_size)
    Linear** proj;   // [num_layers], each projects hidden -> 3*hidden
    int batch_size;
    int hidden_size;
    int num_layers;
} MinGRU;

MinGRU* make_mingru(Weights* weights, int batch_size, int hidden_size, int num_layers) {
    MinGRU* layer = (MinGRU*)calloc(1, sizeof(MinGRU));
    layer->state = (float*)calloc(num_layers * batch_size * hidden_size, sizeof(float));
    layer->output = (float*)calloc(batch_size * hidden_size, sizeof(float));
    layer->proj = (Linear**)calloc(num_layers, sizeof(Linear*));
    layer->batch_size  = batch_size;
    layer->hidden_size = hidden_size;
    layer->num_layers  = num_layers;
    for (int l = 0; l < num_layers; l++) {
        layer->proj[l] = make_linear(weights, batch_size, hidden_size, 3 * hidden_size);
    }
    return layer;
}

void mingru(MinGRU* layer, float* input) {
    int B = layer->batch_size;
    int H = layer->hidden_size;
    float* x = input;
    for (int l = 0; l < layer->num_layers; l++) {
        float* state_l = layer->state + l * B * H;
        linear(layer->proj[l], x);
        float* combined = layer->proj[l]->output;
        for (int b = 0; b < B; b++) {
            float* cb = combined + b * 3 * H;
            float* sb = state_l + b * H;
            float* xb = x + b * H;
            float* ob = layer->output + b * H;
            for (int h = 0; h < H; h++) {
                float hidden     = cb[h];
                float gate       = cb[H + h];
                float hw         = cb[2*H + h];
                float s          = sb[h];
                float gate_s     = _sigmoid(gate);
                float h_tilde    = (hidden >= 0.0f) ? hidden + 0.5f : _sigmoid(hidden);
                float mingru_out = s + gate_s * (h_tilde - s);
                float hw_s       = _sigmoid(hw);
                ob[h] = hw_s * mingru_out + (1.0f - hw_s) * xb[h];
                sb[h] = mingru_out;
            }
        }
        x = layer->output;
    }
}

void free_mingru(MinGRU* layer) {
    for (int l = 0; l < layer->num_layers; l++) free(layer->proj[l]);
    free(layer->state);
    free(layer->output);
    free(layer->proj);
    free(layer);
}

// Trainer zero_term_state: clear carry before encoding a post-terminal obs.
void mingru_zero_term(MinGRU* layer, const float* terminals) {
    int B = layer->batch_size;
    int H = layer->hidden_size;
    for (int b = 0; b < B; b++) {
        if (terminals[b] <= 0.5f) {
            continue;
        }
        for (int l = 0; l < layer->num_layers; l++) {
            memset(layer->state + (l * B + b) * H, 0, H * sizeof(float));
        }
    }
}

// PufferNet: default policy matching the native backend Arch in algo.cu.
// Architecture: Linear encoder -> N x MinGRU -> Linear decoder (fused value).
// Weight file order (matches weights_create reg_params call order):
//   encoder weight (hidden_dim x input_dim)
//   decoder weight ((atn_sum+1) x hidden_dim, last output is value)
//   decoder logstd (1 x num_actions) IF continuous
//   mingru weights[0..num_layers-1] (3*hidden_dim x hidden_dim each)
typedef struct PufferNet {
    int num_agents;
    float* obs;
    Linear* encoder;
    MinGRU* mingru;
    Linear* decoder;   // output_dim = atn_sum+1; last element is value
    float* log_std;
    int is_continuous;
    int num_actions;
    Multidiscrete* multidiscrete;
} PufferNet;

PufferNet* make_puffernet(Weights* weights, int num_agents, int input_dim,
        int hidden_dim, int num_layers, int logit_sizes[], int num_actions) {
    PufferNet* net = (PufferNet*)calloc(1, sizeof(PufferNet));
    net->num_agents = num_agents;
    net->obs = (float*)calloc(num_agents * input_dim, sizeof(float));
    int atn_sum = 0;
    int is_continuous = 1;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
        if (logit_sizes[i] != 1) is_continuous = 0;
    }
    net->is_continuous = is_continuous;
    net->num_actions = num_actions;

    net->encoder = make_linear(weights, num_agents, input_dim, hidden_dim);
    net->decoder = make_linear(weights, num_agents, hidden_dim, atn_sum + 1);
    if (net->is_continuous) {
        net->log_std = get_weights_aligned(weights, num_actions);
    }
    net->mingru  = make_mingru(weights, num_agents, hidden_dim, num_layers);
    if (!net->is_continuous) {
        net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, num_actions);
    }
    return net;
}

void _gaussian_mean(float* input, float* output, int batch_size, int num_actions) {
    for (int b = 0; b < batch_size; b++) {
        // +1 skips the value head fused into the decoder output
        int in_adr = b * (num_actions + 1);
        for (int a = 0; a < num_actions; a++)
            output[b * num_actions + a] = input[in_adr + a];
    }
}

void forward_puffernet(PufferNet* net, float* observations, float* actions,
        const unsigned char* mask, const float* terminals) {
    mingru_zero_term(net->mingru, terminals);
    linear(net->encoder, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    if (net->is_continuous) {
        _gaussian_mean(net->decoder->output, actions, net->num_agents, net->num_actions);
    } else {
        multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, mask);
    }
}

void free_puffernet(PufferNet* net) {
    free(net->obs);
    free(net->encoder);
    free(net->decoder);
    free_mingru(net->mingru);
    if (net->multidiscrete) {
        free(net->multidiscrete);
    }
    free(net);
}

#ifdef PUFFERCPU_EVAL_MAIN

#ifndef ENV_HEADER
#error "ENV_HEADER required for PUFFERCPU_EVAL_MAIN"
#endif

#include ENV_HEADER

#if !defined(PUF_NMMO3_NET) && !defined(PUF_ASTEROIDS_NET) && !defined(PUF_MINIMAL_NET)
static int puf_align8(int n) {
    return (n + 7) & ~7;
}

static int puffernet_weight_count(int input_dim, int hidden, int layers,
        const int act_sizes[], int num_actions) {
    int atn_sum = 0;
    int continuous = 1;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += act_sizes[i];
        if (act_sizes[i] != 1) {
            continuous = 0;
        }
    }
    int n = 0;
    n = puf_align8(n + hidden * input_dim);
    n = puf_align8(n + (atn_sum + 1) * hidden);
    if (continuous) {
        n = puf_align8(n + num_actions);
    }
    for (int l = 0; l < layers; l++) {
        n = puf_align8(n + 3 * hidden * hidden);
    }
    return n;
}
#endif

static void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        size_t pn = strlen(path);
        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && pn >= 4 &&
                strcmp(path + pn - 4, ".bin") == 0 &&
                st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

static const char* puf_model_path(const char* env_name, const char* cli_path,
        int latest, char* out, size_t out_size) {
    if (cli_path) {
        return cli_path;
    }
    if (latest) {
        char root[2048];
        snprintf(root, sizeof(root), "checkpoints/%s", env_name);
        out[0] = 0;
        time_t best_time = 0;
        puf_find_latest_checkpoint(root, out, out_size, &best_time);
        return out[0] ? out : NULL;
    }
    snprintf(out, out_size, "resources/%s/%s_weights.bin", env_name, env_name);
    return out;
}

int main(int argc, char** argv) {
    const char* env_name = PUFFER_ENV_NAME;
    int argi = 1;
    if (argc >= 2 && argv[1][0] && argv[1][0] != '-' &&
            strchr(argv[1], '=') == NULL && strchr(argv[1], '/') == NULL &&
            strstr(argv[1], ".bin") == NULL && strcmp(argv[1], "latest") != 0) {
        env_name = argv[1];
        argi = 2;
    }
    Ini ini = {0};
    int headless = 0;
    const char* cli_path = NULL;
    int cli_latest = 0;

    char* ini_argv[argc > 0 ? (size_t)argc : 1];
    int ini_argc = 0;
    for (int i = argi; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
            continue;
        }
        if (strcmp(argv[i], "latest") == 0) {
            cli_latest = 1;
            continue;
        }
        if (strchr(argv[i], '=') == NULL &&
                (strstr(argv[i], ".bin") != NULL || strchr(argv[i], '/'))) {
            cli_path = argv[i];
            continue;
        }
        ini_argv[ini_argc++] = argv[i];
    }
    puf_ini_load_env(&ini, env_name, ini_argc, ini_argv);
    int eval_episodes = (int)puf_ini_get(&ini, "base", "eval_episodes");

    int act_sizes[] = ACT_SIZES;
    int num_actions = (int)(sizeof(act_sizes) / sizeof(act_sizes[0]));
    char path_buf[1024];
    const char* path = puf_model_path(env_name, cli_path, cli_latest,
        path_buf, sizeof(path_buf));
    Weights* weights = path ? load_weights(path) : NULL;
    int hidden_size = (int)puf_ini_get(&ini, "policy", "hidden_size");
    int num_layers = (int)puf_ini_get(&ini, "policy", "num_layers");
    int file_floats = weights ? (weights->size - 7) : 0;
#ifdef PUF_NMMO3_NET
    int need = nmmo3_weight_count(hidden_size, num_layers);
#elif defined(PUF_ASTEROIDS_NET)
    int need = asteroids_weight_count(hidden_size, num_layers);
#elif defined(PUF_MINIMAL_NET)
    int need = minimal_weight_count(hidden_size, num_layers);
#else
    int need = puffernet_weight_count(OBS_SIZE, hidden_size, num_layers,
        act_sizes, num_actions);
#endif
    assert(!weights || (need - file_floats <= 7 && file_floats <= need));
    int have_net = weights != NULL;
    if (headless) {
        printf("CPU_META env=%s path=%s file_floats=%d need=%d untrained=%d hidden=%d layers=%d\n",
            env_name, path ? path : "-", file_floats, need, !have_net,
            hidden_size, num_layers);
        fflush(stdout);
    }

    Dict* env_sec = puf_ini_section(&ini, "env", 0);
    int frameskip = 1;
    DictItem* fs_item = dict_find(env_sec, "frameskip");
    if (fs_item && fs_item->value > 1.0) {
        frameskip = (int)fs_item->value;
        dict_set(env_sec, "frameskip", 1);
    }
#ifdef PUF_EVAL_SHOULD_FORWARD
    frameskip = 1;
#endif

    Env env = {0};
    env.rng = 0;
    puf_init(&env, env_sec);

    // Heap, not VLAs: multiagent envs (snake=256, obs=968) overflow the
    // 512KB WASM stack (obs_f alone is ~1MB).
    size_t n_obs = (size_t)env.num_agents * (size_t)OBS_SIZE;
    size_t n_atn = (size_t)env.num_agents * (size_t)NUM_ATNS;
    size_t n_agt = (size_t)env.num_agents;
    obs_t* observations = (obs_t*)calloc(n_obs, sizeof(obs_t));
#if !defined(PUF_NMMO3_NET) && !defined(PUF_ASTEROIDS_NET) && !defined(PUF_MINIMAL_NET)
    float* obs_f = (float*)calloc(n_obs, sizeof(float));
#endif
    float* actions = (float*)calloc(n_atn, sizeof(float));
    float* rewards = (float*)calloc(n_agt, sizeof(float));
    float* terminals = (float*)calloc(n_agt, sizeof(float));
    int act_n = 0;
    for (int i = 0; i < num_actions; i++) {
        act_n += act_sizes[i];
    }
    unsigned char* masks = (unsigned char*)calloc(
        (size_t)env.num_agents * (size_t)act_n, 1);
    memset(masks, 1, (size_t)env.num_agents * (size_t)act_n);
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = masks + i * act_n;
        env.agents[i].policy = 0;
    }
    puf_reset(&env);

#ifdef PUF_NMMO3_NET
    MMONet* net = NULL;
    if (have_net) {
        net = init_mmonet_arch(weights, env.num_agents,
            hidden_size, num_layers);
    }
#elif defined(PUF_ASTEROIDS_NET)
    AsteroidsNet* net = NULL;
    if (have_net) {
        net = init_asteroids_net(weights, env.num_agents,
            hidden_size, num_layers);
    }
#elif defined(PUF_MINIMAL_NET)
    MinimalNet* net = NULL;
    if (have_net) {
        net = init_minimal_net(weights, env.num_agents,
            hidden_size, num_layers);
    }
#else
    PufferNet* net = NULL;
    if (have_net) {
        net = make_puffernet(weights, env.num_agents, OBS_SIZE,
            hidden_size, num_layers, act_sizes, num_actions);
    }
#endif

    SetConfigFlags(FLAG_MSAA_4X_HINT);

#ifndef PUF_STEPS_PER_SEC
#define PUF_STEPS_PER_SEC 60
#endif
    int step_period = 1;
    int step_credit = 0;
    if ((int)PUF_STEPS_PER_SEC < 60) {
        step_period = 60 / (int)PUF_STEPS_PER_SEC;
        assert(step_period >= 1);
    }
    int step_hold = 0;
    int hold = 0;
    int steps = 0;
#ifndef PLATFORM_WEB
    if (!headless) {
        SetTargetFPS(60);
    }
#endif
    if (!headless) {
        puf_render(&env);
    }
    while (headless
            ? (env.log.n < (float)eval_episodes)
            : !IsWindowReady() || !WindowShouldClose()) {
        int do_step = headless || (step_hold == 0);
        int ticks = 1;
        if (!headless && (int)PUF_STEPS_PER_SEC > 60) {
            step_credit += (int)PUF_STEPS_PER_SEC;
            ticks = 0;
            while (step_credit >= 60) {
                step_credit -= 60;
                ticks++;
            }
        }
        for (int t = 0; t < ticks; t++) {
#ifdef PUF_EVAL_SHOULD_FORWARD
            int eval_fwd = env.tick_frames_left <= 0;
#else
            int eval_fwd = 1;
#endif
            int eval_human = !headless && IsWindowReady() &&
                IsKeyDown(KEY_LEFT_SHIFT);
            if (have_net && do_step && eval_fwd && hold == 0 && !eval_human) {
#ifdef PUF_NMMO3_NET
                forward(net, observations, terminals, actions);
#elif defined(PUF_ASTEROIDS_NET)
                forward_asteroids(net, (float*)observations, terminals, actions);
#elif defined(PUF_MINIMAL_NET)
                forward_minimal(net, (float*)observations, terminals, actions);
#else
                float* fwd = (float*)observations;
                if (sizeof(obs_t) != sizeof(float)) {
                    for (int i = 0; i < (int)n_obs; i++) {
                        obs_f[i] = (float)observations[i];
                    }
                    fwd = obs_f;
                }
                forward_puffernet(net, fwd, actions, masks, terminals);
#endif
            }
            if (do_step) {
                puf_step(&env);
                if (headless) {
                    steps++;
                }
            }
            if (do_step && eval_fwd) {
                int reset_hold = 0;
                for (int a = 0; a < env.num_agents; a++) {
                    if (rewards[a] != 0.0f || terminals[a] > 0.5f) {
                        reset_hold = 1;
                    }
                }
                hold = reset_hold ? 0 : (hold + 1) % frameskip;
            }
        }
        if (!headless) {
            puf_render(&env);
            step_hold = (step_hold + 1) % step_period;
        }
    }

    if (headless) {
        float n = env.log.n;
        float perf = 0.0f;
        float score = 0.0f;
        if (n > 0.0f) {
            Log avg = env.log;
            float* acc = (float*)&avg;
            int nf = (int)(sizeof(Log) / sizeof(float));
            for (int i = 0; i < nf; i++) {
                acc[i] /= n;
            }
            Dict out = {0};
            puf_log(&avg, &out);
            dict_set(&out, "n", n);
            DictItem* pi = dict_find(&out, "perf");
            DictItem* si = dict_find(&out, "score");
            if (pi) perf = (float)pi->value;
            if (si) score = (float)si->value;
            dict_clear(&out);
        }
        printf("CPU_EVAL env=%s file_floats=%d need=%d match=%d untrained=%d n=%.0f perf=%.6f score=%.6f steps=%d agents=%d\n",
            env_name, file_floats, need,
            (need - file_floats <= 7) ? 1 : 0,
            !have_net, n, perf, score, steps, env.num_agents);
        fflush(stdout);
    }

    puf_close(&env);
    puf_ini_free(&ini);
    return 0;
}

#endif
