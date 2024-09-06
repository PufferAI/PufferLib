#include "moba.h"

// File format is obained by flattening and concatenating all pytorch layers
int load_weights(const char* filename, float* weights, int num_weights) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    rewind(file);
    size_t read_size = fread(weights, sizeof(float), num_weights, file);
    fclose(file);
    if (read_size != num_weights) {
        perror("Error reading file");
        return 1;
    }
    return 0;
}

void linear_layer(float* input, float* weights, float* bias, float* output,
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

void linear_layer_accumulate(float* input, float* weights, float* bias, float* output,
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

void relu(float* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = fmaxf(0.0f, data[i]);
}

float sigmoid(float x);
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void convolution_layer(float* input, float* weights, float* bias,
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

void lstm(float* input, float* state, float* weights_input,
        float* weights_state, float* bias_input, float*bias_state,
        float* output_cell, float *buffer, 
        int batch_size, int input_size, int hidden_size) {
    linear_layer(input, weights_input, bias_input, buffer, batch_size, input_size, 4*hidden_size);
    linear_layer_accumulate(state, weights_state, bias_state, buffer, batch_size, hidden_size, 4*hidden_size);

    // Activation functions
    for (int b=0; b<batch_size; b++) {
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<2*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = sigmoid(buffer[buf_adr]);
        }
        for (int i=2*hidden_size; i<3*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = tanh(buffer[buf_adr]);
        }
        for (int i=3*hidden_size; i<4*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = sigmoid(buffer[buf_adr]);
        }
    }

    // Gates
    for (int b=0; b<batch_size; b++) {
        int inp_offset = b*hidden_size;
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<hidden_size; i++) {
            output_cell[inp_offset + i] = (
                buffer[b_offset + hidden_size + i] * input[inp_offset + i]
                + buffer[b_offset + i] * buffer[b_offset + 2*hidden_size + i]
            );
            state[inp_offset + i] = (
                buffer[b_offset + 3*hidden_size + i] * tanh(output_cell[inp_offset + i])
            );
        }
    }
 
}

float get_out(float* output, int b, int oc, int h, int w) {
    return output[(b*32*3*3) + (oc*3*3) + (h*3) + w];
}

int main() {
    int num_agents = 10;
    int num_creeps = 100;
    int num_neutrals = 72;
    int num_towers = 24;
    int vision_range = 5;
    float agent_speed = 1.0;
    bool discretize = true;
    float reward_death = -1.0;
    float reward_xp = 0.006;
    float reward_distance = 0.05;
    float reward_tower = 3.0;

    /*
    int num_weights = 32*19*5*5;
    int num_bias = 32;
    float* weights = calloc(num_weights + num_bias, sizeof(float));
    float* bias = &weights[num_weights];
    load_weights("moba_weights.pt", weights, num_weights+num_bias) ;

    int batch_size = 16;
    int num_input = batch_size*(11*11*19 + 26);
    float *input = calloc(num_input, sizeof(float));
    for (int i = 0; i < num_input; i++) {
        input[i] = i;
    }

    int num_output = batch_size*32*3*3;
    float* output = calloc(num_output, sizeof(float));

    convolution_layer(input, weights, bias, output, batch_size, 11, 11, 19, 32, 5, 3);
    */

    int batch_size = 16;
    int input_size = 128;
    int hidden_size = 128;

    int num_input = batch_size*input_size;
    int num_buffer = 4*batch_size*hidden_size;
    int num_output = batch_size*hidden_size;

    float* input = calloc(num_input, sizeof(float));
    float* buffer = calloc(num_buffer, sizeof(float));
    float* output_cell = calloc(num_output, sizeof(float));
    float* state = calloc(num_output, sizeof(float));
    float* weights_input = calloc(4*hidden_size*input_size, sizeof(float));
    float* weights_state = calloc(4*hidden_size*hidden_size, sizeof(float));
    float* bias_input = calloc(4*hidden_size, sizeof(float));
    float* bias_state = calloc(4*hidden_size, sizeof(float));

    for (int i = 0; i < num_input; i++) {
        input[i] = i;
    }
    for (int i = 0; i < num_buffer; i++) {
        buffer[i] = i;
    }
    for (int i = 0; i < num_output; i++) {
        output_cell[i] = i;
        state[i] = i;
    }
    for (int i = 0; i < hidden_size*input_size; i++) {
        weights_input[i] = i;
    }
    for (int i = 0; i < hidden_size*hidden_size; i++) {
        weights_state[i] = i;
    }
    for (int i = 0; i < hidden_size; i++) {
        bias_input[i] = i;
        bias_state[i] = i;
    }

    lstm(input, state, weights_input, weights_state, bias_input, bias_state,
        output_cell, buffer, batch_size, input_size, hidden_size);

    MOBA* env = init_moba(num_agents, num_creeps, num_neutrals, num_towers, vision_range,
        agent_speed, discretize, reward_death, reward_xp, reward_distance, reward_tower);

    GameRenderer* renderer = init_game_renderer(32, 41, 23);

    reset(env);
    int frame = 0;
    while (!WindowShouldClose()) {
        if (frame % 12 == 0) {
            step(env);
        }
        render_game(renderer, env, frame);
        frame = (frame + 1) % 12;
    }
    free_moba(env);
    close_game_renderer(renderer);
    return 0;
}