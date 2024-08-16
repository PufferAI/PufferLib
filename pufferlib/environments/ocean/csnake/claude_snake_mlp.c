#include "claude_snake.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void linear_layer(float* input, float* weights, float* bias, float* output,
                  int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            }
            output[b * output_dim + o] = sum + bias[o];
        }
    }
}


void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

// Function to load weights from binary file
float* load_weights(const char* filename, int* total_weights) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    *total_weights = file_size / sizeof(float);
    rewind(file);

    float* weights = (float*)malloc(file_size);
    if (!weights) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(weights, sizeof(float), *total_weights, file);
    if (read_size != *total_weights) {
        perror("Error reading file");
        free(weights);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return weights;
}

void mlp_forward(float* input, float* hidden, float* output, unsigned int* actions, float* weights,
                 int batch_size, int input_dim, int hidden_dim, int output_dim) {
    // First layer
    linear_layer(input, weights, weights + input_dim*hidden_dim, hidden,
                 batch_size, input_dim, hidden_dim);
    relu(hidden, batch_size * hidden_dim);

    // Second layer
    linear_layer(hidden, weights + input_dim*hidden_dim + hidden_dim,
                 weights + input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim,
                 output, batch_size, hidden_dim, output_dim);

    // Max over outputs
    for (int i = 0; i < batch_size; i++) {
        float max_logit = output[i * output_dim];
        unsigned int atn = 0;
        for (int j = 1; j < output_dim; j++) {
            float out = output[i * output_dim + j];
            if (out > max_logit) {
                max_logit = out;
                atn = j;
            }
        }
        actions[i] = atn;
    }
}

void test_mlp(float* hidden, float* output, unsigned int* actions, float* weights,
        int batch_size, int input_dim, int hidden_dim, int output_dim) {
	float* observations = (float*)malloc(batch_size * input_dim * sizeof(float));
    for (int i=0; i<batch_size*input_dim; i++) {
        observations[i] = (float)i / 1000.0f;
    }
    mlp_forward(observations, hidden, output, actions, weights,
        batch_size, input_dim, hidden_dim, output_dim);

    float weight_sum = 0;
    //for (int i=0; i<input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim; i++) {
    for (int i=0; i<input_dim*hidden_dim; i++) {
        weight_sum += weights[i];
    }

    unsigned int observation_sum = 0;
    for (int i=0; i<batch_size*input_dim; i++) {
        observation_sum += observations[i];
    }

    float hidden_sum = 0;
    for (int i=0; i<batch_size*hidden_dim; i++) {
        if (i < 10) {
            printf("Hidden %d: %f \n", i, hidden[i]);
        }
        hidden_sum += hidden[i];
    }

    float output_sum = 0;
    for (int i=0; i<batch_size*output_dim; i++) {
        output_sum += output[i];
    }

    float action_sum = 0;
    for (int i=0; i<batch_size; i++) {
        action_sum += actions[i];
    }

    printf("Weight sum: %f \n", weight_sum);
    printf("Observation sum: %d \n", observation_sum);
    printf("Hidden sum: %f \n", hidden_sum);
    printf("Output sum: %f \n", output_sum);
    printf("Action sum: %f \n", action_sum);
}

// Main function to demonstrate usage
int main() {
    const char* weight_file = "flat_snake.pt";
    int total_weights;
    float* weights = load_weights(weight_file, &total_weights);

    if (!weights) {
        return 1;
    }

    // Initialize CSnake environment
    int num_snakes = 128;
    int width = 160;
    int height = 160;
    int max_snake_length = 200;
    int food = 128;
    int vision = 5;
    bool leave_corpse_on_death = true;
    float reward_food = 1.0f;
    float reward_corpse = 0.5f;
    float reward_death = -1.0f;

    CSnake* env = allocate_csnake(num_snakes, width, height, max_snake_length,
                                  food, vision, leave_corpse_on_death,
                                  reward_food, reward_corpse, reward_death);

    for (int i = 0; i<num_snakes; i++) {
        env->snake_colors[i] = i%4 + 4;
    }

    // Initialize renderer (optional)
    int cell_size = 5;
    Renderer* renderer = init_renderer(cell_size, width, height);

    // Example usage (adjust these values based on your model architecture)
    int batch_size = num_snakes;
    int input_dim = 11*11;
    int hidden_dim = 128;
    int output_dim = 4;

    // Allocate memory for intermediate results
	float* observations = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    float* output = (float*)malloc(batch_size * output_dim * sizeof(float));

    //test_mlp(hidden, output, env->actions, weights,
    //    batch_size, input_dim, hidden_dim, output_dim);

    // Game loop
    reset(env);
    while (!WindowShouldClose()) {
		// Create a sample input (you should replace this with your actual input data)
        for (int i = 0; i < batch_size * input_dim; i++) {
			observations[i] = (float)env->observations[i];
		}
        /*
		for (int i = 0; i < batch_size; i++) {
            for (int j=0; j<11; j++) {
                for (int k=0; k<11; k++) {
                    observations[121*i + 11*k + j] = (float)observations[121*i + 11*j + k];
                }
            }
		}
        */

		// Perform forward pass
		mlp_forward(observations, hidden, output, env->actions, weights,
			batch_size, input_dim, hidden_dim, output_dim);

        /*
        for (int i=1; i<num_snakes; i++){
            env->actions[i] = rand() % 4;
        }
        */

        // Handle input
        if (IsKeyDown(KEY_UP)) env->actions[0] = 0;
        if (IsKeyDown(KEY_DOWN)) env->actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)) env->actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT)) env->actions[0] = 3;

        // Update game state
        step(env);

        // Render (if renderer is initialized)
        if (renderer != NULL) {
            render(renderer, env);
        }
    }

    // Cleanup
    if (renderer != NULL) {
        close_renderer(renderer);
    }
    free_csnake(env);
    free(weights);
    free(observations);
    free(hidden);
    free(output);

    return 0;
}

