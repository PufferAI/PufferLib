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
                sum += input[b * input_dim + i] * weights[o * input_dim + i];
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

// Example MLP forward pass
void mlp_forward(float* input, float* hidden, float* output, unsigned int* actions, float* weights,
		int batch_size, int input_dim, int hidden_dim, int output_dim) {
    // First layer
    linear_layer(input, weights, weights + input_dim * hidden_dim, hidden,
                 batch_size, input_dim, hidden_dim);
    relu(hidden, batch_size * hidden_dim);

    // Second layer
    linear_layer(hidden, weights + input_dim * hidden_dim + hidden_dim,
                 weights + input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim,
                 output, batch_size, hidden_dim, output_dim);

    // Print output (you may want to modify this based on your needs)
	/*
    for (int b = 0; b < batch_size; b++) {
        printf("Sample %d output: ", b);
        for (int o = 0; o < output_dim; o++) {
            printf("%f ", output[b * output_dim + o]);
        }
        printf("\n");
    }
	*/

	// Max over outputs
    int max_logit = -999;
    unsigned int atn = 0;
	for (int i=0; i<batch_size; i++) {
		for (int j=0; i<output_dim; i++) {
			float out = output[i*output_dim + j];
			if (out > max_logit) {
				max_logit = out;
				atn = j;
			}
		}
		actions[i] = atn;
	}
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
    int num_snakes = 16;
    int width = 80;
    int height = 80;
    int max_snake_length = 100;
    int food = 1;
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
    int cell_size = 10;
    Renderer* renderer = init_renderer(cell_size, width, height);

    // Example usage (adjust these values based on your model architecture)
    int batch_size = num_snakes;
    int input_dim = 11*11;
    int hidden_dim = 128;
    int output_dim = 4;

    // Allocate memory for intermediate results
	float* observations = (float*)malloc(batch_size * input_dim);
    float* hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    float* output = (float*)malloc(batch_size * output_dim * sizeof(float));

    // Game loop
    reset(env);
    while (!WindowShouldClose()) {
		// Create a sample input (you should replace this with your actual input data)
		for (int i = 0; i < batch_size * input_dim; i++) {
			observations[i] = (float)env->observations[i];
		}

		// Perform forward pass
		mlp_forward(observations, hidden, output, env->actions, weights,
			batch_size, input_dim, hidden_dim, output_dim);

        for (int i=1; i<num_snakes; i++){
            env->actions[i] = output[i];
            //env->actions[i] = rand() % 4;
        }

        // Handle input
        if (IsKeyPressed(KEY_UP)) env->actions[0] = 0;
        if (IsKeyPressed(KEY_DOWN)) env->actions[0] = 1;
        if (IsKeyPressed(KEY_LEFT)) env->actions[0] = 2;
        if (IsKeyPressed(KEY_RIGHT)) env->actions[0] = 3;

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

