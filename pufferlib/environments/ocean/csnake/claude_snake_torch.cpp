#include "claude_snake.h"
#include <torch/script.h>
#include <iostream>
#include <memory>

// Function to convert observations to torch tensor
torch::Tensor observations_to_tensor(CSnake* env) {
    int obs_size = env->num_snakes * (2 * env->vision + 1) * (2 * env->vision + 1);
    torch::Tensor tensor = torch::zeros({env->num_snakes, 11, 11}, torch::kLong);
    auto tensor_accessor = tensor.accessor<int64_t, 3>();
    
    for (int i = 0; i < env->num_snakes; i++) {
        for (int r = 0; r < 11; r++) {
            for (int c = 0; c < 11; c++) {
                tensor_accessor[i][r][c] = static_cast<int64_t>(env->observations[i * 121 + r * 11 + c]);
            }
        }
    }
    return tensor;
}

// Function to update actions from model output
void update_actions_from_tensor(CSnake* env, torch::Tensor actions_tensor) {
    auto actions_accessor = actions_tensor.accessor<int64_t, 1>();
    for (int i = 0; i < env->num_snakes; i++) {
        env->actions[i] = static_cast<unsigned int>(actions_accessor[i]);
    }
}

int main() {
    // Initialize CSnake environment (same as before)
    int num_snakes = 16;
    int width = 80;
    int height = 80;
    int max_snake_length = 100;
    int food = 32;
    int vision = 5; // Changed to 5 to match 11x11 observation size
    bool leave_corpse_on_death = true;
    float reward_food = 1.0f;
    float reward_corpse = 0.5f;
    float reward_death = -1.0f;
    CSnake* env = allocate_csnake(num_snakes, width, height, max_snake_length,
                                  food, vision, leave_corpse_on_death,
                                  reward_food, reward_corpse, reward_death);
    for (int i = 0; i < num_snakes; i++) {
        env->snake_colors[i] = i % 4 + 4;
    }

    // Initialize renderer (optional)
    int cell_size = 16;
    Renderer* renderer = init_renderer(cell_size, width, height);

    // Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("snake_model_traced.pt");
    }
    catch (const c10::Error& e) {
        printf("Error loading the model\n");
        return -1;
    }
    printf("Model loaded successfully\n");

    // Initialize LSTM state
    torch::Tensor h = torch::zeros({1, num_snakes, 128}); // Adjust the size based on your LSTM configuration
    torch::Tensor c = torch::zeros({1, num_snakes, 128}); // Adjust the size based on your LSTM configuration

    // Game loop
    printf("Resetting the environment\n");
    reset(env);
    printf("Reset complete\n");

    while (!WindowShouldClose()) {
        // Convert observations to tensor
        printf("Converting observations to tensor\n");
        torch::Tensor input_tensor = observations_to_tensor(env);
        printf("Converted observations to tensor\n");

        // Perform inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        inputs.push_back(std::make_tuple(h, c));

        auto output = module.forward(inputs).toTuple();

        torch::Tensor action_tensor = output->elements()[0].toTensor();
        torch::Tensor logprob_tensor = output->elements()[1].toTensor();
        torch::Tensor entropy_tensor = output->elements()[2].toTensor();
        torch::Tensor value_tensor = output->elements()[3].toTensor();
        auto lstm_state = output->elements()[4].toTuple();
        h = lstm_state->elements()[0].toTensor();
        c = lstm_state->elements()[1].toTensor();

        // Update actions from model output
        update_actions_from_tensor(env, action_tensor);

        // Handle input for the first snake (optional, for manual control)
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
    return 0;
}
