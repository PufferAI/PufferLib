/* Pure C demo file for Overcooked with neural network support.
 * Build it with:
 * bash scripts/build_ocean.sh overcooked local (debug)
 * bash scripts/build_ocean.sh overcooked fast
 */

#include <time.h>
#include "overcooked.h"
#include "puffernet.h"

void demo() {
    int num_agents = 1;  // Single agent environment

    // Load neural network weights for 1 agent
    Weights* weights = load_weights("resources/overcooked/puffer_overcooked_weights.bin", 575004);
    int logit_sizes[] = {6};  // 6 actions: up, down, left, right, interact, noop
    LinearLSTM* net = make_linearlstm(weights, num_agents, 83, logit_sizes, 1);
    
    Overcooked env = {
        .width = 5,
        .height = 5,
        .num_agents = num_agents,
        .max_steps = 400,
        .grid_size = 100,
        .reward_dish_served = 20.0f,
        .reward_step_penalty = 0.0f,
        .observation_size = 83  // 83-dimensional observation vector (including reward)
    };
    
    // Allocate required arrays for multiple agents
    env.observations = (float*)calloc(env.observation_size * num_agents, sizeof(float));
    env.actions = (int*)calloc(num_agents, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    
    // Initialize environment
    init(&env);
    c_reset(&env);
    c_render(&env);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Manual control for single agent with Shift key
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Agent controls (WASD + Space)
            env.actions[0] = ACTION_NOOP;
            if (IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
            if (IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
            if (IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
            if (IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
            if (IsKeyPressed(KEY_SPACE)) env.actions[0] = ACTION_INTERACT;
        } else {
            // Use neural network for actions
            forward_linearlstm(net, env.observations, env.actions);
        }
        
        c_step(&env);
        c_render(&env);
        
        // Reset if any agent's episode ends
        int should_reset = 0;
        for (int i = 0; i < num_agents; i++) {
            if (env.terminals[i]) {
                should_reset = 1;
                break;
            }
        }
        if (should_reset) {
            c_reset(&env);
        }
    }
    
    // Clean up
    free_linearlstm(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

void test_performance(float test_time) {
    int num_agents = 1;

    Overcooked env = {
        .width = 5,
        .height = 5,
        .num_agents = num_agents,
        .max_steps = 400,
        .grid_size = 100,
        .reward_dish_served = 1.0f,
        .reward_step_penalty = 0.0f,
        .observation_size = 83  // 83-dimensional observation vector
    };

    // Allocate required arrays
    env.observations = (float*)calloc(env.observation_size * num_agents, sizeof(float));
    env.actions = (int*)calloc(num_agents, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));

    init(&env);
    c_reset(&env);

    int start = time(NULL);
    int steps = 0;
    while (time(NULL) - start < test_time) {
        // Random actions for performance testing
        for (int i = 0; i < num_agents; i++) {
            env.actions[i] = rand() % 6;
        }
        c_step(&env);
        steps++;

        // Reset if any agent's episode ends
        for (int i = 0; i < num_agents; i++) {
            if (env.terminals[i]) {
                c_reset(&env);
                break;
            }
        }
    }

    int end = time(NULL);
    float sps = (float)(num_agents * steps) / (end - start);
    printf("SPS: %f\n", sps);

    // Clean up
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
    demo();
    // test_performance(30);
    return 0;
}