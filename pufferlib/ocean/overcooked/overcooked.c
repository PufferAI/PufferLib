/* Pure C demo file for Overcooked. Build it with:
 * bash scripts/build_ocean.sh overcooked local (debug)
 * bash scripts/build_ocean.sh overcooked fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets.
 */

#include "overcooked.h"

int main() {
    int num_agents = 2;  // Support 2 agents for cooperative play
    
    Overcooked env = {
        .width = 5,
        .height = 5,
        .num_agents = num_agents,
        .max_steps = 200,
        .grid_size = 100,
        .reward_dish_served = 10.0f,
        .reward_step_penalty = -0.1f,
        .observation_size = 5 * 5 * 21 + 2  // 5x5 grid * 21 channels + 2 global values = 527
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
        // Manual control for agent 0 with WASD/Space
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Agent 0 controls (WASD + Space)
            env.actions[0] = ACTION_NOOP;
            if (IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
            if (IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
            if (IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
            if (IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
            if (IsKeyPressed(KEY_SPACE)) env.actions[0] = ACTION_INTERACT;
            
            // Agent 1 controls (Arrow keys + Enter)
            env.actions[1] = ACTION_NOOP;
            if (IsKeyDown(KEY_UP)) env.actions[1] = ACTION_UP;
            if (IsKeyDown(KEY_DOWN)) env.actions[1] = ACTION_DOWN;
            if (IsKeyDown(KEY_LEFT)) env.actions[1] = ACTION_LEFT;
            if (IsKeyDown(KEY_RIGHT)) env.actions[1] = ACTION_RIGHT;
            if (IsKeyPressed(KEY_ENTER)) env.actions[1] = ACTION_INTERACT;
        } else {
            // Random actions for both agents
            for (int i = 0; i < num_agents; i++) {
                env.actions[i] = rand() % 6;  // Random action (0-5)
            }
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
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    
    return 0;
}