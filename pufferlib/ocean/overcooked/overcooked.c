/* Pure C demo file for Overcooked. Build it with:
 * bash scripts/build_ocean.sh overcooked local (debug)
 * bash scripts/build_ocean.sh overcooked fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets.
 */

#include "overcooked.h"

int main() {
    Overcooked env = {
        .width = 5,
        .height = 5,
        .max_steps = 200,
        .grid_size = 100,
        .reward_dish_served = 10.0f,
        .reward_step_penalty = -0.1f,
        .observation_size = 100  // Adjust based on your observation design
    };
    
    // Allocate required arrays
    env.observations = (float*)calloc(env.observation_size, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    
    // Initialize environment
    init(&env);
    c_reset(&env);
    c_render(&env);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Manual control with shift key, random actions otherwise
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = ACTION_NOOP;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
            if (IsKeyPressed(KEY_SPACE)) env.actions[0] = ACTION_INTERACT;
        } else {
            env.actions[0] = rand() % 6;  // Random action (0-5)
        }
        
        c_step(&env);
        c_render(&env);
        
        // Reset if episode ends
        if (env.terminals[0]) {
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