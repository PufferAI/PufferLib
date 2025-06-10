/* Pure C demo file for Robocode. Build it with:
 * bash scripts/build_ocean.sh robocode local (debug)
 * bash scripts/build_ocean.sh robocode fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "robocode.h"

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"

int main() {
    int num_agents = 2;  // Agent + Adversarial
    int num_obs = 15;    // 15 observations per agent


    // Box action space: 5 continuous actions
    // [move, turn, gun_turn, radar_turn, fire]
 

    Robocode env = {
        .width = 768,
        .height = 576,
        .num_agents = num_agents
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    env.observations = calloc(env.num_agents * num_obs, sizeof(float));
    env.actions = calloc(5 * env.num_agents, sizeof(float));  // 5 float actions per agent
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    // while(True) will break web builds
    while (!WindowShouldClose()) {
        // Generate random actions for testing (only for the learning agent)
        // The adversarial agent uses its own hardcoded strategy
        for (int i = 0; i < env.num_agents; i++) {  // Only control agent 0, adversarial is agent 1
            int action_offset = i * 5;
            
            // Random actions within valid ranges
            env.actions[action_offset + 0] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;    // move: -1 to 1
            env.actions[action_offset + 1] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // turn: -10 to 10
            env.actions[action_offset + 2] = ((float)rand() / RAND_MAX) * 40.0f - 20.0f;  // gun_turn: -20 to 20
            env.actions[action_offset + 3] = ((float)rand() / RAND_MAX) * 90.0f - 45.0f;  // radar_turn: -45 to 45
            env.actions[action_offset + 4] = ((float)rand() / RAND_MAX) * 3.0f;           // fire: 0 to 3
        }

        // If you have a trained model, use this instead of random actions: 
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    
    return 0;
}