#include "minimal.h"
#include "puffernet.h"

int main() {
    Env env = {.num_agents = AGENTS};
    int num_obs = 2 + 4*(AGENTS + TARGETS);
    env.observations = calloc(AGENTS*num_obs, sizeof(float));
    env.actions = calloc(2*AGENTS, sizeof(float));
    env.rewards = calloc(AGENTS, sizeof(float));
    env.terminals = calloc(AGENTS, sizeof(float));

    // Works directly with your .bin training checkpoints
    Weights* weights = load_weights("resources/minimal/minimal_weights.bin");
    int logit_sizes[2] = {9, 5};
    PufferNet* net = make_puffernet(weights, env.num_agents, num_obs, 128, 4, logit_sizes, 2);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        forward_puffernet(net, env.observations, env.actions);
        // Always add a human control mode when possible
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 4;
            env.actions[1] = 2;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 0;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 8;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[1] = 4;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[1] = 0;
        }
        c_step(&env);
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
