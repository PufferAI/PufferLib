// Standalone viewer for Craftax (random-action policy).
//
// Build:
//   ./build.sh craftax --fast         # optimized
//   ./build.sh craftax --local        # debug with sanitizers
// Run:
//   ./craftax

#define CRAFTAX_ENABLE_ENV_IMPL
#include "craftax.h"
#include "step_crafting.h"
#include "step_update_mobs.h"
#include "step_spawn_mobs.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x ? x : 0xdeadbeef;
    return x;
}

int main(int argc, char** argv) {
    uint64_t seed = (argc > 1) ? strtoull(argv[1], NULL, 10) : (uint64_t)time(NULL);

    Craftax env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.seed = seed;
    env.rng = (uint32_t)seed;

    // Minimal buffers for a single agent
    env.observations = calloc(CRAFTAX_OBS_SIZE, sizeof(float));
    env.actions = calloc(1, sizeof(float));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(float));

    c_init(&env);
    c_reset(&env);

    uint32_t action_rng = (uint32_t)(seed ^ 0x9E3779B9u);
    bool human_control = false;
    int human_action = CRAFTAX_ACTION_NOOP;

    while (!WindowShouldClose()) {
        // Toggle human control
        if (IsKeyPressed(KEY_H)) human_control = !human_control;

        if (human_control) {
            human_action = CRAFTAX_ACTION_NOOP;
            if (IsKeyPressed(KEY_A) || IsKeyPressed(KEY_LEFT))  human_action = CRAFTAX_ACTION_LEFT;
            if (IsKeyPressed(KEY_D) || IsKeyPressed(KEY_RIGHT)) human_action = CRAFTAX_ACTION_RIGHT;
            if (IsKeyPressed(KEY_W) || IsKeyPressed(KEY_UP))    human_action = CRAFTAX_ACTION_UP;
            if (IsKeyPressed(KEY_S) || IsKeyPressed(KEY_DOWN))  human_action = CRAFTAX_ACTION_DOWN;
            if (IsKeyPressed(KEY_SPACE)) human_action = CRAFTAX_ACTION_DO;
            if (IsKeyPressed(KEY_Z)) human_action = CRAFTAX_ACTION_SLEEP;
            env.actions[0] = (float)human_action;
            if (human_action != CRAFTAX_ACTION_NOOP || IsKeyPressed(KEY_PERIOD)) c_step(&env);
        } else {
            env.actions[0] = (float)(xorshift32(&action_rng) % CRAFTAX_NUM_ACTIONS);
            c_step(&env);
        }

        c_render(&env);
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
