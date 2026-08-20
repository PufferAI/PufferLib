// Standalone viewer for Craftax (random-action policy).
//
// Build:
//   ./build.sh craftax --cpu          # optimized
//   ./build.sh craftax --debug        # debug with sanitizers
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

int main(int argc, char** argv) {
    uint64_t seed = (argc > 1) ? strtoull(argv[1], NULL, 10) : (uint64_t)time(NULL);

    Craftax env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.seed = seed;
    env.rng = (uint32_t)seed;

    // Minimal buffers for a single agent
    env.agents[0].observations = calloc(CRAFTAX_OBS_SIZE, sizeof(float));
    env.agents[0].actions = calloc(1, sizeof(float));
    env.agents[0].rewards = calloc(1, sizeof(float));
    env.agents[0].terminals = calloc(1, sizeof(float));

    c_init(&env);
    puf_reset(&env);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    return 0;
}
