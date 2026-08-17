#include <time.h>

#include "affine_lock.h"

int main(void) {
    AffineLock env;
    memset(&env, 0, sizeof(env));

    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    Dict kwargs = {0};
    dict_set(&kwargs, "start_depth", 2);
    dict_set(&kwargs, "max_depth", 16);
    dict_set(&kwargs, "step_grace", 2);
    dict_set(&kwargs, "perf_weighting", PERF_WEIGHTING_QUADRATIC);
    dict_set(&kwargs, "seed", (double)(unsigned int)time(NULL));
    puf_init(&env, &kwargs);
    dict_clear(&kwargs);

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}
