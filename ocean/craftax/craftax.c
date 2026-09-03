#include "craftax.h"

int main(void) {
    Craftax env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.rng = 1;
    env.seed = 1;
    env.use_action_mask = 1;

    env.agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.agents[0].action_mask = (unsigned char*)calloc(ATN_DIM, 1);
    puf_reset(&env);
    env.agents[0].actions[0] = -1.0f;

    puf_render(&env);
    while (!WindowShouldClose()) {
        int action = key_to_action();
        if (action < 0) {
            env.agents[0].actions[0] = -1.0f;
            puf_render(&env);
            continue;
        }
        env.agents[0].actions[0] = (float)action;
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    free(env.agents[0].action_mask);
    return 0;
}
