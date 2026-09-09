#include "admiral.h"

void demo(void) {
    Admiral env = {
        .num_agents = N_TEAMS,
        .reward_damage_mult = 0.01,
        .width = 2000,
        .height = 2000,
    };
    allocate_env(&env);
    env.curr_level = MAX_LEVEL;
    obs_t observations[N_TEAMS * OBS_SIZE] = {0};
    float actions[N_TEAMS * NUM_ATNS] = {0};
    float rewards[N_TEAMS] = {0};
    float terminals[N_TEAMS] = {0};
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }
    puf_reset(&env);

    env.client = make_client(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        for (int ship = 0; ship < NUM_SHIPS; ship++) {
            int offset = ship * NUM_ACTIONS;
            actions[offset] = 2.0f;
            actions[offset + 1] = 2.0f;
            actions[offset + 2] = 1.0f;
        }

        if (step(&env)) puf_reset(&env);
        puf_render(&env);
    }
    puf_close(&env);
    close_client(env.client);
}

int main() {
    demo();
    return 0;
}
