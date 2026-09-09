#include "admiral.h"
#include <time.h>

static void bind_agents(Admiral* env, obs_t* observations,
        float* actions, float* rewards, float* terminals) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = observations + i * OBS_SIZE;
        env->agents[i].actions = actions + i * NUM_ATNS;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
}

void performance_test() {
    long test_time = 10;
    Admiral env = {
        .num_agents = N_TEAMS,
        .width = 800,
        .height = 600,
        .reward_damage_mult = 0.01f,
        .max_ticks = 3000,
        .rng = 42,
    };
    allocate_env(&env);
    obs_t observations[N_TEAMS * OBS_SIZE] = {0};
    float actions[N_TEAMS * NUM_ATNS] = {0};
    float rewards[2] = {0};
    float terminals[2] = {0};
    bind_agents(&env, observations, actions, rewards, terminals);
    puf_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        for (int ship = 0; ship < NUM_SHIPS; ship++) {
            int offset = ship * NUM_ACTIONS;
            actions[offset] = rand_r(&env.rng) % 5;
            actions[offset + 1] = rand_r(&env.rng) % 5;
            actions[offset + 2] = rand_r(&env.rng) % 3;
        }
        puf_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long)i*env.num_agents / (end - start));
    puf_close(&env);
}

void demo(void) {
    Admiral env = {
        .num_agents = N_TEAMS,
        .reward_damage_mult = 0.01,
        .width = 800,
        .height = 600,
        .max_ticks = 512,
    };
    allocate_env(&env);
    obs_t observations[N_TEAMS * OBS_SIZE] = {0};
    float actions[N_TEAMS * NUM_ATNS] = {0};
    float rewards[N_TEAMS] = {0};
    float terminals[N_TEAMS] = {0};
    bind_agents(&env, observations, actions, rewards, terminals);
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

        if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) actions[1] = 4.0f;
        if (IsKeyDown(KEY_S)) actions[1] = 0.0f;
        if (IsKeyDown(KEY_A)) actions[0] = 0.0f;
        if (IsKeyDown(KEY_D)) actions[0] = 4.0f;
        if (IsKeyDown(KEY_Q)) actions[2] = 0.0f;
        if (IsKeyDown(KEY_E)) actions[2] = 2.0f;

        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    close_client(env.client);
}

int main() {
    demo();
    //performance_test();
    return 0;
}
