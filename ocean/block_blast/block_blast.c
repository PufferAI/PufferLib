#include <stdio.h>
#include <time.h>
#include "block_blast.h"

static BlockBlast* g_env = NULL;

static void demo_cleanup(void) {
    if (g_env == NULL) {
        return;
    }
    free(g_env->observations);
    free(g_env->actions);
    free(g_env->rewards);
    free(g_env->terminals);
    free(g_env->action_mask);
    c_close(g_env);
    g_env = NULL;
}

static int random_legal_action(BlockBlast* env) {
    int legal_count = 0;
    for (int i = 0; i < BB_ACTIONS; i++) {
        legal_count += env->action_mask[i] != 0;
    }
    if (legal_count <= 0) {
        return 0;
    }

    int pick = (int)(rand_r(&env->rng) % (unsigned int)legal_count);
    for (int i = 0; i < BB_ACTIONS; i++) {
        if (env->action_mask[i] == 0) {
            continue;
        }
        if (pick == 0) {
            return i;
        }
        pick--;
    }
    return 0;
}

int main(void) {
    BlockBlast env = {
        .num_agents = 1,
        .rng = (unsigned int)time(NULL),
        .max_steps = 512,
        .place_reward = 0.01f,
        .line_reward = 1.0f,
        .combo_reward = 0.15f,
        .free_space_reward = 0.03f,
        .mobility_reward = 0.04f,
        .fill_penalty = 0.05f,
        .no_clear_penalty = 0.01f,
        .dead_space_penalty = 0.0f,
        .fragmentation_penalty = 0.0f,
        .low_mobility_penalty = 0.0f,
        .low_mobility_threshold = 12,
        .invalid_penalty = -0.5f,
        .terminal_penalty = 1.0f,
    };

    g_env = &env;
    atexit(demo_cleanup);

    env.observations = (unsigned char*)calloc(BB_OBS_SIZE, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    env.action_mask = (unsigned char*)calloc(BB_ACTIONS, sizeof(unsigned char));

    c_reset(&env);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = (float)random_legal_action(&env);
            c_step(&env);
        } else if (IsKeyDown(KEY_ENTER)) {
            env.actions[0] = (float)random_legal_action(&env);
            c_step(&env);
        } else if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        c_render(&env);
    }

    demo_cleanup();
    return 0;
}
