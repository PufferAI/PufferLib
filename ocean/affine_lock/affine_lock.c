#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "affine_lock.h"

static AffineLock* g_env = NULL;
static AffineLockShared* g_shared = NULL;

static void demo_cleanup(void) {
    if (g_env != NULL) {
        free(g_env->observations);
        free(g_env->actions);
        free(g_env->rewards);
        free(g_env->terminals);
        c_close(g_env);
        g_env = NULL;
    }
    if (g_shared != NULL) {
        affine_lock_free_shared(g_shared);
        free(g_shared);
        g_shared = NULL;
    }
}

static int key_to_action(void) {
    static const int keys[AFFINE_LOCK_NUM_ACTIONS] = {
        KEY_ONE, KEY_TWO, KEY_THREE, KEY_FOUR,
        KEY_FIVE, KEY_SIX, KEY_SEVEN, KEY_EIGHT,
    };

    for (int i = 0; i < AFFINE_LOCK_NUM_ACTIONS; i++) {
        if (IsKeyPressed(keys[i])) {
            return i;
        }
    }
    return -1;
}

int main(void) {
    g_shared = (AffineLockShared*)calloc(1, sizeof(AffineLockShared));
    if (g_shared == NULL ||
            affine_lock_init_shared(g_shared, 2, 16, 2, AFFINE_LOCK_PERF_WEIGHTING_QUADRATIC) != 0) {
        fprintf(stderr, "failed to initialize affine_lock demo\n");
        demo_cleanup();
        return 1;
    }
    if (affine_lock_prepare_visible_targets(g_shared) != 0) {
        fprintf(stderr, "failed to configure affine_lock demo\n");
        demo_cleanup();
        return 1;
    }

    AffineLock env;
    memset(&env, 0, sizeof(env));
    g_env = &env;
    atexit(demo_cleanup);

    // Standalone demo buffers match the FloatTensor/float vecenv contract.
    env.observations = (float*)calloc(AFFINE_LOCK_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(AFFINE_LOCK_NUM_ATNS, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    if (env.observations == NULL || env.actions == NULL ||
            env.rewards == NULL || env.terminals == NULL) {
        fprintf(stderr, "failed to allocate affine_lock demo buffers\n");
        return 1;
    }

    affine_lock_init_env(&env, g_shared, (unsigned int)time(NULL));
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsWindowReady() && IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        int action = key_to_action();

        if (action >= 0) {
            env.actions[0] = (float)action;
            c_step(&env);
        }

        c_render(&env);
    }

    demo_cleanup();
    return 0;
}
