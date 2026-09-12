/* Run normally for headless trace parity, or with --render for the standard
 * Puffer render hook. FC_INTEGRATION_ORIGINAL allows the same trace to be
 * compiled against the unmodified adapter for an independent comparison. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "fight_caves.h"
#include <assert.h>

static FightCaves* test_env(void) {
    FightCaves* env = calloc(1, sizeof(*env));
    assert(env);
    env->num_agents = 1;
    env->rng = 73;
    env->observations = calloc(FC_PUFFER_OBS_SIZE, sizeof(float));
    env->actions = calloc(FC_PUFFER_NUM_ATNS, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));
    env->action_mask = calloc(FC_PUFFER_MASK_SIZE, 1);
    env->reward_params = fc_reward_default_params();
    fc_init(&env->state);
    c_reset(env);
    return env;
}

static void test_free(FightCaves* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->action_mask);
    free(env);
}

static uint32_t digest(uint32_t hash, const void* data, size_t size) {
    const unsigned char* bytes = data;
    for (size_t i = 0; i < size; i++) hash = (hash ^ bytes[i]) * 16777619u;
    return hash;
}

int main(int argc, char** argv) {
    int graphical = argc > 1 && strcmp(argv[1], "--render") == 0;
    FightCaves* env = test_env();
    uint32_t trace = 2166136261u;
    int episodes = 0;
#ifndef FC_INTEGRATION_ORIGINAL
    FightCaves* rendered = test_env();
    if (graphical) {
        c_render(rendered);
        rendered->viewer->tps = 60.0f;
    } else {
        /* Exercise snapshot capture without allocating graphics in a test. */
        rendered->viewer = calloc(1, sizeof(*rendered->viewer));
        assert(rendered->viewer);
    }
#else
    (void)graphical;
#endif
    for (int tick = 0; tick < 2048; tick++) {
        env->actions[0] = tick % 17;
        env->actions[1] = (tick / 3) % 9;
        env->actions[2] = (tick / 7) % 8;
#ifndef FC_INTEGRATION_ORIGINAL
        memcpy(rendered->actions, env->actions, sizeof(float) * FC_PUFFER_NUM_ATNS);
#endif
        c_step(env);
        uint32_t hash = fc_state_hash(&env->state);
        trace = digest(trace, &hash, sizeof(hash));
        trace = digest(trace, env->observations, sizeof(float) * FC_PUFFER_OBS_SIZE);
        trace = digest(trace, env->action_mask, FC_PUFFER_MASK_SIZE);
        trace = digest(trace, env->rewards, sizeof(float));
        trace = digest(trace, env->terminals, sizeof(float));
        episodes += env->terminals[0] != 0;
#ifndef FC_INTEGRATION_ORIGINAL
        c_step(rendered);
        assert(hash == fc_state_hash(&rendered->state));
        assert(memcmp(env->observations, rendered->observations,
                      sizeof(float) * FC_PUFFER_OBS_SIZE) == 0);
        assert(memcmp(env->action_mask, rendered->action_mask,
                      FC_PUFFER_MASK_SIZE) == 0);
        assert(env->rewards[0] == rendered->rewards[0]);
        assert(env->terminals[0] == rendered->terminals[0]);
        assert(rendered->viewer->pending_frame);
        if (env->terminals[0]) {
            assert(rendered->state.tick == 0);
            assert(fc_is_terminal(&rendered->viewer->pending_state));
        }
        if (graphical) {
            c_render(rendered);
            assert(hash == fc_state_hash(&rendered->state));
            assert(!rendered->viewer->pending_frame);
            if (env->terminals[0])
                assert(fc_is_terminal(&rendered->viewer->state));
            if (tick == 8) {
                Image frame = LoadImageFromScreen();
                assert(ExportImage(frame, "build/fight-caves-standard-eval.png"));
                UnloadImage(frame);
            }
        }
#endif
    }
    assert(episodes > 0);
#ifndef FC_INTEGRATION_ORIGINAL
    c_reset(env);
    c_reset(rendered);
    if (graphical) {
        c_render(rendered);
        assert(fc_state_hash(&rendered->viewer->state) == fc_state_hash(&env->state));
    }
#endif
    printf("adapter trace: steps=2048 episodes=%d digest=%08x\n", episodes, trace);
#ifndef FC_INTEGRATION_ORIGINAL
    if (!graphical) {
        free(rendered->viewer);
        rendered->viewer = NULL;
    }
    test_free(rendered);
#endif
    test_free(env);
    return 0;
}
