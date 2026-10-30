/* Compile once as C for the layout probe, then again as C++17 (or CUDA C++17)
 * with FC_LINKAGE_DRIVER for main. Link both to binding.c compiled as C.
 * This tests the public header without including any game/viewer implementation. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stddef.h>
/* The native trainer includes ini.h before the environment header as well. */
#include "ini.h"
#include "fight_caves.h"

#ifdef __cplusplus
#include <type_traits>
static_assert(std::is_standard_layout<Env>::value, "Env must have a C-compatible layout");
static_assert(std::is_trivially_copyable<Env>::value, "Puffer allocates/reallocates Env");
#define FC_TEST_ALIGNOF alignof
#else
#define FC_TEST_ALIGNOF _Alignof
#endif

static const size_t layout[] = {
    sizeof(Env), FC_TEST_ALIGNOF(Env),
    offsetof(Env, log), offsetof(Env, agents), offsetof(Env, num_agents),
    offsetof(Env, rng), offsetof(Env, tag), offsetof(Env, boundary_reached),
    offsetof(Env, state), offsetof(Env, viewer), offsetof(Env, reward_params),
    offsetof(Env, initial_sharks), offsetof(Env, initial_prayer_doses),
    offsetof(Env, reward_runtime), offsetof(Env, seed_counter),
    sizeof(Agent), FC_TEST_ALIGNOF(Agent), offsetof(Agent, observations),
    offsetof(Agent, actions), offsetof(Agent, rewards), offsetof(Agent, terminals),
    offsetof(Agent, action_mask), offsetof(Agent, policy),
    sizeof(Log), FC_TEST_ALIGNOF(Log), offsetof(Log, n),
    sizeof(FcState), FC_TEST_ALIGNOF(FcState),
    offsetof(FcState, player), offsetof(FcState, npcs), offsetof(FcState, terminal),
    sizeof(FcRewardParams), sizeof(FcRewardRuntime),
    sizeof(Dict), offsetof(Dict, items), sizeof(DictItem), offsetof(DictItem, value),
};

#ifdef __cplusplus
extern "C" {
#endif
const size_t* fc_test_c_layout(size_t* count);
#ifdef __cplusplus
}
#endif

#ifndef FC_LINKAGE_DRIVER
const size_t* fc_test_c_layout(size_t* count) {
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}
#else
int main(void) {
    size_t count = 0;
    const size_t* c_layout = fc_test_c_layout(&count);
    assert(count == sizeof(layout) / sizeof(layout[0]));
    assert(memcmp(layout, c_layout, sizeof(layout)) == 0);

    Dict kwargs = {};
    dict_set(&kwargs, "w_progress", 0.001);
    dict_set(&kwargs, "shape_no_attack_start", 17);
    /* Mirror Puffer's initialization before reallocating and wiring buffers. */
    Env* envs = (Env*)calloc(3, sizeof(Env));
    assert(envs);
    for (int i = 0; i < 2; i++) {
        envs[i].rng = 73 + i;
        puf_init(&envs[i], &kwargs);
        assert(envs[i].num_agents == 1 && envs[i].viewer == NULL);
        assert(envs[i].reward_params.w_progress == 0.001f);
        assert(envs[i].reward_params.shape_no_attack_start == 17);
    }
    Env* resized = (Env*)realloc(envs, 2 * sizeof(Env));
    assert(resized);
    envs = resized;
    float obs[2][OBS_SIZE] = {};
    float actions[2][NUM_ATNS] = {};
    float rewards[2] = {}, terminals[2] = {};
    unsigned char masks[2][FC_PUFFER_MASK_SIZE] = {};
    for (int i = 0; i < 2; i++) {
        Agent* a = &envs[i].agents[0];
        a->observations = obs[i];
        a->actions = actions[i];
        a->rewards = &rewards[i];
        a->terminals = &terminals[i];
        a->action_mask = masks[i];
        puf_reset(&envs[i]);
        assert(envs[i].state.tick == 0 && envs[i].seed_counter == 1);
    }
    assert(envs[0].state.rng_seed != envs[1].state.rng_seed);

    uint32_t trace = 2166136261u;
    int episodes = 0;
    for (int tick = 0; tick < 2048; tick++) {
        for (int i = 0; i < 2; i++) {
            Env* env = &envs[i];
            actions[i][0] = (tick + i) % 17;
            actions[i][1] = (tick / 3) % 9;
            actions[i][2] = (tick / 7) % 8;
            puf_step(env);
            assert(env->viewer == NULL); /* No graphics initialization in training. */
            assert(isfinite(rewards[i]));
            float expected[FC_POLICY_OBS_SIZE], mask[FC_ACTION_MASK_SIZE];
            fc_write_obs(&env->state, expected);
            fc_write_mask(&env->state, mask);
            assert(memcmp(obs[i], expected, FC_POLICY_OBS_SIZE * sizeof(float)) == 0);
            for (int m = 0; m < FC_PUFFER_MASK_SIZE; m++) {
                assert(obs[i][FC_POLICY_OBS_SIZE + m] == mask[m]);
                assert(masks[i][m] == (mask[m] != 0.0f));
            }
            episodes += terminals[i] != 0.0f;
            trace = (trace ^ fc_state_hash(&env->state)) * 16777619u;
        }
    }
    assert(episodes > 0);
    Dict out = {};
    puf_log(&envs[0].log, &out);
    assert(dict_get(&out, "perf") == envs[0].log.jad_kill_rate);
    for (int i = 0; i < 2; i++) puf_close(&envs[i]);
    free(envs);
    dict_clear(&kwargs);
    dict_clear(&out);
    printf("C/C++ linkage: %zu layout checks, 4096 steps, %d episodes, trace=%08x passed\n",
           count, episodes, trace);
    return 0;
}
#endif
