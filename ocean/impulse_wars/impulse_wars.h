#include <stdint.h>
#ifndef IMPULSE_WARS_OBS_T
#define IMPULSE_WARS_OBS_T
typedef uint8_t obs_t;
#endif

#ifdef __cplusplus
// Game/Box2D/collections-c are C (void* implicit conv); pufferl.cu is C++17.
#define puf_init puf_init_cxx_decl
#define puf_reset puf_reset_cxx_decl
#define puf_step puf_step_cxx_decl
#define puf_render puf_render_cxx_decl
#define puf_close puf_close_cxx_decl
#define puf_log puf_log_cxx_decl
#define CC_ARRAY_NO_IMPL
#include "types.h"
#undef puf_init
#undef puf_reset
#undef puf_step
#undef puf_render
#undef puf_close
#undef puf_log

extern "C" {
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
}

#else

#ifndef IW_DEBUG
#define NDEBUG
#endif
#include "env.h"
#include <pthread.h>

int b2InternalAssertFcn(const char* condition, const char* fileName,
        int lineNumber) {
    fprintf(stderr, "box2d assert %s at %s:%d\n", condition, fileName,
        lineNumber);
    return 1;
}

void puf_init(Env* env, Dict* kwargs) {
    uint8_t num_drones = dict_get(kwargs, "num_drones");
    uint8_t num_agents = dict_get(kwargs, "num_agents");
    int8_t map_idx = dict_get(kwargs, "map_idx");
    uint64_t seed = dict_get(kwargs, "seed");
    bool enable_teams = dict_get(kwargs, "enable_teams");
    bool sitting_duck = dict_get(kwargs, "sitting_duck");
    bool is_training = dict_get(kwargs, "is_training");
    bool continuous = dict_get(kwargs, "continuous");
    // initMaps builds every map inside the env it is given, which would leave
    // the first env in a different state from the rest; give it a scratch env.
    static pthread_mutex_t maps_mu = PTHREAD_MUTEX_INITIALIZER;
    static int maps_ready = 0;
    pthread_mutex_lock(&maps_mu);
    if (!maps_ready) {
        iwEnv* scratch = calloc(1, sizeof(iwEnv));
        initEnv(scratch, num_drones, num_agents, map_idx, seed, enable_teams,
            sitting_duck, is_training, continuous);
        initMaps(scratch);
        puf_close(scratch);
        free(scratch);
        maps_ready = 1;
    }
    pthread_mutex_unlock(&maps_mu);
    initEnv(env, num_drones, num_agents, map_idx, seed, enable_teams,
        sitting_duck, is_training, continuous);
    setRewards(
        env,
        dict_get(kwargs, "reward_win"),
        dict_get(kwargs, "reward_self_kill"),
        dict_get(kwargs, "reward_enemy_death"),
        dict_get(kwargs, "reward_enemy_kill"),
        0.0f,
        0.0f,
        dict_get(kwargs, "reward_death"),
        dict_get(kwargs, "reward_energy_emptied"),
        dict_get(kwargs, "reward_weapon_pickup"),
        dict_get(kwargs, "reward_shield_break"),
        dict_get(kwargs, "reward_shot_hit_coef"),
        dict_get(kwargs, "reward_explosion_hit_coef"));
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);
    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);
    dict_set(out, "n", log->n);
}

#endif

#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1}
#define NUM_ATNS 7
#define OBS_SIZE 1192
