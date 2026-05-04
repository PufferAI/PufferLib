#include "tower_defence.h"

#define OBS_TENSOR_T FloatTensor
#define OBS_SIZE TD_OBS_SIZE
#define NUM_ATNS TD_FACTOR_NUM_ATNS
#define ACT_SIZES {TD_FACTOR_VERB_COUNT, TD_NUM_PLACEMENT_SLOTS, TD_NUM_UPGRADE_PATHS}

#define Env TowerDefence
#include "vecenv.h"

static int td_get_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return fallback;
    }
    return (int)item->value;
}

static float td_get_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return fallback;
    }
    return item->value;
}

void my_init(Env* env, Dict* kwargs) {
    memset(&env->log, 0, sizeof(env->log));
    td_init(env);
    env->max_episode_steps = td_get_int(
        kwargs, "max_episode_steps", TD_DEFAULT_MAX_EPISODE_STEPS);
    env->base_seed = td_get_int(kwargs, "base_seed", TD_DEFAULT_BASE_SEED);
    env->raw_observation_scalars = td_get_int(kwargs, "raw_observation_scalars", 1);
    env->episode_index = 0;
    env->step_count = 0;
    env->invalid_action_count = 0;
    memset(env->valid_actions, 0, sizeof(env->valid_actions));
    env->valid_actions[TD_ACTION_NOOP] = 1;
    env->invalid_action_reward = td_get_float(
        kwargs, "invalid_action_reward", TD_DEFAULT_INVALID_ACTION_REWARD);
    env->episode_return = 0.0f;
    env->latest_score = 0.0f;
    env->latest_perf = 0.0f;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
}
