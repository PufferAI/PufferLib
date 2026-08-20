#include "tower_defence.h"

#define OBS_TENSOR_T FloatTensor
#define OBS_SIZE TD_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES                                                                                  \
    { TD_NUM_ACTIONS }
#define MY_ACTION_MASK TD_NUM_ACTIONS

#define Env TowerDefence
#include "vecenv.h"

static void td_config_error(const char *key, const char *message) {
    fprintf(stderr, "Invalid tower_defence config %s: %s\n", key, message);
    exit(1);
}

static double td_get_finite_double(Dict *kwargs, const char *key, double fallback) {
    DictItem *item = dict_get_unsafe(kwargs, key);
    double value = item == NULL ? fallback : item->value;
    if (!isfinite(value)) {
        td_config_error(key, "expected a finite number");
    }
    return value;
}

static float td_get_finite_float(Dict *kwargs, const char *key, float fallback) {
    double value = td_get_finite_double(kwargs, key, (double)fallback);
    if (value < -(double)FLT_MAX || value > (double)FLT_MAX) {
        td_config_error(key, "outside float range");
    }
    return (float)value;
}

static float td_get_reward_float(Dict *kwargs, const char *key, float fallback) {
    float value = td_get_finite_float(kwargs, key, fallback);
    if (fabsf(value) > 1000000.0f) {
        td_config_error(key, "reward magnitude must not exceed 1e6");
    }
    return value;
}

static int td_get_int(Dict *kwargs, const char *key, int fallback) {
    double value = td_get_finite_double(kwargs, key, (double)fallback);
    if (value < (double)INT_MIN || value > (double)INT_MAX || floor(value) != value) {
        td_config_error(key, "expected an integer");
    }
    return (int)value;
}

static int td_get_positive_int(Dict *kwargs, const char *key, int fallback) {
    int value = td_get_int(kwargs, key, fallback);
    if (value < 1) {
        td_config_error(key, "expected a positive integer");
    }
    return value;
}

static int td_get_binary_int(Dict *kwargs, const char *key, int fallback) {
    int value = td_get_int(kwargs, key, fallback);
    if (value != 0 && value != 1) {
        td_config_error(key, "expected 0 or 1");
    }
    return value;
}

void my_init(Env *env, Dict *kwargs) {
    memset(&env->log, 0, sizeof(env->log));
    td_init(env);
    env->max_episode_steps =
        td_get_positive_int(kwargs, "max_episode_steps", TD_DEFAULT_MAX_EPISODE_STEPS);
    env->base_seed = td_get_int(kwargs, "base_seed", TD_DEFAULT_BASE_SEED);
    env->target_round = td_get_positive_int(kwargs, "target_round", TD_DEFAULT_TARGET_ROUND);
    env->invalid_action_reward =
        td_get_reward_float(kwargs, "invalid_action_reward", TD_DEFAULT_INVALID_ACTION_REWARD);
    env->reward_enemy_scale =
        td_get_reward_float(kwargs, "reward_enemy_scale", TD_DEFAULT_REWARD_ENEMY_SCALE);
    env->reward_round_clear_scale = td_get_reward_float(kwargs, "reward_round_clear_scale",
                                                        TD_DEFAULT_REWARD_ROUND_CLEAR_SCALE);
    env->reward_tower_placement_scale = td_get_reward_float(
        kwargs, "reward_tower_placement_scale", TD_DEFAULT_REWARD_TOWER_PLACEMENT_SCALE);
    env->reward_tower_investment_scale = td_get_reward_float(
        kwargs, "reward_tower_investment_scale", TD_DEFAULT_REWARD_TOWER_INVESTMENT_SCALE);
    env->reward_sell_penalty_scale = td_get_reward_float(kwargs, "reward_sell_penalty_scale",
                                                         TD_DEFAULT_REWARD_SELL_PENALTY_SCALE);
    env->reward_trigger_ready_noop_penalty = td_get_reward_float(
        kwargs, "reward_trigger_ready_noop_penalty", TD_DEFAULT_REWARD_TRIGGER_READY_NOOP_PENALTY);
    env->reward_round_advance_scale = td_get_reward_float(kwargs, "reward_round_advance_scale",
                                                          TD_DEFAULT_REWARD_ROUND_ADVANCE_SCALE);
    env->reward_score_advance_scale = td_get_reward_float(kwargs, "reward_score_advance_scale",
                                                          TD_DEFAULT_REWARD_SCORE_ADVANCE_SCALE);
    env->reward_leak_penalty_scale = td_get_reward_float(kwargs, "reward_leak_penalty_scale",
                                                         TD_DEFAULT_REWARD_LEAK_PENALTY_SCALE);
    env->reward_completion_bonus =
        td_get_reward_float(kwargs, "reward_completion_bonus", TD_DEFAULT_REWARD_COMPLETION_BONUS);
    env->reward_defeat_penalty =
        td_get_reward_float(kwargs, "reward_defeat_penalty", TD_DEFAULT_REWARD_DEFEAT_PENALTY);
    env->reward_truncation_penalty = td_get_reward_float(kwargs, "reward_truncation_penalty",
                                                         TD_DEFAULT_REWARD_TRUNCATION_PENALTY);
    env->reward_clamp_enabled =
        td_get_binary_int(kwargs, "reward_clamp_enabled", TD_DEFAULT_REWARD_CLAMP_ENABLED);
    env->reward_clamp_min =
        td_get_reward_float(kwargs, "reward_clamp_min", TD_DEFAULT_REWARD_CLAMP_MIN);
    env->reward_clamp_max =
        td_get_reward_float(kwargs, "reward_clamp_max", TD_DEFAULT_REWARD_CLAMP_MAX);
    if (env->reward_clamp_enabled && env->reward_clamp_min > env->reward_clamp_max) {
        td_config_error("reward_clamp_min", "must be less than or equal to reward_clamp_max");
    }
}

void my_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
}
