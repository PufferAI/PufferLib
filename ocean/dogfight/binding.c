#include "dogfight.h"
#define OBS_SIZE 26
#define NUM_ATNS 5
#define ACT_SIZES {1, 1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env Dogfight
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;

    int obs_scheme = (int)dict_get(kwargs, "obs_scheme")->value;
    int curriculum_enabled = (int)dict_get(kwargs, "curriculum_enabled")->value;
    int curriculum_randomize = (int)dict_get(kwargs, "curriculum_randomize")->value;

    RewardConfig rcfg = {
        .aim_scale = dict_get(kwargs, "reward_aim_scale")->value,
        .closing_scale = dict_get(kwargs, "reward_closing_scale")->value,
        .neg_g = dict_get(kwargs, "penalty_neg_g")->value,
        .control_rate_penalty = dict_get(kwargs, "control_rate_penalty")->value,
        .low_altitude_threshold = dict_get(kwargs, "low_altitude_threshold")->value,
        .low_altitude_penalty = dict_get(kwargs, "low_altitude_penalty")->value,
        .speed_min = dict_get(kwargs, "speed_min")->value,
        .aim_decay_stage = dict_get(kwargs, "aim_decay_stage")->value,
        .shaping_decay_start = (long)dict_get(kwargs, "shaping_decay_start")->value,
        .shaping_decay_end = (long)dict_get(kwargs, "shaping_decay_end")->value,
        .energy_gain_scale = dict_get(kwargs, "energy_gain_scale")->value,
        .energy_loss_scale = dict_get(kwargs, "energy_loss_scale")->value,
        .energy_advantage_scale = dict_get(kwargs, "energy_advantage_scale")->value,
    };

    init(env, obs_scheme, &rcfg, curriculum_enabled, curriculum_randomize, env->rng);
    c_reset(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
