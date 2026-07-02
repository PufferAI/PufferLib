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
    // Core metrics (vecenv divides every Log float by n before calling my_log,
    // so each export here is a per-episode mean — perf is kill rate, etc.)
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);

    // Combat
    dict_set(out, "shots_fired", log->shots_fired);
    dict_set(out, "accuracy", log->accuracy);
    dict_set(out, "sp_player_kills", log->sp_player_kills);
    dict_set(out, "sp_opp_kills", log->sp_opp_kills);

    // Curriculum / stage
    dict_set(out, "stage", log->stage);
    dict_set(out, "avg_stage", log->stage_sum);
    dict_set(out, "avg_stage_weight", log->total_stage_weight);

    // Directional + control health (KEY: surfaces "always banks one direction")
    dict_set(out, "avg_abs_bias", log->total_abs_bias);
    dict_set(out, "avg_signed_bias", log->total_signed_bias);
    dict_set(out, "avg_control_rate", log->total_control_rate);

    // Death-mode diagnostics
    dict_set(out, "player_ground", log->player_ground_hits);
    dict_set(out, "opponent_ground", log->opponent_ground_hits);
    dict_set(out, "clean_fights", log->clean_fights);
    dict_set(out, "altitude_kills", log->altitude_kills);
    dict_set(out, "recovery_triggers", log->recovery_triggers);
}
