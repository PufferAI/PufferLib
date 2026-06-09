#include "bat.h"
#define OBS_SIZE 39
#define NUM_ATNS 6
#define ACT_SIZES {3, 3, 8, 8, 4, 2}
#define OBS_TENSOR_T FloatTensor

#define Env Bat
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_obstacles = dict_get(kwargs, "num_obstacles")->value;
    env->bat_radius = dict_get(kwargs, "bat_radius")->value;
    env->ear_separation_scale = dict_get(kwargs, "ear_separation_scale")->value;
    env->bug_radius = dict_get(kwargs, "bug_radius")->value;
    env->bat_max_speed = dict_get(kwargs, "bat_max_speed")->value;
    env->bat_accel = dict_get(kwargs, "bat_accel")->value;
    env->bat_turn_rate = dict_get(kwargs, "bat_turn_rate")->value;
    env->bug_speed = dict_get(kwargs, "bug_speed")->value;
    env->max_steps = dict_get(kwargs, "max_steps")->value;
    env->curriculum_enabled = dict_get(kwargs, "curriculum_enabled")->value;
    env->curriculum_initial_level = dict_get(kwargs, "curriculum_initial_level")->value;
    env->curriculum_start_obstacles = dict_get(kwargs, "curriculum_start_obstacles")->value;
    env->curriculum_max_obstacles = dict_get(kwargs, "curriculum_max_obstacles")->value;
    env->curriculum_obstacle_step = dict_get(kwargs, "curriculum_obstacle_step")->value;
    env->curriculum_successes_per_level = dict_get(kwargs, "curriculum_successes_per_level")->value;
    env->curriculum_start_bug_distance = dict_get(kwargs, "curriculum_start_bug_distance")->value;
    env->curriculum_max_bug_distance = dict_get(kwargs, "curriculum_max_bug_distance")->value;
    env->curriculum_bug_distance_step = dict_get(kwargs, "curriculum_bug_distance_step")->value;
    env->freq_bins_per_ear = dict_get(kwargs, "freq_bins_per_ear")->value;
    env->max_echo_range = dict_get(kwargs, "max_echo_range")->value;
    env->sound_speed = dict_get(kwargs, "sound_speed")->value;
    env->reflector_spacing = dict_get(kwargs, "reflector_spacing")->value;
    env->max_chirp_age_ticks = dict_get(kwargs, "max_chirp_age_ticks")->value;
    env->chirp_cooldown_ticks = dict_get(kwargs, "chirp_cooldown_ticks")->value;
    env->chirp_cost = dict_get(kwargs, "chirp_cost")->value;
    env->valid_chirp_reward = dict_get(kwargs, "valid_chirp_reward")->value;
    env->early_chirp_penalty = dict_get(kwargs, "early_chirp_penalty")->value;
    env->bug_echo_reward_scale = dict_get(kwargs, "bug_echo_reward_scale")->value;
    env->step_cost = dict_get(kwargs, "step_cost")->value;
    env->progress_reward_scale = dict_get(kwargs, "progress_reward_scale")->value;
    env->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "collision", log->collision);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "bug_distance_start", log->bug_distance_start);
    dict_set(out, "bug_distance_final", log->bug_distance_final);
    dict_set(out, "bug_distance_delta", log->bug_distance_delta);
    dict_set(out, "chirps_emitted", log->chirps_emitted);
    dict_set(out, "mean_chirp_duration", log->mean_chirp_duration);
    dict_set(out, "mean_chirp_bandwidth", log->mean_chirp_bandwidth);
    dict_set(out, "mean_echo_energy_left", log->mean_echo_energy_left);
    dict_set(out, "mean_echo_energy_right", log->mean_echo_energy_right);
}
