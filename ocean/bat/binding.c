#include "bat.h"
#define OBS_SIZE 40
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
    env->bat_min_speed = dict_get(kwargs, "bat_min_speed")->value;
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
    env->max_chirps_per_episode = dict_get(kwargs, "max_chirps_per_episode")->value;
    env->min_chirps_per_episode = dict_get(kwargs, "min_chirps_per_episode")->value;
    env->chirp_budget_decay_levels = dict_get(kwargs, "chirp_budget_decay_levels")->value;
    env->chirp_cost = dict_get(kwargs, "chirp_cost")->value;
    env->chirp_efficiency_reward = dict_get(kwargs, "chirp_efficiency_reward")->value;
    env->valid_chirp_reward = dict_get(kwargs, "valid_chirp_reward")->value;
    env->early_chirp_penalty = dict_get(kwargs, "early_chirp_penalty")->value;
    env->chirp_overlap_penalty = dict_get(kwargs, "chirp_overlap_penalty")->value;
    env->bug_echo_reward_scale = dict_get(kwargs, "bug_echo_reward_scale")->value;
    env->bug_echo_farther_penalty_scale = dict_get(kwargs, "bug_echo_farther_penalty_scale")->value;
    env->bug_echo_min_displacement = dict_get(kwargs, "bug_echo_min_displacement")->value;
    env->step_cost = dict_get(kwargs, "step_cost")->value;
    env->progress_reward_scale = dict_get(kwargs, "progress_reward_scale")->value;
    env->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "base_perf", log->base_perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "collision", log->collision);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "curriculum_level", log->curriculum_level);
    dict_set(out, "curriculum_difficulty", log->curriculum_difficulty);
    dict_set(out, "curriculum_perf", log->curriculum_perf);
    dict_set(out, "curriculum_distance_difficulty", log->curriculum_distance_difficulty);
    dict_set(out, "curriculum_obstacle_difficulty", log->curriculum_obstacle_difficulty);
    dict_set(out, "curriculum_chirp_budget_difficulty", log->curriculum_chirp_budget_difficulty);
    dict_set(out, "num_obstacles", log->num_obstacles);
    dict_set(out, "bug_distance_start", log->bug_distance_start);
    dict_set(out, "bug_distance_final", log->bug_distance_final);
    dict_set(out, "bug_distance_delta", log->bug_distance_delta);
    dict_set(out, "chirps_emitted", log->chirps_emitted);
    dict_set(out, "chirp_budget", log->chirp_budget);
    dict_set(out, "chirps_used_ratio", log->chirps_used_ratio);
    dict_set(out, "chirp_efficiency", log->chirp_efficiency);
    dict_set(out, "chirp_perf", log->chirp_perf);
    dict_set(out, "chirp_overlap_fraction", log->chirp_overlap_fraction);
    dict_set(out, "far_chirp_rate", log->far_chirp_rate);
    dict_set(out, "near_chirp_rate", log->near_chirp_rate);
    dict_set(out, "chirp_tempo_ratio", log->chirp_tempo_ratio);
    dict_set(out, "first_chirp_tick_norm", log->first_chirp_tick_norm);
    dict_set(out, "mean_chirp_tick_norm", log->mean_chirp_tick_norm);
    dict_set(out, "mean_chirp_duration", log->mean_chirp_duration);
    dict_set(out, "mean_chirp_bandwidth", log->mean_chirp_bandwidth);
    dict_set(out, "mean_echo_energy_left", log->mean_echo_energy_left);
    dict_set(out, "mean_echo_energy_right", log->mean_echo_energy_right);
}
