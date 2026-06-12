#include "bat.h"
#define NUM_ATNS NUM_ACTIONS
#define ACT_SIZES {MOVE_ACTIONS, TURN_ACTIONS, CHIRP_FREQ_BINS, CHIRP_FREQ_BINS, CHIRP_DURATION_BINS, CHIRP_EMIT_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env Bat
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = NUM_AGENTS;
    env->ear_separation_scale = dict_get(kwargs, "ear_separation_scale")->value;
    env->ear_rear_gain = dict_get(kwargs, "ear_rear_gain")->value;
    env->ear_front_gain = dict_get(kwargs, "ear_front_gain")->value;
    env->ear_side_gain = dict_get(kwargs, "ear_side_gain")->value;
    env->max_speed = dict_get(kwargs, "max_speed")->value;
    env->min_speed = dict_get(kwargs, "min_speed")->value;
    env->accel = dict_get(kwargs, "accel")->value;
    env->turn_rate = dict_get(kwargs, "turn_rate")->value;
    env->render_target_fps = dict_get(kwargs, "render_target_fps")->value;
    env->record_video = dict_get(kwargs, "record_video")->value;
    env->record_video_fps = dict_get(kwargs, "record_video_fps")->value;
    env->record_video_seconds = dict_get(kwargs, "record_video_seconds")->value;
    env->record_video_audio = dict_get(kwargs, "record_video_audio")->value;
    env->curriculum_initial_level = dict_get(kwargs, "curriculum_initial_level")->value;
    env->curriculum_obstacle_step = dict_get(kwargs, "curriculum_obstacle_step")->value;
    env->curriculum_successes_per_level = dict_get(kwargs, "curriculum_successes_per_level")->value;
    env->curriculum_start_bug_distance = dict_get(kwargs, "curriculum_start_bug_distance")->value;
    env->sound_speed = dict_get(kwargs, "sound_speed")->value;
    env->reflector_strength = dict_get(kwargs, "reflector_strength")->value;
    env->chirp_cooldown_ticks = dict_get(kwargs, "chirp_cooldown_ticks")->value;
    env->chirp_efficiency_reward = dict_get(kwargs, "chirp_efficiency_reward")->value;
    env->valid_chirp_reward = dict_get(kwargs, "valid_chirp_reward")->value;
    env->early_chirp_penalty = dict_get(kwargs, "early_chirp_penalty")->value;
    env->chirp_overlap_penalty = dict_get(kwargs, "chirp_overlap_penalty")->value;
    env->bug_echo_reward_scale = dict_get(kwargs, "bug_echo_reward_scale")->value;
    env->bug_echo_farther_penalty_scale = dict_get(kwargs, "bug_echo_farther_penalty_scale")->value;
    env->bug_wing_sideband_gain = dict_get(kwargs, "bug_wing_sideband_gain")->value;
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
    dict_set(out, "base_perf", log->base_perf);
    dict_set(out, "collision", log->collision);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "curriculum_level", log->curriculum_level);
    dict_set(out, "curriculum_difficulty", log->curriculum_difficulty);
    dict_set(out, "curriculum_perf", log->curriculum_perf);
    dict_set(out, "curriculum_distance_difficulty", log->curriculum_distance_difficulty);
    dict_set(out, "curriculum_obstacle_difficulty", log->curriculum_obstacle_difficulty);
    dict_set(out, "curriculum_motion_difficulty", log->curriculum_motion_difficulty);
    dict_set(out, "num_obstacles", log->num_obstacles);
    dict_set(out, "chirps_emitted", log->chirps_emitted);
    dict_set(out, "chirp_perf", log->chirp_perf);
    dict_set(out, "chirp_overlap_fraction", log->chirp_overlap_fraction);
}
