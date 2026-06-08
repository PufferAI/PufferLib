#include "docking.h"

#define OBS_SIZE DOCKING_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T FloatTensor

#define Env Docking
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = (int)dict_get(kwargs, "width")->value;
    env->height = (int)dict_get(kwargs, "height")->value;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->max_speed = (float)dict_get(kwargs, "max_speed")->value;
    env->turn_rate = (float)dict_get(kwargs, "turn_rate")->value;
    env->accel = (float)dict_get(kwargs, "accel")->value;
    env->drag = (float)dict_get(kwargs, "drag")->value;
    env->dock_radius = (float)dict_get(kwargs, "dock_radius")->value;
    env->dock_speed_threshold = (float)dict_get(kwargs, "dock_speed_threshold")->value;
    env->dock_heading_threshold = (float)dict_get(kwargs, "dock_heading_threshold")->value;
    env->step_penalty = (float)dict_get(kwargs, "step_penalty")->value;
    env->progress_reward_scale = (float)dict_get(kwargs, "progress_reward_scale")->value;
    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "crash_rate", log->crash_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "final_distance", log->final_distance);
    dict_set(out, "alignment_error", log->alignment_error);
}
