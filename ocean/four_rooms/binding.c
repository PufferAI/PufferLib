#include "four_rooms.h"

#define OBS_SIZE (FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_OBS_CHANNELS)
#define NUM_ATNS 1
#define ACT_SIZES {FOUR_ROOMS_NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env FourRooms
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = (int)dict_get(kwargs, "size")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
