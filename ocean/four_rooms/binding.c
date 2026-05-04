#include "four_rooms.h"

#define OBS_SIZE (7 * 7 * 3)
#define NUM_ATNS 1
#define ACT_SIZES {7}
#define OBS_TENSOR_T ByteTensor

#define Env FourRooms
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = (int)dict_get(kwargs, "size")->value;
    env->see_through_walls = 0;
    env->grid = (unsigned char*)calloc(env->size * env->size, sizeof(unsigned char));
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
