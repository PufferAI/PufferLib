#include "four_rooms.h"

#define OBS_SIZE (7 * 7 * 3)
#define NUM_ATNS 1
#define ACT_SIZES {7}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_STEP four_rooms_vec_step
#define MY_VEC_STEP_RANGE four_rooms_vec_step_range
#define Env FourRooms
#include "vecenv.h"

void four_rooms_vec_step(StaticVec* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    FourRooms* envs = (FourRooms*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        c_step(&envs[i]);
    }
}

void four_rooms_vec_step_range(StaticVec* vec, int env_start, int env_count, int num_workers) {
    (void)num_workers;
    FourRooms* envs = (FourRooms*)vec->envs;
    for (int i = env_start; i < env_start + env_count; i++) {
        c_step(&envs[i]);
    }
}

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
