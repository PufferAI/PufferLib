#include "lightsout.h"

#define GRID_SIZE 5
#define OBS_SIZE (GRID_SIZE * GRID_SIZE)
#define NUM_ATNS 1
#define ACT_SIZES {GRID_SIZE * GRID_SIZE}
#define OBS_TENSOR_T ByteTensor

#define Env LightsOut
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->grid_size = GRID_SIZE;
    env->cell_size = 1280 / GRID_SIZE;
    if (1280 % GRID_SIZE != 0) env->cell_size++; // ceil
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->observation_size = OBS_SIZE;
    env->num_agents = 1;

    env->ema = 0.5f;
    env->score_ema = 0.0f;
    env->scramble_prob = 0.15f;

    init_lightsout(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "scramble_p", log->scramble_p);
}
