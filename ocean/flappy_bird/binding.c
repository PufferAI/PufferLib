#include "flappy_bird.h"
#define OBS_SIZE (2 + 4 * MAX_PIPES)
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env FlappyBird
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    c_init(env);
    env->gravity = (float)dict_get(kwargs, "gravity")->value;
    env->pipe_speed = (int)dict_get(kwargs, "pipe_speed")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
