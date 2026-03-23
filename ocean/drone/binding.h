#include "drone.h"
#define OBS_SIZE 26
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env DroneEnv
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->max_rings = dict_get(kwargs, "max_rings")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collision", log->ring_collision);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
