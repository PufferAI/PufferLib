#include "boids.h"

// Each boid observes (dx, dy, dvx, dvy) of every boid.
// Must match num_boids in config/boids.ini (checked in my_init).
#define NUM_BOIDS 64
#define OBS_SIZE (4*NUM_BOIDS)
#define NUM_ATNS 2
#define ACT_SIZES {5, 5}
#define OBS_TENSOR_T FloatTensor

#define Env Boids
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_boids = dict_get(kwargs, "num_boids")->value;
    if (4*env->num_boids != OBS_SIZE) {
        fprintf(stderr, "boids: num_boids=%u does not match compiled NUM_BOIDS=%d\n",
            env->num_boids, NUM_BOIDS);
        exit(1);
    }
    env->num_agents = env->num_boids;
    env->report_interval = dict_get(kwargs, "report_interval")->value;
    env->margin_turn_factor = dict_get(kwargs, "margin_turn_factor")->value;
    env->centering_factor = dict_get(kwargs, "centering_factor")->value;
    env->avoid_factor = dict_get(kwargs, "avoid_factor")->value;
    env->matching_factor = dict_get(kwargs, "matching_factor")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
