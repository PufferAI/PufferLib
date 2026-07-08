#include "battle.h"

// Per-agent obs: 3 per army (delta to nearest base) + 4*AGENT_OBS nearest
// units + 22 self stats + 8 unit one-hot. Must match num_armies in
// config/battle.ini (checked in my_init).
#define NUM_ARMIES 2
#define OBS_SIZE (3*NUM_ARMIES + 4*AGENT_OBS + 22 + 8)
#define NUM_ATNS 3
#define ACT_SIZES {1, 1, 1} // continuous, was Box(-1, 1, (3,))
#define OBS_TENSOR_T FloatTensor

#define Env Battle
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->size_x = dict_get(kwargs, "size_x")->value;
    env->size_y = dict_get(kwargs, "size_y")->value;
    env->size_z = dict_get(kwargs, "size_z")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->num_armies = dict_get(kwargs, "num_armies")->value;
    if (env->num_armies != NUM_ARMIES) {
        fprintf(stderr, "battle: num_armies=%d does not match compiled NUM_ARMIES=%d\n",
            env->num_armies, NUM_ARMIES);
        exit(1);
    }
    // Learner agents plus an equal number of scripted opponents
    // (the 3.x wrapper passed num_agents*2 to the C env).
    env->num_entities = 2*env->num_agents;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "oob_rate", log->oob_rate);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
