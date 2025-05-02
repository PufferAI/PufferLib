#include "blastar.h"
#define Env Blastar
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_obs = unpack(kwargs, "num_obs");
    init(env, env->num_obs);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "lives", log->lives);
    assign_to_dict(dict, "vertical_closeness_rew", log->vertical_closeness_rew);
    assign_to_dict(dict, "fired_bullet_rew", log->fired_bullet_rew);
    assign_to_dict(dict, "kill_streak", log->kill_streak);
    assign_to_dict(dict, "hit_enemy_with_bullet_rew", log->hit_enemy_with_bullet_rew);
    assign_to_dict(dict, "avg_score_difference", log->avg_score_difference);
    return 0;
}
