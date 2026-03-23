#include "blastar.h"
#define OBS_SIZE 10
#define NUM_ATNS 1
#define ACT_SIZES {6}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Blastar
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    int num_obs = dict_get(kwargs, "num_obs")->value;
    init(env, num_obs);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lives", log->lives);
    dict_set(out, "vertical_closeness_rew", log->vertical_closeness_rew);
    dict_set(out, "fired_bullet_rew", log->fired_bullet_rew);
    dict_set(out, "kill_streak", log->kill_streak);
    dict_set(out, "hit_enemy_with_bullet_rew", log->hit_enemy_with_bullet_rew);
    dict_set(out, "avg_score_difference", log->avg_score_difference);
}
