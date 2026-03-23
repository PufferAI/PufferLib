#include "tetris.h"
#define OBS_SIZE 234
#define NUM_ATNS 1
#define ACT_SIZES {7}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Tetris
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->n_cols = dict_get(kwargs, "n_cols")->value;
    env->n_rows = dict_get(kwargs, "n_rows")->value;
    env->use_deck_obs = dict_get(kwargs, "use_deck_obs")->value;
    env->n_noise_obs = dict_get(kwargs, "n_noise_obs")->value;
    env->n_init_garbage = dict_get(kwargs, "n_init_garbage")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lines_deleted", log->lines_deleted);
    dict_set(out, "avg_combo", log->avg_combo);
    dict_set(out, "atn_frac_soft_drop", log->atn_frac_soft_drop);
    dict_set(out, "atn_frac_hard_drop", log->atn_frac_hard_drop);
    dict_set(out, "atn_frac_rotate", log->atn_frac_rotate);
    dict_set(out, "atn_frac_hold", log->atn_frac_hold);
    dict_set(out, "game_level", log->game_level);
    dict_set(out, "ticks_per_line", log->ticks_per_line);
}
