#include "go.h"
// 9x9 - obs 326, act 82
// 13x13 - obs 678, act 170
// 19x19 - obs 1446, act 362
#define OBS_SIZE 326
#define NUM_ATNS 1
#define ACT_SIZES {82}
#define OBS_TENSOR_T FloatTensor

#define Env CGo
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->side = (rand_r(&env->rng) % 2) + 1;
    env->selfplay = dict_get(kwargs, "selfplay")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->grid_size = dict_get(kwargs, "grid_size")->value;
    env->board_width = dict_get(kwargs, "board_width")->value;
    env->board_height = dict_get(kwargs, "board_height")->value;
    env->grid_square_size = dict_get(kwargs, "grid_square_size")->value;
    env->komi = dict_get(kwargs, "komi")->value;
    env->reward_move_pass = dict_get(kwargs, "reward_move_pass")->value;
    env->reward_move_invalid = dict_get(kwargs, "reward_move_invalid")->value;
    env->reward_move_valid = dict_get(kwargs, "reward_move_valid")->value;
    env->reward_player_capture = dict_get(kwargs, "reward_player_capture")->value;
    env->reward_opponent_capture = dict_get(kwargs, "reward_opponent_capture")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "n", log->n);
}
