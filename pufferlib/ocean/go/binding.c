#include "go.h"
#define Env CGo
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->grid_size = unpack(kwargs, "grid_size");
    env->board_width = unpack(kwargs, "board_width");
    env->board_height = unpack(kwargs, "board_height");
    env->grid_square_size = unpack(kwargs, "grid_square_size");
    env->moves_made = unpack(kwargs, "moves_made");
    env->komi = unpack(kwargs, "komi");
    env->score = unpack(kwargs, "score");
    env->last_capture_position = unpack(kwargs, "last_capture_position");
    env->reward_move_pass = unpack(kwargs, "reward_move_pass");
    env->reward_move_invalid = unpack(kwargs, "reward_move_invalid");
    env->reward_move_valid = unpack(kwargs, "reward_move_valid");
    env->reward_player_capture = unpack(kwargs, "reward_player_capture");
    env->reward_opponent_capture = unpack(kwargs, "reward_opponent_capture");
    
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
