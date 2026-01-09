#include "backgammon.h"

#define Env CBackgammon
#include "../env_binding.h"

static int my_init(Env* env, PyObject *args, PyObject* kwargs) {
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "win_rate", log->win_rate);
    assign_to_dict(dict, "black_win_rate", log->black_win_rate);
    assign_to_dict(dict, "avg_moves_per_turn", log->avg_moves_per_turn);
    assign_to_dict(dict, "hit_rate", log->hit_rate);
    assign_to_dict(dict, "checkers_home", log->checkers_home);
    assign_to_dict(dict, "checkers_off", log->checkers_off);
    return 0;
}