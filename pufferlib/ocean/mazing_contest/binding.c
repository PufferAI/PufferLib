#include "mazing_contest.h"

#define Env MazingContest
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    int width = unpack(kwargs, "width");
    int height = unpack(kwargs, "height");
    int cell_size = unpack(kwargs, "cell_size");
    env->cell_size_render = (float)cell_size;
    
    allocate(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "best_time", log->best_time);
    assign_to_dict(dict, "path_length_rewards", log->path_length_rewards);
    assign_to_dict(dict, "wall_touch_rewards", log->wall_touch_rewards);
    assign_to_dict(dict, "thunderclap_rewards", log->thunderclap_rewards);
    assign_to_dict(dict, "n", log->n);
    return 0;
}