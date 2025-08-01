#include "four_rooms.h"

#define Env FourRooms
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = unpack(kwargs, "size");
    env->see_through_walls = 0;
    // Allocate grid memory for full state (stores OBJECT_IDX values)
    env->grid = (unsigned char*)calloc(env->size * env->size, sizeof(unsigned char));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
