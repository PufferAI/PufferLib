#include "grixel.h"

#define Env Grixel
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {

    // This is called once at the start of each
    // experiment, generating a fixed set of maps,
    // from which we will choose randomly for each
    // episode

    int num_maps = unpack(kwargs, "num_maps");
    int max_size = unpack(kwargs, "max_size");
    int size = unpack(kwargs, "size");
    
    // These are needed because we're using init_grid, which relies on 
    // them being set
    int pixelize = unpack(kwargs, "pixelize");
    int block_size = unpack(kwargs, "block_size");

    State* levels = calloc(num_maps, sizeof(State));

    if (max_size <= 5) {
        PyErr_SetString(PyExc_ValueError, "max_size must be >5");
        return NULL;
    }

    // Temporary env used to gen maps
    Grixel env;
    env.max_size = max_size;
    env.pixelize = pixelize;
    env.block_size= block_size;
    env.additional_obs_size = unpack(kwargs, "additional_obs_size");
    env.nb_object_types = unpack(kwargs, "nb_object_types");

    // Hmmm... at that point, block_size and pixelize are not defined,
    // even though they're used in init_grid!
    init_grid(&env);  // This allocates env, with 1 agent and 1 max-size grid
    
    srand(time(NULL));
    int start_seed = rand();
    for (int i = 0; i < num_maps; i++) {
        int sz = size;
        if (size == -1) {
            int min = 9;
            //if (max_size / 2 > min)
            //    min = max_size/2;
            sz = min + (rand() % (max_size-min));
            //sz = 5 + (rand() % (max_size-5));
            //sz = max_size / 2;
        }

        if (sz % 2 == 0) {
            sz -= 1;
        }

        float difficulty = (float)rand()/(float)(RAND_MAX);
        create_maze_level(&env, sz, sz, difficulty, start_seed + i);
        init_state(&levels[i], max_size, 1); // allocates the grid, with num_agents=1
        get_state(&env, &levels[i]); // this copies from env to levels
        // if env and levels have different num_agents strange things might happen?
    }

    return PyLong_FromVoidPtr(levels);
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_size = unpack(kwargs, "max_size");
    env->num_maps = unpack(kwargs, "num_maps");
    env->texture_mode = unpack(kwargs, "texture_mode");
    env->pixelize = unpack(kwargs, "pixelize");
    env->additional_obs_size = unpack(kwargs, "additional_obs_size");
    env->nb_object_types = unpack(kwargs, "nb_object_types");
    env->block_size= unpack(kwargs, "block_size");
    
    init_grid(env); //requires block_size to be pre-set
    
    env->is_mobile[REWARD] = 1;
    env->is_mobile[ZOMBIE] = 1;
    env->is_pickable[OBJECT] = 1;

    PyObject* handle_obj = PyDict_GetItemString(kwargs, "state");
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "state handle must be an integer");
        return 1;
    }

    State* levels = (State*)PyLong_AsVoidPtr(handle_obj);
    if (!levels) {
        PyErr_SetString(PyExc_ValueError, "Invalid state handle");
        return 1;
    }

    env->levels = levels;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
