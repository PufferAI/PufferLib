#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "../env_binding.h"

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject* act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject* rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject* term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = num_agents;
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);
    // getting env count
    while(total_agent_count < num_agents && env_count < max_envs){
        char map_file[100];
        int map_id = rand() % num_maps;
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);
        // Store map_id
        PyObject* map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject* offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for(int j=0;j<env->num_entities;j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_car_indices);
        free(env->expert_static_car_indices);
        free(env);
    }
    if(total_agent_count >= num_agents){
        total_agent_count = num_agents;
    }
    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject* final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject* resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject* resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    //
    //Py_DECREF(agent_offsets);
    //Py_DECREF(map_ids);
    // create a tuple
    PyObject* tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;

    //Py_DECREF(num);
    /*
    for(int i = 0;i<num_envs; i++) {
        for(int j=0;j<temp_envs[i].num_entities;j++) {
            free_entity(&temp_envs[i].entities[j]);
        }
        free(temp_envs[i].entities);
        free(temp_envs[i].active_agent_indices);
        free(temp_envs[i].static_car_indices);
    }
    free(temp_envs);
    */
    // return agent_offsets;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = unpack(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = unpack(kwargs, "reward_offroad_collision");
    env->reward_goal_post_respawn = unpack(kwargs, "reward_goal_post_respawn");
    env->reward_vehicle_collision_post_respawn = unpack(kwargs, "reward_vehicle_collision_post_respawn");
    env->spawn_immunity_timer = unpack(kwargs, "spawn_immunity_timer");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");

    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "clean_collision_rate", log->clean_collision_rate);
    return 0;
}
