#include "table_hockey.h"
#define Env TableHockey
#define HAS_TRUNCATIONS
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->render_mode = (RenderMode)(int)unpack(kwargs, "render_mode");
    env->action_mode = (ActionMode)(int)unpack(kwargs, "action_mode");
    env->dt = unpack(kwargs, "dt");
    env->max_paddle_speed = unpack(kwargs, "max_paddle_speed");
    env->puck_hit_reward = unpack(kwargs, "puck_hit_reward");
    env->goal_reward = unpack(kwargs, "goal_reward");
    
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "puck_hits_per_episode", log->puck_hits_per_episode);
    assign_to_dict(dict, "player_goals", log->player_goals);
    assign_to_dict(dict, "opponent_goals", log->opponent_goals);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "n", log->n);
    return 0;
}