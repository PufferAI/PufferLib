#include "pathfinder.h"
#define OBS_SIZE PATHFINDER_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T FloatTensor

#define Env Pathfinder
static inline void puffer_state_refresh(Pathfinder* env) { refresh_state(env); }
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->branch_prob = (float)dict_get(kwargs, "branch_prob")->value;
    env->loop_prob = (float)dict_get(kwargs, "loop_prob")->value;
    env->extra_entry_prob = (float)dict_get(kwargs, "extra_entry_prob")->value;
    env->min_solution_len = (int)dict_get(kwargs, "min_solution_len")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "wall_hits", log->wall_hits);
    dict_set(out, "known_walls", log->known_walls);
    dict_set(out, "known_open_edges", log->known_open_edges);
    dict_set(out, "shortest_path_len", log->shortest_path_len);
    dict_set(out, "agent_path_len", log->agent_path_len);
}
