#include "pathfinder.h"
#define OBS_SIZE PATHFINDER_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T FloatTensor

#define Env Pathfinder
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->branch_prob = (float)dict_get(kwargs, "branch_prob")->value;
    env->loop_prob = (float)dict_get(kwargs, "loop_prob")->value;
    env->step_penalty = (float)dict_get(kwargs, "step_penalty")->value;
    env->new_wall_penalty = (float)dict_get(kwargs, "new_wall_penalty")->value;
    env->known_wall_death_penalty = (float)dict_get(kwargs, "known_wall_death_penalty")->value;
    env->repeat_move_death_penalty = (float)dict_get(kwargs, "repeat_move_death_penalty")->value;
    env->new_cell_reward = (float)dict_get(kwargs, "new_cell_reward")->value;
    env->revisit_penalty = (float)dict_get(kwargs, "revisit_penalty")->value;
    env->impossible_penalty = (float)dict_get(kwargs, "impossible_penalty")->value;
    env->goal_reward = (float)dict_get(kwargs, "goal_reward")->value;
    env->start_solution_len = (int)dict_get(kwargs, "start_solution_len")->value;
    env->curriculum_enabled = (int)dict_get(kwargs, "curriculum_enabled")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "wins", log->wins);
    dict_set(out, "wall_hits", log->wall_hits);
    dict_set(out, "revisits", log->revisits);
    dict_set(out, "known_wall_deaths", log->known_wall_deaths);
    dict_set(out, "repeat_move_deaths", log->repeat_move_deaths);
    dict_set(out, "shortest_path_len", log->shortest_path_len);
    dict_set(out, "agent_path_len", log->agent_path_len);
    dict_set(out, "curriculum_level", log->curriculum_level);
    dict_set(out, "curriculum_target_len", log->curriculum_target_len);
    dict_set(out, "curriculum_next_target_len", log->curriculum_next_target_len);
}
