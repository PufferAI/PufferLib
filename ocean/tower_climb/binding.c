#include "tower_climb.h"
#define OBS_SIZE 228
#define NUM_ATNS 1
#define ACT_SIZES {6}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_INIT
#define Env CTowerClimb
#include "vecenv.h"

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int num_envs = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;

    float reward_climb_row = dict_get(env_kwargs, "reward_climb_row")->value;
    float reward_fall_row = dict_get(env_kwargs, "reward_fall_row")->value;
    float reward_illegal_move = dict_get(env_kwargs, "reward_illegal_move")->value;
    float reward_move_block = dict_get(env_kwargs, "reward_move_block")->value;

    const char* path = "resources/tower_climb/maps.bin";
    int num_maps = 0;

    Level* levels = load_levels_from_file(&num_maps, path);
    if (levels == NULL) {
        *num_envs_out = 0;
        return NULL;
    }

    PuzzleState* puzzle_states = calloc(num_maps, sizeof(PuzzleState));
    for (int i = 0; i < num_maps; i++) {
        init_puzzle_state(&puzzle_states[i]);
        levelToPuzzleState(&levels[i], &puzzle_states[i]);
    }

    Env* envs = (Env*)calloc(num_envs, sizeof(Env));

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->rng = i;
        env->num_agents = 1;
        env->reward_climb_row = reward_climb_row;
        env->reward_fall_row = reward_fall_row;
        env->reward_illegal_move = reward_illegal_move;
        env->reward_move_block = reward_move_block;
        env->all_levels = levels;
        env->all_puzzles = puzzle_states;
        env->num_maps = num_maps;
        init(env);
    }

    int agents_per_buffer = num_envs / num_buffers;
    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->reward_climb_row = dict_get(kwargs, "reward_climb_row")->value;
    env->reward_fall_row = dict_get(kwargs, "reward_fall_row")->value;
    env->reward_illegal_move = dict_get(kwargs, "reward_illegal_move")->value;
    env->reward_move_block = dict_get(kwargs, "reward_move_block")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
