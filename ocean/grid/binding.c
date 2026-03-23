#include "grid.h"
#define OBS_SIZE 121
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor
#define ACT_TYPE DOUBLE

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define Env Grid
#include "vecenv.h"

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int num_envs = total_agents;

    int max_size = (int)dict_get(env_kwargs, "max_size")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;
    int map_size = (int)dict_get(env_kwargs, "map_size")->value;

    if (max_size <= 5) {
        *num_envs_out = 0;
        return NULL;
    }

    // Generate maze levels (shared across all envs)
    State* levels = calloc(num_maps, sizeof(State));

    // Temporary env used to generate maps
    Grid temp_env;
    temp_env.max_size = max_size;
    init_grid(&temp_env);

    unsigned int map_rng = 42;
    for (int i = 0; i < num_maps; i++) {
        int sz = map_size;
        if (map_size == -1) {
            sz = 5 + (rand_r(&map_rng) % (max_size - 5));
        }

        if (sz % 2 == 0) {
            sz -= 1;
        }

        float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
        create_maze_level(&temp_env, sz, sz, difficulty, i);
        init_state(&levels[i], max_size, 1);
        get_state(&temp_env, &levels[i]);
    }

    // Free temp env internal allocations
    free(temp_env.grid);
    free(temp_env.counts);
    free(temp_env.agents);

    // Allocate all environments
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->rng = i;
        env->max_size = max_size;
        env->num_maps = num_maps;
        env->num_agents = 1;
        env->levels = levels;
        init_grid(env);

        buf_agents += env->num_agents;
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

void my_vec_close(Env* envs) {
    free(envs[0].levels);
}

void my_init(Env* env, Dict* kwargs) {
    env->max_size = (int)dict_get(kwargs, "max_size")->value;
    env->num_maps = (int)dict_get(kwargs, "num_maps")->value;
    env->num_agents = 1;
    init_grid(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
