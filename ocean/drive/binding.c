#include "drive.h"
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TENSOR_T FloatTensor

#define MAP_BINARY_DIR "drive_data/binaries"

#define MY_VEC_INIT
#define Env Drive
#include "vecenv.h"

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;
    int agents_per_buffer = total_agents / num_buffers;

    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision")->value;
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision")->value;
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn")->value;
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn")->value;
    int human_agent_idx = (int)dict_get(env_kwargs, "human_agent_idx")->value;

    // Verify that the path has valid binaries
    char first_map[512];
    snprintf(first_map, sizeof(first_map), "%s/map_%03d.bin", MAP_BINARY_DIR, 0);
    FILE* test_fp = fopen(first_map, "rb");
    if (!test_fp) {
        printf("ERROR: Cannot find map files at %s/\n", MAP_BINARY_DIR);
        *num_envs_out = 0;
        return NULL;
    }
    fclose(test_fp);

    // Scan all maps for agent counts; collect valid (>0) ones
    int agents_per_map[num_maps];
    int valid_map_ids[num_maps];
    int num_valid_maps = 0;
    for (int m = 0; m < num_maps; m++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, m);
        Env temp_env = {0};
        temp_env.map_name = map_file;
        init(&temp_env);
        agents_per_map[m] = temp_env.active_agent_count < MAX_AGENTS
                          ? temp_env.active_agent_count : MAX_AGENTS;
        c_close(&temp_env);
        if (agents_per_map[m] > 0) {
            valid_map_ids[num_valid_maps++] = m;
        }
    }
    printf("Scanned %d maps from %s/, %d valid\n", num_maps, MAP_BINARY_DIR, num_valid_maps);

    if (num_valid_maps == 0) {
        printf("ERROR: No valid maps found\n");
        *num_envs_out = 0;
        return NULL;
    }

    // Build per-env layout. Each buffer advances the global cursor so different
    // buffers get different maps. If the next full map would overflow a buffer,
    // pack remaining slots with 1-agent envs advancing the cursor each time.
    int max_envs = agents_per_buffer * num_buffers; // upper bound (all 1-agent)
    int* env_map_ids = (int*)malloc(max_envs * sizeof(int));
    int* env_max_agents = (int*)malloc(max_envs * sizeof(int));
    int total_envs = 0;
    int cursor = 0; // advances across buffers

    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int m = valid_map_ids[cursor % num_valid_maps];
            int cap = agents_per_map[m];
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                // Full map fits
                env_map_ids[total_envs] = m;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
                cursor++;
            } else {
                // Pack remaining slots as 1-agent envs, one map each
                while (buffer_agents < agents_per_buffer) {
                    int mm = valid_map_ids[cursor % num_valid_maps];
                    env_map_ids[total_envs] = mm;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                    cursor++;
                }
            }
        }
        buffer_env_counts[b] = total_envs - buffer_env_starts[b];
    }

    printf("total envs: %d (%d maps cycled)\n", total_envs, cursor);

    // Initialize all envs
    Env* envs = (Env*)calloc(total_envs, sizeof(Env));
    for (int i = 0; i < total_envs; i++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, env_map_ids[i]);
        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->max_agents = env_max_agents[i];
        init(env);
        env->num_agents = env->active_agent_count;
    }

    free(env_map_ids);
    free(env_max_agents);

    printf("Created %d envs, %d total agents (target %d)\n",
           total_envs, total_agents, total_agents);

    *num_envs_out = total_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx")->value;
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision")->value;
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision")->value;
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn")->value;
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn")->value;
    int map_id = dict_get(kwargs, "map_id")->value;
    int max_agents = dict_get(kwargs, "max_agents")->value;

    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "dnf_rate", log->dnf_rate);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
    dict_set(out, "clean_collision_rate", log->clean_collision_rate);
}
