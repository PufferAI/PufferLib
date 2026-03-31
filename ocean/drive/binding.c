#include "drive.h"
#define OBS_SIZE 1848
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TENSOR_T FloatTensor

#define MAP_BINARY_DIR "resources/drive/binaries/training"

#define MY_VEC_INIT
#define Env Drive
#include "vecenv.h"

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;

    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision")->value;
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision")->value;
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn")->value;
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn")->value;
    int spawn_immunity_timer = (int)dict_get(env_kwargs, "spawn_immunity_timer")->value;
    int human_agent_idx = (int)dict_get(env_kwargs, "human_agent_idx")->value;
    int render_mode = (int)dict_get(env_kwargs, "render_mode")->value;

    // Verify first map exists
    char first_map[512];
    snprintf(first_map, sizeof(first_map), "%s/map_%03d.bin", MAP_BINARY_DIR, 0);
    FILE* test_fp = fopen(first_map, "rb");
    if (!test_fp) {
        printf("ERROR: Cannot find map files at %s/\n", MAP_BINARY_DIR);
        *num_envs_out = 0;
        return NULL;
    }
    fclose(test_fp);

    // Check the number of controllable agents per map
    int agents_per_map[num_maps];
    for (int m = 0; m < num_maps; m++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, m);
        Env temp_env = {0};
        temp_env.map_name = map_file;
        temp_env.num_agents = 0;
        init(&temp_env);
        agents_per_map[m] = temp_env.active_agent_count < MAX_CARS
                          ? temp_env.active_agent_count : MAX_CARS;
        c_close(&temp_env);
        //printf("  map_%03d.bin: %d agents\n", m, agents_per_map[m]);
    }
    printf("Scanned %d maps from %s/\n", num_maps, MAP_BINARY_DIR);

    int agents_per_buffer = total_agents / num_buffers;
    int envs_per_buffer = 0;
    int agents_in_buffer = 0;
    while (agents_in_buffer < agents_per_buffer) {
        int m = envs_per_buffer % num_maps;
        agents_in_buffer += agents_per_map[m];
        envs_per_buffer++;
    }

    // How many excess agents are in the last map of each buffer?
    int excess = agents_in_buffer - agents_per_buffer;
    int last_map_idx = (envs_per_buffer - 1) % num_maps;
    int last_map_capped_agents = agents_per_map[last_map_idx] - excess;

    int total_envs = envs_per_buffer * num_buffers;

    // Fill buffer info
    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = b * envs_per_buffer;
        buffer_env_counts[b] = envs_per_buffer;
    }

    Env* envs = (Env*)calloc(total_envs, sizeof(Env));
    int actual_total_agents = 0;

    for (int i = 0; i < total_envs; i++) {
        int local_idx = i % envs_per_buffer;
        int m = local_idx % num_maps;
        int is_last_in_buffer = (local_idx == envs_per_buffer - 1);

        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, m);

        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->spawn_immunity_timer = spawn_immunity_timer;
        env->render_mode = render_mode;
        env->num_agents = is_last_in_buffer ? last_map_capped_agents : agents_per_map[m];

        init(env);
        actual_total_agents += env->active_agent_count;
    }

    printf("Created %d envs (%d per buffer x %d buffers), %d agents per buffer, %d total agents (target %d)\n",
           total_envs, envs_per_buffer, num_buffers, agents_in_buffer, actual_total_agents, total_agents);

    *num_envs_out = total_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx")->value;
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision")->value;
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision")->value;
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn")->value;
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn")->value;
    env->spawn_immunity_timer = dict_get(kwargs, "spawn_immunity_timer")->value;
    env->render_mode = (int)dict_get(kwargs, "render_mode")->value;
    int map_id = dict_get(kwargs, "map_id")->value;
    int max_agents = dict_get(kwargs, "max_agents")->value;

    char map_file[512];
    sprintf(map_file, "%s/map_%03d.bin", MAP_BINARY_DIR, map_id);
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