#include "moba.h"
#define OBS_SIZE 510
#define NUM_ATNS 6
#define ACT_SIZES {7, 7, 3, 2, 2, 2}
#define OBS_TENSOR_T ByteTensor

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define Env MOBA
#include "vecenv.h"

void my_vec_close(Env* envs) {
    free(envs[0].ai_paths);
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int num_envs = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;

    int vision_range = (int)dict_get(env_kwargs, "vision_range")->value;
    float agent_speed = dict_get(env_kwargs, "agent_speed")->value;
    float reward_death = dict_get(env_kwargs, "reward_death")->value;
    float reward_xp = dict_get(env_kwargs, "reward_xp")->value;
    float reward_distance = dict_get(env_kwargs, "reward_distance")->value;
    float reward_tower = dict_get(env_kwargs, "reward_tower")->value;
    int script_opponents = (int)dict_get(env_kwargs, "script_opponents")->value;



    // ai_paths (256 MB) is shared — same map, so BFS results are identical across envs.
    // ai_path_buffer (1.5 MB) must be per-env: bfs() uses it as a scratch queue
    // starting from index 0 on every call, so concurrent BFS calls corrupt each other.
    unsigned char* ai_paths = calloc(128*128*128*128, sizeof(unsigned char));
    for (int i = 0; i < 128*128*128*128; i++) {
        ai_paths[i] = 255;
    }

    // Calculate agents per env based on script_opponents
    int agents_per_env = script_opponents ? 5 : 10;
    int total_envs = num_envs / agents_per_env;

    Env* envs = (Env*)calloc(total_envs, sizeof(Env));

    for (int i = 0; i < total_envs; i++) {
        Env* env = &envs[i];
        env->num_agents = agents_per_env;
        env->vision_range = vision_range;
        env->agent_speed = agent_speed;
        env->reward_death = reward_death;
        env->reward_xp = reward_xp;
        env->reward_distance = reward_distance;
        env->reward_tower = reward_tower;
        env->script_opponents = script_opponents;
        env->ai_path_buffer = calloc(3*8*128*128, sizeof(int));
        env->ai_paths = ai_paths;
        init_moba(env, game_map_npy);
    }

    int agents_per_buffer = num_envs / num_buffers;
    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < total_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = total_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->vision_range = dict_get(kwargs, "vision_range")->value;
    env->agent_speed = dict_get(kwargs, "agent_speed")->value;
    env->reward_death = dict_get(kwargs, "reward_death")->value;
    env->reward_xp = dict_get(kwargs, "reward_xp")->value;
    env->reward_distance = dict_get(kwargs, "reward_distance")->value;
    env->reward_tower = dict_get(kwargs, "reward_tower")->value;
    env->script_opponents = dict_get(kwargs, "script_opponents")->value;
    env->num_agents = env->script_opponents ? 5 : 10;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "radiant_victory", log->radiant_victory);
    dict_set(out, "dire_victory", log->dire_victory);
    dict_set(out, "radiant_level", log->radiant_level);
    dict_set(out, "dire_level", log->dire_level);
    dict_set(out, "radiant_towers_alive", log->radiant_towers_alive);
    dict_set(out, "dire_towers_alive", log->dire_towers_alive);
    dict_set(out, "radiant_support_episode_return", log->radiant_support_episode_return);
    dict_set(out, "radiant_support_reward_death", log->radiant_support_reward_death);
    dict_set(out, "radiant_support_reward_xp", log->radiant_support_reward_xp);
    dict_set(out, "radiant_support_reward_distance", log->radiant_support_reward_distance);
    dict_set(out, "radiant_support_reward_tower", log->radiant_support_reward_tower);
    dict_set(out, "radiant_support_level", log->radiant_support_level);
    dict_set(out, "radiant_support_kills", log->radiant_support_kills);
    dict_set(out, "radiant_support_deaths", log->radiant_support_deaths);
    dict_set(out, "radiant_support_damage_dealt", log->radiant_support_damage_dealt);
    dict_set(out, "radiant_support_damage_received", log->radiant_support_damage_received);
    dict_set(out, "radiant_support_healing_dealt", log->radiant_support_healing_dealt);
    dict_set(out, "radiant_support_healing_received", log->radiant_support_healing_received);
    dict_set(out, "radiant_support_creeps_killed", log->radiant_support_creeps_killed);
    dict_set(out, "radiant_support_neutrals_killed", log->radiant_support_neutrals_killed);
    dict_set(out, "radiant_support_towers_killed", log->radiant_support_towers_killed);
    dict_set(out, "radiant_support_usage_auto", log->radiant_support_usage_auto);
    dict_set(out, "radiant_support_usage_q", log->radiant_support_usage_q);
    dict_set(out, "radiant_support_usage_w", log->radiant_support_usage_w);
    dict_set(out, "radiant_support_usage_e", log->radiant_support_usage_e);
}
