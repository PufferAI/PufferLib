#define CRAFTAX_ENABLE_ENV_IMPL
#include "craftax.h"
#include "step_crafting.h"
#include "step_update_mobs.h"
#include "step_spawn_mobs.h"

#define OBS_SIZE CRAFTAX_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CRAFTAX_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define CRAFTAX_VEC_TILE_SIZE 128
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define MY_VEC_STEP craftax_vec_step
#define MY_VEC_STEP_RANGE craftax_vec_step_range
#define Env Craftax
#include "vecenv.h"

// Tiled vector step: process agents in tiles that fit comfortably in cache.
// Each thread processes a contiguous block of lightweight env handles while
// the heavier CraftaxState storage lives in a separate arena.
void craftax_vec_step(StaticVec* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    Craftax* envs = (Craftax*)vec->envs;
    int size = vec->size;
    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < size; tile += CRAFTAX_VEC_TILE_SIZE) {
        int end = tile + CRAFTAX_VEC_TILE_SIZE;
        if (end > size) end = size;
        for (int i = tile; i < end; i++) {
            c_step_gameplay(&envs[i]);
            c_step_encode(&envs[i]);
        }
    }
}

void craftax_vec_step_range(StaticVec* vec, int env_start, int env_count, int num_workers) {
    (void)num_workers;
    Craftax* envs = (Craftax*)vec->envs;
    int env_end = env_start + env_count;
    for (int tile = env_start; tile < env_end; tile += CRAFTAX_VEC_TILE_SIZE) {
        int end = tile + CRAFTAX_VEC_TILE_SIZE;
        if (end > env_end) end = env_end;
        for (int i = tile; i < end; i++) {
            c_step_gameplay(&envs[i]);
            c_step_encode(&envs[i]);
        }
    }
}

static CraftaxState* craftax_alloc_state_arena(int num_envs) {
    return (CraftaxState*)calloc((size_t)num_envs, sizeof(CraftaxState));
}

Env* my_vec_init(
    int* num_envs_out,
    int* buffer_env_starts,
    int* buffer_env_counts,
    Dict* vec_kwargs,
    Dict* env_kwargs
) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int num_envs = total_agents;

    Env* envs = (Env*)calloc((size_t)num_envs, sizeof(Env));
    CraftaxArena* arena = (CraftaxArena*)calloc(1, sizeof(CraftaxArena));
    arena->states = craftax_alloc_state_arena(num_envs);
    arena->num_envs = num_envs;
    arena->packet_size = CRAFTAX_ARENA_PACKET_SIZE;
    arena->num_packets = (num_envs + CRAFTAX_ARENA_PACKET_SIZE - 1)
        / CRAFTAX_ARENA_PACKET_SIZE;

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->rng = (unsigned int)i;
        env->arena = arena;
        env->state = &arena->states[i];
        env->packet_id = i / arena->packet_size;
        env->lane_id = i % arena->packet_size;
        env->owns_state_storage = false;
        my_init(env, env_kwargs);

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
    if (envs == NULL || envs[0].arena == NULL) {
        return;
    }

    CraftaxArena* arena = envs[0].arena;
    free(arena->states);
    free(arena);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;

    uint64_t seed_offset = 0;
    DictItem* item = dict_get_unsafe(kwargs, "seed_offset");
    if (item != NULL) {
        seed_offset = (uint64_t)item->value;
    }
    env->seed = seed_offset + (uint64_t)env->rng;

    // Process-wide reset pool (first caller wins, rest block until ready).
    // 0 disables caching -- regenerate every reset (exact parity mode).
    int reset_pool_size = 0;
    DictItem* pool_item = dict_get_unsafe(kwargs, "reset_pool_size");
    if (pool_item != NULL) reset_pool_size = (int)pool_item->value;
    craftax_set_reset_pool_size(reset_pool_size);

    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    // Log 8 checkpoint achievements that form the tech / exploration curve.
    // perf (above) already aggregates all 67 into a normalized score; the
    // individual lines here are the milestones worth watching on a dashboard.
    // The env still tracks all 67 internally for reward and perf; we just
    // don't send every one through the log Dict.
    struct { const char* name; int idx; } checkpoints[] = {
        {"collect_wood",         0},
        {"make_wood_pickaxe",    5},
        {"make_stone_pickaxe",  13},
        {"collect_iron",        18},
        {"make_iron_pickaxe",   20},
        {"collect_diamond",     19},
        {"enter_gnomish_mines", 28},
        {"defeat_necromancer",  48},
    };
    for (int i = 0; i < (int)(sizeof(checkpoints) / sizeof(checkpoints[0])); i++) {
        dict_set(out, checkpoints[i].name, log->achievements[checkpoints[i].idx]);
    }
}
