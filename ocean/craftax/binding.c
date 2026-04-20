#define CRAFTAX_ENABLE_ENV_IMPL
#include "craftax.h"
#include "step_crafting.h"
#include "step_update_mobs.h"
#include "step_spawn_mobs.h"

#define OBS_SIZE CRAFTAX_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CRAFTAX_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env Craftax
#include "vecenv.h"

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
