#include "craftax_classic.h"

#define OBS_SIZE 1345
#define NUM_ATNS 1
#define ACT_SIZES {17}
#define OBS_TENSOR_T FloatTensor

#define Env CraftaxClassic
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    // Process-wide reset pool size. First caller wins (setter is idempotent).
    // 0 disables caching (baseline: generate_world on every reset).
    int reset_pool_size = 0;
    DictItem* item = dict_get_unsafe(kwargs, "reset_pool_size");
    if (item != NULL) reset_pool_size = (int)item->value;
    craftax_classic_set_reset_pool_size(reset_pool_size);
    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf",           log->perf);
    dict_set(out, "score",          log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    static const char* ACH_NAMES[NUM_ACHIEVEMENTS] = {
        "collect_wood",   "place_table",    "eat_cow",       "collect_sapling",
        "collect_drink",  "make_wood_pick", "make_wood_sword","place_plant",
        "defeat_zombie",  "collect_stone",  "place_stone",   "eat_plant",
        "defeat_skeleton","make_stone_pick","make_stone_sword","wake_up",
        "place_furnace",  "collect_coal",   "collect_iron",  "collect_diamond",
        "make_iron_pick", "make_iron_sword",
    };
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        dict_set(out, ACH_NAMES[i], log->achievements[i]);
    }
}
