#include "craftax.h"

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

    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    static const char* ACH_NAMES[CRAFTAX_NUM_ACHIEVEMENTS] = {
        "collect_wood",
        "place_table",
        "eat_cow",
        "collect_sapling",
        "collect_drink",
        "make_wood_pickaxe",
        "make_wood_sword",
        "place_plant",
        "defeat_zombie",
        "collect_stone",
        "place_stone",
        "eat_plant",
        "defeat_skeleton",
        "make_stone_pickaxe",
        "make_stone_sword",
        "wake_up",
        "place_furnace",
        "collect_coal",
        "collect_iron",
        "collect_diamond",
        "make_iron_pickaxe",
        "make_iron_sword",
        "make_arrow",
        "make_torch",
        "place_torch",
        "make_diamond_sword",
        "make_iron_armour",
        "make_diamond_armour",
        "enter_gnomish_mines",
        "enter_dungeon",
        "enter_sewers",
        "enter_vault",
        "enter_troll_mines",
        "enter_fire_realm",
        "enter_ice_realm",
        "enter_graveyard",
        "defeat_gnome_warrior",
        "defeat_gnome_archer",
        "defeat_orc_solider",
        "defeat_orc_mage",
        "defeat_lizard",
        "defeat_kobold",
        "defeat_troll",
        "defeat_deep_thing",
        "defeat_pigman",
        "defeat_fire_elemental",
        "defeat_frost_troll",
        "defeat_ice_elemental",
        "damage_necromancer",
        "defeat_necromancer",
        "eat_bat",
        "eat_snail",
        "find_bow",
        "fire_bow",
        "collect_sapphire",
        "learn_fireball",
        "cast_fireball",
        "learn_iceball",
        "cast_iceball",
        "collect_ruby",
        "make_diamond_pickaxe",
        "open_chest",
        "drink_potion",
        "enchant_sword",
        "enchant_armour",
        "defeat_knight",
        "defeat_archer",
    };

    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        dict_set(out, ACH_NAMES[i], log->achievements[i]);
    }
}
