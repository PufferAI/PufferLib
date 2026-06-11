#include "nmmo3.h"
#define OBS_SIZE 1707
#define NUM_ATNS 1
#define ACT_SIZES {26}
#define OBS_TENSOR_T ByteTensor

#define MY_STATE
#define Env MMO
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->num_enemies = dict_get(kwargs, "num_enemies")->value;
    env->num_resources = dict_get(kwargs, "num_resources")->value;
    env->num_weapons = dict_get(kwargs, "num_weapons")->value;
    env->num_gems = dict_get(kwargs, "num_gems")->value;
    env->tiers = dict_get(kwargs, "tiers")->value;
    env->levels = dict_get(kwargs, "levels")->value;
    env->teleportitis_prob = dict_get(kwargs, "teleportitis_prob")->value;
    env->enemy_respawn_ticks = dict_get(kwargs, "enemy_respawn_ticks")->value;
    env->item_respawn_ticks = dict_get(kwargs, "item_respawn_ticks")->value;
    env->x_window = dict_get(kwargs, "x_window")->value;
    env->y_window = dict_get(kwargs, "y_window")->value;
    env->reward_combat_level = dict_get(kwargs, "reward_combat_level")->value;
    env->reward_prof_level = dict_get(kwargs, "reward_prof_level")->value;
    env->reward_item_level = dict_get(kwargs, "reward_item_level")->value;
    env->reward_market = dict_get(kwargs, "reward_market")->value;
    env->reward_death = dict_get(kwargs, "reward_death")->value;
    init(env);
}

#define NMMO3_STATE_COLS 10

// Exports terrain, packed entity rows, and the tick counter for rendering
// and telemetry. Rows are players first, then enemies:
// (kind, r, c, hp, hp_max, comb_lvl, prof_lvl, dir, anim, in_combat),
// kind 0 = player, 1 = enemy. Terrain is allocated once in init and only
// rewritten by c_reset, so it is exported zero-copy; positions and tick
// live in scratch valid until the next my_state call and are copied by
// the binding layer.
int my_state(void* e, StateField* fields, int max_fields) {
    if (max_fields < 3) {
        return 0;
    }
    MMO* env = (MMO*)e;
    int num_entities = env->num_agents + env->num_enemies;

    static int* positions = NULL;
    static int positions_cap = 0;
    if (num_entities * NMMO3_STATE_COLS > positions_cap) {
        positions_cap = num_entities * NMMO3_STATE_COLS;
        positions = realloc(positions, positions_cap * sizeof(int));
    }
    for (int i = 0; i < num_entities; i++) {
        int kind = i >= env->num_agents;
        Entity* ent = kind ? &env->enemies[i - env->num_agents] : &env->players[i];
        int* row = &positions[i * NMMO3_STATE_COLS];
        row[0] = kind;
        row[1] = ent->r;
        row[2] = ent->c;
        row[3] = ent->hp;
        row[4] = ent->hp_max;
        row[5] = ent->comb_lvl;
        row[6] = ent->prof_lvl;
        row[7] = ent->dir;
        row[8] = ent->anim;
        row[9] = ent->in_combat;
    }

    static int tick;
    tick = env->tick;

    fields[0] = (StateField){"terrain", env->terrain, "int8", 2,
        {env->height, env->width}, PUFF_STATE_ZERO_COPY};
    fields[1] = (StateField){"positions", positions, "int32", 2,
        {num_entities, NMMO3_STATE_COLS}, 0};
    fields[2] = (StateField){"tick", &tick, "int32", 1, {1}, 0};
    return 3;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "return_comb_lvl", log->return_comb_lvl);
    dict_set(out, "return_prof_lvl", log->return_prof_lvl);
    dict_set(out, "return_item_atk_lvl", log->return_item_atk_lvl);
    dict_set(out, "return_item_def_lvl", log->return_item_def_lvl);
    dict_set(out, "return_market_buy", log->return_market_buy);
    dict_set(out, "return_market_sell", log->return_market_sell);
    dict_set(out, "return_death", log->return_death);
    dict_set(out, "min_comb_prof", log->min_comb_prof);
    dict_set(out, "purchases", log->purchases);
    dict_set(out, "sales", log->sales);
    dict_set(out, "equip_attack", log->equip_attack);
    dict_set(out, "equip_defense", log->equip_defense);
    dict_set(out, "r", log->r);
    dict_set(out, "c", log->c);
}
