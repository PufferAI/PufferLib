#include "nmmo3.h"

#define Env MMO
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_players = unpack(kwargs, "num_players");
    env->num_enemies = unpack(kwargs, "num_enemies");
    env->num_resources = unpack(kwargs, "num_resources");
    env->num_weapons = unpack(kwargs, "num_weapons");
    env->num_gems = unpack(kwargs, "num_gems");
    env->tiers = unpack(kwargs, "tiers");
    env->levels = unpack(kwargs, "levels");
    env->teleportitis_prob = unpack(kwargs, "teleportitis_prob");
    env->enemy_respawn_ticks = unpack(kwargs, "enemy_respawn_ticks");
    env->item_respawn_ticks = unpack(kwargs, "item_respawn_ticks");
    env->x_window = unpack(kwargs, "x_window");
    env->y_window = unpack(kwargs, "y_window");
    env->reward_combat_level = unpack(kwargs, "reward_combat_level");
    env->reward_prof_level = unpack(kwargs, "reward_prof_level");
    env->reward_item_level = unpack(kwargs, "reward_item_level");
    env->reward_market = unpack(kwargs, "reward_market");
    env->reward_death = unpack(kwargs, "reward_death");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "return_comb_lvl", log->return_comb_lvl);
    assign_to_dict(dict, "return_prof_lvl", log->return_prof_lvl);
    assign_to_dict(dict, "return_item_atk_lvl", log->return_item_atk_lvl);
    assign_to_dict(dict, "return_item_def_lvl", log->return_item_def_lvl);
    assign_to_dict(dict, "return_market_buy", log->return_market_buy);
    assign_to_dict(dict, "return_market_sell", log->return_market_sell);
    assign_to_dict(dict, "return_death", log->return_death);
    assign_to_dict(dict, "min_comb_prof", log->min_comb_prof);
    assign_to_dict(dict, "purchases", log->purchases);
    assign_to_dict(dict, "sales", log->sales);
    assign_to_dict(dict, "equip_attack", log->equip_attack);
    assign_to_dict(dict, "equip_defense", log->equip_defense);
    assign_to_dict(dict, "r", log->r);
    assign_to_dict(dict, "c", log->c);
    return 0;
}
