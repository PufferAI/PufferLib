#include "block_blast.h"

#define OBS_SIZE BB_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {BB_ACTIONS}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK BB_ACTIONS

#define Env BlockBlast
static inline void puffer_state_refresh(BlockBlast* env) { bb_compute_observations(env); }
#include "vecenv.h"

static inline int dict_get_int_default(Dict* kwargs, const char* key, int default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return default_value;
    }
    return (int)item->value;
}

static inline float dict_get_float_default(Dict* kwargs, const char* key, float default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return default_value;
    }
    return (float)item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = dict_get_int_default(kwargs, "max_steps", 512);
    env->place_reward = dict_get_float_default(kwargs, "place_reward", 0.01f);
    env->line_reward = dict_get_float_default(kwargs, "line_reward", 1.0f);
    env->combo_reward = dict_get_float_default(kwargs, "combo_reward", 0.15f);
    env->free_space_reward = dict_get_float_default(kwargs, "free_space_reward", 0.03f);
    env->mobility_reward = dict_get_float_default(kwargs, "mobility_reward", 0.04f);
    env->fill_penalty = dict_get_float_default(kwargs, "fill_penalty", 0.05f);
    env->no_clear_penalty = dict_get_float_default(kwargs, "no_clear_penalty", 0.01f);
    env->dead_space_penalty = dict_get_float_default(kwargs, "dead_space_penalty", 0.0f);
    env->fragmentation_penalty = dict_get_float_default(kwargs, "fragmentation_penalty", 0.0f);
    env->low_mobility_penalty = dict_get_float_default(kwargs, "low_mobility_penalty", 0.0f);
    env->low_mobility_threshold = dict_get_int_default(kwargs, "low_mobility_threshold", 12);
    env->invalid_penalty = dict_get_float_default(kwargs, "invalid_penalty", -0.5f);
    env->terminal_penalty = dict_get_float_default(kwargs, "terminal_penalty", 1.0f);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lines_cleared", log->lines_cleared);
    dict_set(out, "clears", log->clears);
    dict_set(out, "placements", log->placements);
    dict_set(out, "invalid_actions", log->invalid_actions);
    dict_set(out, "board_fill", log->board_fill);
    dict_set(out, "available_moves", log->available_moves);
    dict_set(out, "avg_board_fill", log->avg_board_fill);
    dict_set(out, "avg_available_moves", log->avg_available_moves);
    dict_set(out, "dead_space", log->dead_space);
    dict_set(out, "empty_components", log->empty_components);
    dict_set(out, "largest_empty_region", log->largest_empty_region);
    dict_set(out, "avg_dead_space", log->avg_dead_space);
    dict_set(out, "avg_empty_components", log->avg_empty_components);
    dict_set(out, "avg_largest_empty_region", log->avg_largest_empty_region);
    dict_set(out, "avg_min_slot_moves", log->avg_min_slot_moves);
    dict_set(out, "low_mobility_steps", log->low_mobility_steps);
    dict_set(out, "game_over", log->game_over);
    dict_set(out, "max_combo", log->max_combo);
    dict_set(out, "survived_max_steps", log->survived_max_steps);
}
