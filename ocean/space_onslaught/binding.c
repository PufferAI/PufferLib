#include "space_onslaught.h"
#define OBS_SIZE SO_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {SO_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env SpaceOnslaught
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->alien_width = dict_get(kwargs, "alien_width")->value;
    env->alien_height = dict_get(kwargs, "alien_height")->value;
    env->alien_spacing_x = dict_get(kwargs, "alien_spacing_x")->value;
    env->alien_spacing_y = dict_get(kwargs, "alien_spacing_y")->value;
    env->edge_margin = dict_get(kwargs, "edge_margin")->value;
    env->alien_step_x = dict_get(kwargs, "alien_step_x")->value;
    env->alien_step_down = dict_get(kwargs, "alien_step_down")->value;
    env->base_move_interval = dict_get(kwargs, "base_move_interval")->value;
    env->min_move_interval = dict_get(kwargs, "min_move_interval")->value;
    env->player_width = dict_get(kwargs, "player_width")->value;
    env->player_height = dict_get(kwargs, "player_height")->value;
    env->player_speed = dict_get(kwargs, "player_speed")->value;
    env->player_bullet_speed = dict_get(kwargs, "player_bullet_speed")->value;
    env->alien_bullet_speed = dict_get(kwargs, "alien_bullet_speed")->value;
    env->initial_lives = dict_get(kwargs, "initial_lives")->value;
    env->alien_fire_prob = dict_get(kwargs, "alien_fire_prob")->value;
    if (env->frameskip < 1) {
        env->frameskip = 1;
    }
    if (env->initial_lives < 1) {
        env->initial_lives = 1;
    }
    if (env->width < 1) {
        env->width = 1;
    }
    if (env->height < 1) {
        env->height = 1;
    }
    if (env->min_move_interval < 1) {
        env->min_move_interval = 1;
    }
    if (env->base_move_interval < 1) {
        env->base_move_interval = 1;
    }
    if (env->alien_step_x < 1) {
        env->alien_step_x = 1;
    }
    if (env->alien_fire_prob < 0.0f) {
        env->alien_fire_prob = 0.0f;
    }
    if (env->alien_fire_prob > 1.0f) {
        env->alien_fire_prob = 1.0f;
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
