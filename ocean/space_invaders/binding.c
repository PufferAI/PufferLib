#include "space_invaders.h"
#define OBS_SIZE (9 + SI_NUM_INVADERS + 3 * SI_MAX_ENEMY_BULLETS)
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T FloatTensor

#define Env SpaceInvaders
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->player_speed = dict_get(kwargs, "player_speed")->value;
    env->player_bullet_speed = dict_get(kwargs, "player_bullet_speed")->value;
    env->enemy_bullet_speed = dict_get(kwargs, "enemy_bullet_speed")->value;
    env->formation_dx = dict_get(kwargs, "formation_dx")->value;
    env->formation_dy = dict_get(kwargs, "formation_dy")->value;
    env->formation_start_interval = dict_get(kwargs, "formation_start_interval")->value;
    env->enemy_fire_interval = dict_get(kwargs, "enemy_fire_interval")->value;
    env->invader_w = dict_get(kwargs, "invader_w")->value;
    env->invader_h = dict_get(kwargs, "invader_h")->value;
    env->invader_spacing_x = dict_get(kwargs, "invader_spacing_x")->value;
    env->invader_spacing_y = dict_get(kwargs, "invader_spacing_y")->value;
    env->formation_margin_x = dict_get(kwargs, "formation_margin_x")->value;
    env->formation_margin_y = dict_get(kwargs, "formation_margin_y")->value;
    env->player_w = dict_get(kwargs, "player_w")->value;
    env->player_h = dict_get(kwargs, "player_h")->value;
    env->player_y_offset = dict_get(kwargs, "player_y_offset")->value;
    env->bullet_w = dict_get(kwargs, "bullet_w")->value;
    env->bullet_h = dict_get(kwargs, "bullet_h")->value;
    env->max_lives = dict_get(kwargs, "max_lives")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
