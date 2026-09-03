// 4.0 trailer starfield — gravitational spiral / nova, same shaders.
#ifndef TRAILER_STARS_H
#define TRAILER_STARS_H

#include "raylib.h"

void stars_init(void);
void stars_reset(void);
void stars_drift(float dt);
void stars_spiral(float dt);
void stars_nova(float dt);
void stars_draw(Shader *sh, float alpha);
void stars_draw_nova(Shader *sh, float alpha);

// Brighten an existing field star nearest (x,y). Returns screen position.
void stars_clear_heroes(void);
Vector2 stars_make_hero(float x, float y);
Vector2 stars_hero_pos(int slot);

#endif
