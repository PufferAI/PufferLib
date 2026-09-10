// Breakout scale plot (hidden size vs SPS / VRAM) as a trailer scene.
#ifndef TRAILER_PLOT_SCALE_H
#define TRAILER_PLOT_SCALE_H

#include "raylib.h"

void plot_scale_init(void);
void plot_scale_draw(Font title, Font body, Shader *star, float reveal, float alpha);
void plot_scale_draw_box(Font title, Font body, Shader *star, float reveal, float alpha,
                         Rectangle box);

#endif
