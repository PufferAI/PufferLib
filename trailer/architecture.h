// Five Levels of Parallelism — scene API for the 5.0 trailer.
// Standalone: ./build.sh trailer && ./trailer/architecture
#ifndef TRAILER_ARCHITECTURE_H
#define TRAILER_ARCHITECTURE_H

#include "raylib.h"

typedef struct Arch Arch;

Arch *arch_create(Font ui, Font mono, Texture2D logo, Shader star);
void arch_destroy(Arch *a);          // does not unload fonts / logo / shader
void arch_reset(Arch *a);
void arch_update(Arch *a, float dt);
void arch_draw(Arch *a, float alpha); // into the current framebuffer; no BeginDrawing
void arch_hide_hud(Arch *a, int hide);
float arch_loop_sec(void);

#endif
