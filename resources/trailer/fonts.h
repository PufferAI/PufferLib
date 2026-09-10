#ifndef TRAILER_FONTS_H
#define TRAILER_FONTS_H

#include "raylib.h"

#define TTF_UI   "resources/shared/Montserrat-Regular.ttf"
#define TTF_MONO "resources/shared/JetBrainsMono-Medium.ttf"

// Rasterize (path, px) once and reuse. Draw at f.baseSize for 1:1 glyphs.
Font tfont(const char *path, int px);
void tfont_families(Font ui, Font mono);
Font tfont_fit(Font proto, int px);
void tfont_draw(Font proto, const char *s, float x, float y, float sz, float spacing, Color c);
float tfont_width(Font proto, const char *s, float sz, float spacing);
void tfont_unload(void);

#endif
