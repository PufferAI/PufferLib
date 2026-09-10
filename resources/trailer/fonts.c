#include "trailer/fonts.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char path[96];
    int px;
    Font f;
} TFont;

static TFont g_tf[48];
static int g_ntf;
static unsigned g_ui_tex;
static unsigned g_mo_tex;

Font tfont(const char *path, int px) {
    if (px < 8) px = 8;
    if (!path) path = TTF_UI;
    for (int i = 0; i < g_ntf; i++) {
        if (g_tf[i].px == px && strcmp(g_tf[i].path, path) == 0) {
            return g_tf[i].f;
        }
    }
    int cps[224];
    for (int i = 0; i < 224; i++) {
        cps[i] = 32 + i;
    }
    Font f = LoadFontEx(path, px, cps, 224);
    SetTextureFilter(f.texture, px <= 24 ? TEXTURE_FILTER_POINT : TEXTURE_FILTER_BILINEAR);
    if (g_ntf < (int)(sizeof g_tf / sizeof g_tf[0])) {
        snprintf(g_tf[g_ntf].path, sizeof g_tf[g_ntf].path, "%s", path);
        g_tf[g_ntf].px = px;
        g_tf[g_ntf].f = f;
        g_ntf++;
    }
    return f;
}

void tfont_families(Font ui, Font mono) {
    g_ui_tex = ui.texture.id;
    g_mo_tex = mono.texture.id;
}

Font tfont_fit(Font proto, int px) {
    const char *path = TTF_MONO;
    if (g_ui_tex && proto.texture.id == g_ui_tex) {
        path = TTF_UI;
    } else if (g_mo_tex && proto.texture.id == g_mo_tex) {
        path = TTF_MONO;
    }
    return tfont(path, px);
}

void tfont_draw(Font proto, const char *s, float x, float y, float sz, float spacing, Color c) {
    Font f = tfont_fit(proto, (int)(sz + 0.5f));
    DrawTextEx(f, s, (Vector2){roundf(x), roundf(y)}, (float)f.baseSize, spacing, c);
}

float tfont_width(Font proto, const char *s, float sz, float spacing) {
    Font f = tfont_fit(proto, (int)(sz + 0.5f));
    return MeasureTextEx(f, s, (float)f.baseSize, spacing).x;
}

void tfont_unload(void) {
    for (int i = 0; i < g_ntf; i++) {
        UnloadFont(g_tf[i].f);
    }
    g_ntf = 0;
    g_ui_tex = 0;
    g_mo_tex = 0;
}
