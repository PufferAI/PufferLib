#ifndef RUNEC_VIEWER_UI_ASSETS_H
#define RUNEC_VIEWER_UI_ASSETS_H

#include "raylib.h"

#define RUNEC_UI_ASSET_MAX 512

typedef struct RuneCUiAssets {
    Texture2D textures[RUNEC_UI_ASSET_MAX];
    unsigned char loaded[RUNEC_UI_ASSET_MAX];
    Font font;
    Font small_font;
    int font_loaded;
    int small_font_loaded;
    int loaded_count;
    int missing_count;
    int required_count;
    int missing_required_count;
    int missing_optional_count;
} RuneCUiAssets;

void runec_ui_assets_load(RuneCUiAssets *assets);
void runec_ui_assets_unload(RuneCUiAssets *assets);

const Texture2D *runec_ui_asset(const RuneCUiAssets *assets, const char *name);
int runec_ui_asset_ready(const RuneCUiAssets *assets, const char *name);
void runec_ui_draw_asset(const RuneCUiAssets *assets, const char *name,
                         Rectangle dst, Color tint);
Font runec_ui_font(const RuneCUiAssets *assets);
Font runec_ui_font_for_size(const RuneCUiAssets *assets, float size);
void runec_ui_draw_text_shadow(const RuneCUiAssets *assets, const char *text,
                               float x, float y, float size, Color color);

#endif
