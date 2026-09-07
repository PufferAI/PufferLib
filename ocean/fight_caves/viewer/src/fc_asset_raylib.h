#ifndef FC_ASSET_RAYLIB_H
#define FC_ASSET_RAYLIB_H

#include "raylib.h"

Texture2D fc_load_texture_asset(const char* path);
Image fc_load_image_asset(const char* path);
Font fc_load_font_asset(const char* path, int font_size);

#endif /* FC_ASSET_RAYLIB_H */
