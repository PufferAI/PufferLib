#include "fc_asset_raylib.h"

#include "fc_assets.h"

#include <stdlib.h>
#include <string.h>

static const char* fc_asset_extension(const char* path, const char* fallback) {
    const char* dot = path ? strrchr(path, '.') : NULL;
    return dot && dot[0] ? dot : fallback;
}

Image fc_load_image_asset(const char* path) {
    Image empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Image image;

    if (!bytes || size == 0) return empty;
    image = LoadImageFromMemory(fc_asset_extension(path, ".png"), bytes,
                                (int)size);
    free(bytes);
    return image;
}

Texture2D fc_load_texture_asset(const char* path) {
    Texture2D empty = {0};
    Image image = fc_load_image_asset(path);
    Texture2D texture;

    if (!image.data) return empty;
    texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

Font fc_load_font_asset(const char* path, int font_size) {
    Font empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Font font;

    if (!bytes || size == 0) return empty;
    font = LoadFontFromMemory(fc_asset_extension(path, ".ttf"), bytes,
                              (int)size, font_size, NULL, 95);
    free(bytes);
    return font.texture.id != 0 ? font : empty;
}
