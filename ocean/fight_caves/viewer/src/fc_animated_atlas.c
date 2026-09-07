#include "fc_animated_atlas.h"

#include "fc_assets.h"
#include "fc_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FC_ATLAS_MAGIC 0x41544C53u
#define FC_TEXTURE_ANIM_MAGIC 0x4D4E4154u
#define FC_TEXTURE_ANIM_VERSION 1u

_Static_assert(sizeof(FcTextureAnimRow) == 16,
               "TANM rows must retain their 16-byte file layout");

static int fc_companion_path(char* out, size_t cap, const char* path,
                             const char* extension) {
    char* dot;
    int written;
    size_t offset;
    size_t remaining;
    if (!out || cap == 0 || !path || !extension) return 0;
    written = snprintf(out, cap, "%s", path);
    if (written < 0 || (size_t)written >= cap) return 0;
    dot = strrchr(out, '.');
    if (dot) {
        offset = (size_t)(dot - out);
    } else {
        offset = strlen(out);
    }
    remaining = cap - offset;
    written = snprintf(out + offset, remaining, "%s", extension);
    return written >= 0 && (size_t)written < remaining;
}

static void fc_animated_atlas_load_anims(FcAnimatedAtlas* atlas,
                                         const char* atlas_path) {
    char path[FC_ASSET_PATH_MAX];
    FILE* file;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    FcTextureAnimRow* rows;

    if (!atlas || !atlas->base_pixels || !atlas->pixels
            || !fc_companion_path(path, sizeof(path), atlas_path, ".tanim")
            || !fc_asset_exists(path))
        return;
    file = fc_asset_fopen(path, "rb");
    if (!file) return;
    if (!fc_read_exact(file, &magic, sizeof(magic), 1, path, "tanim magic")
            || !fc_read_exact(file, &version, sizeof(version), 1, path,
                              "tanim version")
            || !fc_read_exact(file, &count, sizeof(count), 1, path,
                              "tanim count")
            || magic != FC_TEXTURE_ANIM_MAGIC
            || version != FC_TEXTURE_ANIM_VERSION) {
        fc_asset_close(file);
        return;
    }
    rows = count > 0 ? calloc(count, sizeof(*rows)) : NULL;
    if (count > 0 && !rows) {
        fc_asset_close(file);
        return;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!fc_read_exact(file, &rows[i], sizeof(rows[i]), 1, path,
                           "tanim row")) {
            free(rows);
            fc_asset_close(file);
            return;
        }
    }
    fc_asset_close(file);
    atlas->anims = rows;
    atlas->anim_count = (int)count;
    fprintf(stderr, "animated atlas: %d cells loaded from %s\n",
            atlas->anim_count, path);
}

int fc_animated_atlas_load(FcAnimatedAtlas* atlas, const char* atlas_path,
                           int enable_animation) {
    FcAnimatedAtlas loaded = {0};
    FILE* file;
    uint32_t magic = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t pixel_count;
    size_t pixel_size;
    unsigned char* source_pixels;
    Image image;

    if (!atlas || !atlas_path || atlas->texture.id > 0) return 0;
    file = fc_asset_fopen(atlas_path, "rb");
    if (!file) return 0;
    if (!fc_read_exact(file, &magic, sizeof(magic), 1, atlas_path,
                       "atlas magic")
            || !fc_read_exact(file, &width, sizeof(width), 1, atlas_path,
                              "atlas width")
            || !fc_read_exact(file, &height, sizeof(height), 1, atlas_path,
                              "atlas height")
            || magic != FC_ATLAS_MAGIC || width == 0 || height == 0
            || (size_t)width > SIZE_MAX / (size_t)height) {
        fc_asset_close(file);
        return 0;
    }
    pixel_count = (size_t)width * (size_t)height;
    if (pixel_count > SIZE_MAX / 4) {
        fc_asset_close(file);
        return 0;
    }
    pixel_size = pixel_count * 4;
    source_pixels = malloc(pixel_size);
    if (!source_pixels
            || !fc_read_exact(file, source_pixels, 1, pixel_size, atlas_path,
                              "atlas pixels")) {
        free(source_pixels);
        fc_asset_close(file);
        return 0;
    }
    fc_asset_close(file);

    image = (Image) {
        .data = source_pixels,
        .width = (int)width,
        .height = (int)height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    loaded.texture = LoadTextureFromImage(image);
    if (loaded.texture.id == 0) {
        free(source_pixels);
        return 0;
    }
    SetTextureFilter(loaded.texture, TEXTURE_FILTER_POINT);
    loaded.width = (int)width;
    loaded.height = (int)height;
    if (enable_animation) {
        loaded.base_pixels = malloc(pixel_size);
        loaded.pixels = malloc(pixel_size);
        if (loaded.base_pixels && loaded.pixels) {
            memcpy(loaded.base_pixels, source_pixels, pixel_size);
            memcpy(loaded.pixels, source_pixels, pixel_size);
        } else {
            free(loaded.base_pixels);
            free(loaded.pixels);
            loaded.base_pixels = NULL;
            loaded.pixels = NULL;
        }
    }
    free(source_pixels);
    if (enable_animation)
        fc_animated_atlas_load_anims(&loaded, atlas_path);
    *atlas = loaded;
    fprintf(stderr, "animated atlas: %ux%u loaded from %s\n",
            width, height, atlas_path);
    return 1;
}

void fc_animated_atlas_update(FcAnimatedAtlas* atlas, float dt) {
    size_t total;
    if (!atlas || !atlas->pixels || !atlas->base_pixels
            || atlas->texture.id == 0 || atlas->anim_count <= 0)
        return;

    atlas->anim_ticks += dt * 50.0f;
    total = (size_t)atlas->width * (size_t)atlas->height * 4;
    memcpy(atlas->pixels, atlas->base_pixels, total);
    for (int r = 0; r < atlas->anim_count; r++) {
        FcTextureAnimRow* row = &atlas->anims[r];
        int shift;
        if (row->w == 0 || row->h == 0
                || row->x + row->w > atlas->width
                || row->y + row->h > atlas->height
                || row->speed == 0)
            continue;
        shift = (int)(atlas->anim_ticks * (float)row->speed);
        if (row->direction == 1 || row->direction == 3) {
            int pad = row->pad;
            int center_h;
            if (pad * 2 >= row->h) pad = 0;
            center_h = row->h - pad * 2;
            shift %= center_h;
            if (row->direction == 1) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                int sy = (y - pad + shift) % center_h;
                if (sy < 0) sy += center_h;
                sy += pad;
                for (int x = 0; x < row->w; x++) {
                    size_t dst = ((size_t)(row->y + y) * atlas->width
                                  + row->x + x) * 4;
                    size_t src = ((size_t)(row->y + sy) * atlas->width
                                  + row->x + x) * 4;
                    memcpy(&atlas->pixels[dst], &atlas->base_pixels[src], 4);
                }
            }
        } else if (row->direction == 2 || row->direction == 4) {
            shift %= row->w;
            if (row->direction == 2) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                for (int x = 0; x < row->w; x++) {
                    int sx = (x + shift) % row->w;
                    size_t dst;
                    size_t src;
                    if (sx < 0) sx += row->w;
                    dst = ((size_t)(row->y + y) * atlas->width
                           + row->x + x) * 4;
                    src = ((size_t)(row->y + y) * atlas->width
                           + row->x + sx) * 4;
                    memcpy(&atlas->pixels[dst], &atlas->base_pixels[src], 4);
                }
            }
        }
    }
    UpdateTexture(atlas->texture, atlas->pixels);
}

void fc_animated_atlas_unload(FcAnimatedAtlas* atlas) {
    if (!atlas) return;
    if (atlas->texture.id > 0) UnloadTexture(atlas->texture);
    free(atlas->base_pixels);
    free(atlas->pixels);
    free(atlas->anims);
    *atlas = (FcAnimatedAtlas) {0};
}
