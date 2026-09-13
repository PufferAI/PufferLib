#ifndef OSRS_ASSET_RAYLIB_H
#define OSRS_ASSET_RAYLIB_H

#include "osrs_assets.h"

#if __has_include("raylib.h")
#include "raylib.h"
#elif __has_include("raylib-5.5_macos/include/raylib.h")
#include "raylib-5.5_macos/include/raylib.h"
#else
#error "raylib.h not found"
#endif

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline const char* osrs_asset_ext(const char* path, const char* fallback) {
    const char* dot = path ? strrchr(path, '.') : NULL;
    return dot && dot[0] ? dot : fallback;
}

static inline Image osrs_asset_load_image(const char* path) {
    Image empty = {0};
    OsrsAssetBytes bytes = osrs_asset_read_all(path);
    if (!bytes.data || bytes.size == 0) {
        osrs_asset_bytes_free(&bytes);
        return empty;
    }
    if (bytes.size > (size_t)INT_MAX) {
        fprintf(stderr, "image asset too large: %s (%zu bytes)\n", path, bytes.size);
        abort();
    }
    Image image = LoadImageFromMemory(
        osrs_asset_ext(path, ".png"), bytes.data, (int)bytes.size);
    osrs_asset_bytes_free(&bytes);
    return image;
}

#define ATLS_MAGIC 0x41544C53

// .atlas is ATLS (raw RGBA) or a PNG with the same filename. Decodes to RGBA8.
static inline Image osrs_asset_load_atlas_image(const char* path) {
    Image empty = {0};
    OsrsAssetBytes bytes = osrs_asset_read_all(path);
    if (!bytes.data || bytes.size < 8) {
        fprintf(stderr, "osrs_asset_load_atlas_image: could not read %s\n",
            path ? path : "(null)");
        osrs_asset_bytes_free(&bytes);
        abort();
    }

    static const unsigned char png_magic[8] = {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
    };
    if (bytes.size >= 8 && memcmp(bytes.data, png_magic, 8) == 0) {
        if (bytes.size > (size_t)INT_MAX) {
            fprintf(stderr, "atlas PNG too large: %s (%zu bytes)\n", path, bytes.size);
            abort();
        }
        Image image = LoadImageFromMemory(".png", bytes.data, (int)bytes.size);
        osrs_asset_bytes_free(&bytes);
        if (!image.data || image.width <= 0 || image.height <= 0) {
            fprintf(stderr, "osrs_asset_load_atlas_image: PNG decode failed: %s\n", path);
            abort();
        }
        if (image.format != PIXELFORMAT_UNCOMPRESSED_R8G8B8A8) {
            ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        }
        return image;
    }

    if (bytes.size < 12) {
        fprintf(stderr, "osrs_asset_load_atlas_image: truncated ATLS: %s\n", path);
        abort();
    }
    uint32_t magic, width, height;
    memcpy(&magic, bytes.data, 4);
    memcpy(&width, bytes.data + 4, 4);
    memcpy(&height, bytes.data + 8, 4);
    size_t pixel_bytes = (size_t)width * (size_t)height * 4;
    if (magic != ATLS_MAGIC || width == 0 || height == 0 ||
            bytes.size != 12 + pixel_bytes) {
        fprintf(stderr,
            "osrs_asset_load_atlas_image: bad ATLS %s magic=%08x %ux%u size=%zu\n",
            path, magic, width, height, bytes.size);
        abort();
    }
    unsigned char* pixels = (unsigned char*)osrs_malloc_or_abort(
        pixel_bytes, "atlas pixels");
    memcpy(pixels, bytes.data + 12, pixel_bytes);
    osrs_asset_bytes_free(&bytes);
    empty.data = pixels;
    empty.width = (int)width;
    empty.height = (int)height;
    empty.mipmaps = 1;
    empty.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    return empty;
}

static inline Texture2D osrs_asset_load_texture(const char* path) {
    Texture2D empty = {0};
    Image image = osrs_asset_load_image(path);
    if (!image.data) return empty;
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

static inline Font osrs_asset_load_font(const char* path, int font_size) {
    Font empty = {0};
    OsrsAssetBytes bytes = osrs_asset_read_all(path);
    if (!bytes.data || bytes.size == 0) {
        osrs_asset_bytes_free(&bytes);
        return empty;
    }
    if (bytes.size > (size_t)INT_MAX) {
        fprintf(stderr, "font asset too large: %s (%zu bytes)\n", path, bytes.size);
        abort();
    }
    Font font = LoadFontFromMemory(
        osrs_asset_ext(path, ".ttf"), bytes.data, (int)bytes.size, font_size, NULL, 95);
    osrs_asset_bytes_free(&bytes);
    return font.texture.id != 0 ? font : empty;
}

#endif
