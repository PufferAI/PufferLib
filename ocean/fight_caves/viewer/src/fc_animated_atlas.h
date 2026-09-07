#ifndef FC_ANIMATED_ATLAS_H
#define FC_ANIMATED_ATLAS_H

#include "raylib.h"

#include <stdint.h>

typedef struct {
    uint32_t texture_id;
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
    uint8_t direction;
    uint8_t speed;
    uint16_t pad;
} FcTextureAnimRow;

typedef struct {
    Texture2D texture;
    unsigned char* base_pixels;
    unsigned char* pixels;
    FcTextureAnimRow* anims;
    int width;
    int height;
    int anim_count;
    float anim_ticks;
} FcAnimatedAtlas;

int fc_animated_atlas_load(FcAnimatedAtlas* atlas, const char* atlas_path,
                           int enable_animation);
void fc_animated_atlas_update(FcAnimatedAtlas* atlas, float dt);
void fc_animated_atlas_unload(FcAnimatedAtlas* atlas);

#endif /* FC_ANIMATED_ATLAS_H */
