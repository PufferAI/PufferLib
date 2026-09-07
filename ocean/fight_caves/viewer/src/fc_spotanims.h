#ifndef FC_SPOTANIMS_H
#define FC_SPOTANIMS_H

#include <stdint.h>

typedef struct {
    uint32_t id;
    int32_t model_id;
    int32_t animation_id;
    uint32_t resize_xy;
    uint32_t resize_z;
    uint32_t rotation;
    int32_t brightness;
    int32_t shadow;
} SpotAnimDef;

typedef struct {
    SpotAnimDef *defs;
    int count;
    int loaded;
} SpotAnimSet;

SpotAnimSet *spotanims_load(const char *path);
const SpotAnimDef *spotanim_find(const SpotAnimSet *set, int id);
void spotanims_free(SpotAnimSet *set);

#endif
