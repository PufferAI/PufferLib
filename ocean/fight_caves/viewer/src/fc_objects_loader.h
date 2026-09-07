#ifndef FC_OBJECTS_LOADER_H
#define FC_OBJECTS_LOADER_H

#include "fc_animated_atlas.h"
#include "raylib.h"
#include <stdint.h>

#define OANM_FLAG_DYNAMIC_BASE 1u
#define OANM_FLAG_DYNAMIC_REPLACEMENT 2u

typedef struct {
    Model model;
    FcAnimatedAtlas atlas;
    int placement_count;
    int total_vertex_count;
    int min_world_x;
    int min_world_y;
    int has_textures;
    int loaded;
} ObjectMesh;

typedef struct {
    uint32_t model_id;
    uint32_t obj_id;
    int32_t animation_id;
    int32_t world_x;
    int32_t world_y;
    uint8_t plane;
    uint8_t obj_type;
    uint8_t rotation;
    uint8_t flags;
    float pos_x;
    float pos_y;
    float pos_z;
    float phase_ticks;
} ObjectAnimPlacement;

typedef struct {
    ObjectAnimPlacement *rows;
    int count;
    int loaded;
} ObjectAnimSet;

ObjectMesh *objects_load(const char *path);
ObjectAnimSet *object_anims_load(const char *path);
void object_anims_offset(ObjectAnimSet *set, int wx, int wy);
void objects_offset(ObjectMesh *om, int wx, int wy);
void objects_free(ObjectMesh *om);
void object_anims_free(ObjectAnimSet *set);

#endif
