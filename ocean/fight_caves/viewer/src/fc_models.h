#ifndef FC_MODELS_H
#define FC_MODELS_H

#include "raylib.h"
#include <stdint.h>

typedef struct {
    uint8_t textured;
    uint16_t tex_a;
    uint16_t tex_b;
    uint16_t tex_c;
    float u_base;
    float v_base;
    float u_scale;
    float v_scale;
    float repeat_v_margin;
} ModelFaceUvInfo;

typedef struct {
    uint32_t model_id;
    Model model;
    int loaded;
    float *rest_verts;
    float *rest_texcoords;
    int16_t *base_verts;
    uint8_t *vertex_skins;
    uint16_t *face_indices;
    uint8_t *face_priorities;
    ModelFaceUvInfo *face_uvs;
    int base_vert_count;
    int face_count;
} ModelEntry;

typedef struct {
    ModelEntry *entries;
    int *index_by_id;
    int count;
    int index_limit;
    int has_textures;
    int loaded;
} ModelSet;

ModelSet *models_load(const char *path, Texture2D atlas_texture);
ModelEntry *model_find(ModelSet *set, uint32_t id);
void models_recompute_texture_uvs_from_vertices(ModelEntry *entry,
                                                const int16_t *verts);
void models_free(ModelSet *set);

#endif
