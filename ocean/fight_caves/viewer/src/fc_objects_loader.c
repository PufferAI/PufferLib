/**
 * @fileoverview Loads placed map objects from .objects binary into a single raylib Model.
 *
 * Supports two binary formats:
 *   v1 (OBJS): vertices + colors only (flat vertex coloring)
 *   v2 (OBJ2): vertices + colors + texcoords (texture atlas support)
 *
 * When v2 format is detected, also loads the companion .atlas file (raw RGBA)
 * and assigns it as the model's diffuse texture. Vertex colors are multiplied
 * by the texture sample: textured faces use white vertex color + real texture,
 * non-textured faces use HSL vertex color + white atlas pixel.
 */

#include "fc_objects_loader.h"
#include "fc_animated_atlas.h"
#include "raylib.h"
#include "rlgl.h"
#include "fc_assets.h"
#include "fc_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OBJS_MAGIC 0x4F424A53  /* "OBJS" v1 */
#define OBJ2_MAGIC 0x4F424A32  /* "OBJ2" v2 with texcoords */
#define OANM_MAGIC 0x4D4E414F  /* "OANM" animated object placements */
#define OANM_VERSION 1
ObjectMesh* objects_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "objects_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, placement_count, total_verts;
    int32_t min_wx, min_wy;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "object magic")) {
        fc_asset_close(f);
        return NULL;
    }

    int has_textures = 0;
    if (magic == OBJ2_MAGIC) {
        has_textures = 1;
    } else if (magic != OBJS_MAGIC) {
        fprintf(stderr, "objects_load: bad magic %08x\n", magic);
        fc_asset_close(f);
        return NULL;
    }

    if (!fc_read_exact(f, &placement_count, sizeof(placement_count), 1, path, "object placement count") ||
        !fc_read_exact(f, &min_wx, sizeof(min_wx), 1, path, "object min world x") ||
        !fc_read_exact(f, &min_wy, sizeof(min_wy), 1, path, "object min world y") ||
        !fc_read_exact(f, &total_verts, sizeof(total_verts), 1, path, "object vertex count")) {
        fc_asset_close(f);
        return NULL;
    }

    fprintf(stderr, "objects_load: %u placements, %u verts, format=%s\n",
            placement_count, total_verts, has_textures ? "OBJ2" : "OBJS");

    /* read vertices */
    float* raw_verts = (float*)malloc(total_verts * 3 * sizeof(float));
    if (!raw_verts ||
        !fc_read_exact(f, raw_verts, sizeof(float), total_verts * 3, path, "object vertices")) {
        free(raw_verts);
        fc_asset_close(f);
        return NULL;
    }

    /* read colors */
    unsigned char* raw_colors = (unsigned char*)malloc(total_verts * 4);
    if (!raw_colors ||
        !fc_read_exact(f, raw_colors, 1, total_verts * 4, path, "object colors")) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
        return NULL;
    }
    /* read texture coordinates (v2 only) */
    float* raw_texcoords = NULL;
    if (has_textures) {
        raw_texcoords = (float*)malloc(total_verts * 2 * sizeof(float));
        if (!raw_texcoords ||
            !fc_read_exact(f, raw_texcoords, sizeof(float), total_verts * 2,
                           path, "object texcoords")) {
            free(raw_verts);
            free(raw_colors);
            free(raw_texcoords);
            fc_asset_close(f);
            return NULL;
        }
    }
    fc_asset_close(f);

    /* build raylib mesh */
    Mesh mesh = { 0 };
    mesh.vertexCount = (int)total_verts;
    mesh.triangleCount = (int)(total_verts / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;
    mesh.texcoords = raw_texcoords;

    /* compute normals */
    mesh.normals = (float*)calloc(total_verts * 3, sizeof(float));
    if (!mesh.normals) {
        free(raw_verts);
        free(raw_colors);
        free(raw_texcoords);
        return NULL;
    }
    for (int i = 0; i < mesh.triangleCount; i++) {
        int base = i * 9;
        float ax = raw_verts[base + 0], ay = raw_verts[base + 1], az = raw_verts[base + 2];
        float bx = raw_verts[base + 3], by = raw_verts[base + 4], bz = raw_verts[base + 5];
        float cx = raw_verts[base + 6], cy = raw_verts[base + 7], cz = raw_verts[base + 8];

        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 0.0001f) { nx /= len; ny /= len; nz /= len; }

        for (int v = 0; v < 3; v++) {
            mesh.normals[i * 9 + v * 3 + 0] = nx;
            mesh.normals[i * 9 + v * 3 + 1] = ny;
            mesh.normals[i * 9 + v * 3 + 2] = nz;
        }
    }

    UploadMesh(&mesh, false);

    ObjectMesh* om = (ObjectMesh*)calloc(1, sizeof(ObjectMesh));
    om->model = LoadModelFromMesh(mesh);
    om->placement_count = (int)placement_count;
    om->total_vertex_count = (int)total_verts;
    om->min_world_x = min_wx;
    om->min_world_y = min_wy;
    om->has_textures = has_textures;
    om->loaded = 1;

    /* load atlas texture if v2 format */
    if (has_textures) {
        /* derive atlas path from objects path: replace .objects with .atlas */
        char atlas_path[1024];
        strncpy(atlas_path, path, sizeof(atlas_path) - 1);
        atlas_path[sizeof(atlas_path) - 1] = '\0';
        char* dot = strrchr(atlas_path, '.');
        if (dot) {
            strcpy(dot, ".atlas");
        } else {
            strncat(atlas_path, ".atlas", sizeof(atlas_path) - strlen(atlas_path) - 1);
        }

        if (fc_animated_atlas_load(&om->atlas, atlas_path, 1)) {
            /* assign atlas as diffuse map for the model's material */
            om->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture =
                om->atlas.texture;
        }
    }

    return om;
}

ObjectAnimSet* object_anims_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "object_anims_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic = 0, version = 0, count = 0;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "oanim magic") ||
        !fc_read_exact(f, &version, sizeof(version), 1, path, "oanim version") ||
        !fc_read_exact(f, &count, sizeof(count), 1, path, "oanim count") ||
        magic != OANM_MAGIC || version != OANM_VERSION) {
        fc_asset_close(f);
        return NULL;
    }

    ObjectAnimSet* set = (ObjectAnimSet*)calloc(1, sizeof(*set));
    if (!set) {
        fc_asset_close(f);
        return NULL;
    }
    set->rows = (ObjectAnimPlacement*)calloc(count, sizeof(*set->rows));
    if (count > 0 && !set->rows) {
        fc_asset_close(f);
        free(set);
        return NULL;
    }
    set->count = (int)count;

    for (uint32_t i = 0; i < count; i++) {
        ObjectAnimPlacement* row = &set->rows[i];
        if (!fc_read_exact(f, &row->model_id, sizeof(row->model_id), 1, path, "oanim model id") ||
            !fc_read_exact(f, &row->obj_id, sizeof(row->obj_id), 1, path, "oanim object id") ||
            !fc_read_exact(f, &row->animation_id, sizeof(row->animation_id), 1, path, "oanim animation id") ||
            !fc_read_exact(f, &row->world_x, sizeof(row->world_x), 1, path, "oanim world x") ||
            !fc_read_exact(f, &row->world_y, sizeof(row->world_y), 1, path, "oanim world y") ||
            !fc_read_exact(f, &row->plane, sizeof(row->plane), 1, path, "oanim plane") ||
            !fc_read_exact(f, &row->obj_type, sizeof(row->obj_type), 1, path, "oanim obj type") ||
            !fc_read_exact(f, &row->rotation, sizeof(row->rotation), 1, path, "oanim rotation") ||
            !fc_read_exact(f, &row->flags, sizeof(row->flags), 1, path, "oanim flags") ||
            !fc_read_exact(f, &row->pos_x, sizeof(row->pos_x), 1, path, "oanim pos x") ||
            !fc_read_exact(f, &row->pos_y, sizeof(row->pos_y), 1, path, "oanim pos y") ||
            !fc_read_exact(f, &row->pos_z, sizeof(row->pos_z), 1, path, "oanim pos z") ||
            !fc_read_exact(f, &row->phase_ticks, sizeof(row->phase_ticks), 1, path, "oanim phase")) {
            fc_asset_close(f);
            free(set->rows);
            free(set);
            return NULL;
        }
    }
    fc_asset_close(f);
    set->loaded = 1;
    fprintf(stderr, "object_anims_load: loaded %d animated placements from %s\n",
            set->count, path);
    return set;
}

void object_anims_offset(ObjectAnimSet* set, int wx, int wy) {
    if (!set || !set->loaded) return;
    for (int i = 0; i < set->count; i++) {
        set->rows[i].pos_x -= (float)wx;
        set->rows[i].pos_z += (float)wy;
        set->rows[i].world_x -= wx;
        set->rows[i].world_y -= wy;
    }
}

/* shift object vertices so world coordinates (wx, wy) become local (0, 0).
   must match terrain_offset() values for alignment. */
void objects_offset(ObjectMesh* om, int wx, int wy) {
    if (!om || !om->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;
    float* verts = om->model.meshes[0].vertices;
    for (int i = 0; i < om->total_vertex_count; i++) {
        verts[i * 3 + 0] -= dx;        /* X */
        verts[i * 3 + 2] += dz;        /* Z (negated world Y) */
    }
    UpdateMeshBuffer(om->model.meshes[0], 0, verts,
                     om->total_vertex_count * 3 * sizeof(float), 0);
    om->min_world_x -= wx;
    om->min_world_y -= wy;
    fprintf(stderr, "objects_offset: shifted by (%d, %d)\n", wx, wy);
}

void objects_free(ObjectMesh* om) {
    if (!om) return;
    fc_animated_atlas_unload(&om->atlas);
    if (om->loaded) UnloadModel(om->model);
    free(om);
}

void object_anims_free(ObjectAnimSet* set) {
    if (!set) return;
    free(set->rows);
    free(set);
}
