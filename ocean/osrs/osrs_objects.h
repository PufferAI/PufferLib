#ifndef OSRS_OBJECTS_H
#define OSRS_OBJECTS_H

#include "raylib.h"
#include "rlgl.h"
#include "osrs_assets.h"
#include "osrs_asset_raylib.h"
#include "osrs_binary_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OBJS_MAGIC 0x4F424A53
#define OBJ2_MAGIC 0x4F424A32

typedef struct {
    Model model;
    Texture2D atlas_texture;
    int placement_count;
    int total_vertex_count;
    int min_world_x;
    int min_world_y;
    int has_textures;
    int loaded;
} ObjectMesh;

static Texture2D objects_load_atlas(const char* atlas_path) {
    Image img = osrs_asset_load_atlas_image(atlas_path);
    Texture2D tex = LoadTextureFromImage(img);
    if (tex.id > 0) SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    fprintf(stderr, "objects_load_atlas: loaded %dx%d atlas texture\n",
        img.width, img.height);
    UnloadImage(img);
    return tex;
}

static ObjectMesh* objects_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "objects_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, placement_count, total_verts;
    int32_t min_wx, min_wy;
    osrs_read_exact(f, &magic, 4, 1, path, "magic");

    int has_textures = 0;
    if (magic == OBJ2_MAGIC) {
        has_textures = 1;
    } else if (magic != OBJS_MAGIC) {
        fprintf(stderr, "objects_load: bad magic %08x\n", magic);
        abort();
    }

    osrs_read_exact(f, &placement_count, 4, 1, path, "placement count");
    osrs_read_exact(f, &min_wx, 4, 1, path, "min world x");
    osrs_read_exact(f, &min_wy, 4, 1, path, "min world y");
    osrs_read_exact(f, &total_verts, 4, 1, path, "vertex count");

    fprintf(stderr, "objects_load: %u placements, %u verts, format=%s\n",
            placement_count, total_verts, has_textures ? "OBJ2" : "OBJS");

    float* raw_verts = (float*)osrs_malloc_or_abort(
        total_verts * 3 * sizeof(float), "object vertices");
    osrs_read_exact(f, raw_verts, sizeof(float), total_verts * 3, path, "vertices");

    unsigned char* raw_colors = (unsigned char*)osrs_malloc_or_abort(
        total_verts * 4, "object colors");
    osrs_read_exact(f, raw_colors, 1, total_verts * 4, path, "colors");

    float* raw_texcoords = NULL;
    if (has_textures) {
        raw_texcoords = (float*)osrs_malloc_or_abort(
            total_verts * 2 * sizeof(float), "object texture coordinates");
        osrs_read_exact(f, raw_texcoords, sizeof(float),
            total_verts * 2, path, "texture coordinates");
    }
    fclose(f);

    Mesh mesh = { 0 };
    mesh.vertexCount = (int)total_verts;
    mesh.triangleCount = (int)(total_verts / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;
    mesh.texcoords = raw_texcoords;

    mesh.normals = (float*)osrs_calloc_or_abort(
        total_verts * 3, sizeof(float), "object normals");
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

    ObjectMesh* om = (ObjectMesh*)osrs_calloc_or_abort(
        1, sizeof(ObjectMesh), "object mesh");
    om->model = LoadModelFromMesh(mesh);
    om->placement_count = (int)placement_count;
    om->total_vertex_count = (int)total_verts;
    om->min_world_x = min_wx;
    om->min_world_y = min_wy;
    om->has_textures = has_textures;
    om->loaded = 1;

    if (has_textures) {
        char atlas_path[1024];
        strncpy(atlas_path, path, sizeof(atlas_path) - 1);
        atlas_path[sizeof(atlas_path) - 1] = '\0';
        char* dot = strrchr(atlas_path, '.');
        if (dot) {
            strcpy(dot, ".atlas");
        } else {
            strncat(atlas_path, ".atlas", sizeof(atlas_path) - strlen(atlas_path) - 1);
        }

        om->atlas_texture = objects_load_atlas(atlas_path);
        if (om->atlas_texture.id > 0) {
            om->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = om->atlas_texture;
        }
    }

    return om;
}

static void objects_offset(ObjectMesh* om, int wx, int wy) {
    if (!om || !om->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;
    float* verts = om->model.meshes[0].vertices;
    for (int i = 0; i < om->total_vertex_count; i++) {
        verts[i * 3 + 0] -= dx;
        verts[i * 3 + 2] += dz;
    }
    UpdateMeshBuffer(om->model.meshes[0], 0, verts,
                     om->total_vertex_count * 3 * sizeof(float), 0);
    om->min_world_x -= wx;
    om->min_world_y -= wy;
    fprintf(stderr, "objects_offset: shifted by (%d, %d)\n", wx, wy);
}

static void objects_free(ObjectMesh* om) {
    if (!om) return;
    if (om->loaded) {
        if (om->atlas_texture.id > 0) {
            UnloadTexture(om->atlas_texture);
        }
        UnloadModel(om->model);
    }
    free(om);
}

#endif
