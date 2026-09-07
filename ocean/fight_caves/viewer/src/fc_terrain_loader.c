/**
 * @fileoverview Loads terrain mesh from .terrain binary into raylib Model.
 *
 * Binary format:
 *   magic: uint32 "TERR" (0x54455252)
 *   vertex_count: uint32
 *   region_count: uint32
 *   min_world_x: int32
 *   min_world_y: int32
 *   vertices: float32[vertex_count * 3]
 *   colors: uint8[vertex_count * 4]
 */

#include "fc_terrain_loader.h"
#include "raylib.h"
#include "fc_assets.h"
#include "fc_io.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TERR_MAGIC 0x54455252

TerrainMesh* terrain_load(const char* path) {
    FILE* f = fc_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "terrain_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, vert_count, region_count;
    int32_t min_wx, min_wy;
    if (!fc_read_exact(f, &magic, sizeof(magic), 1, path, "terrain magic")) {
        fc_asset_close(f);
        return NULL;
    }
    if (magic != TERR_MAGIC) {
        fprintf(stderr, "terrain_load: bad magic %08x\n", magic);
        fc_asset_close(f);
        return NULL;
    }
    if (!fc_read_exact(f, &vert_count, sizeof(vert_count), 1, path, "terrain vertex count") ||
        !fc_read_exact(f, &region_count, sizeof(region_count), 1, path, "terrain region count") ||
        !fc_read_exact(f, &min_wx, sizeof(min_wx), 1, path, "terrain min world x") ||
        !fc_read_exact(f, &min_wy, sizeof(min_wy), 1, path, "terrain min world y")) {
        fc_asset_close(f);
        return NULL;
    }

    fprintf(stderr, "terrain_load: %u verts, %u regions, origin (%d, %d)\n",
            vert_count, region_count, min_wx, min_wy);

    /* read vertices */
    float* raw_verts = (float*)malloc(vert_count * 3 * sizeof(float));
    if (!raw_verts ||
        !fc_read_exact(f, raw_verts, sizeof(float), vert_count * 3, path, "terrain vertices")) {
        free(raw_verts);
        fc_asset_close(f);
        return NULL;
    }

    /* read colors */
    unsigned char* raw_colors = (unsigned char*)malloc(vert_count * 4);
    if (!raw_colors ||
        !fc_read_exact(f, raw_colors, 1, vert_count * 4, path, "terrain colors")) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
        return NULL;
    }
    /* build raylib mesh */
    Mesh mesh = { 0 };
    mesh.vertexCount = (int)vert_count;
    mesh.triangleCount = (int)(vert_count / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;

    /* compute normals for proper lighting */
    mesh.normals = (float*)calloc(vert_count * 3, sizeof(float));
    if (!mesh.normals) {
        free(raw_verts);
        free(raw_colors);
        fc_asset_close(f);
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

    TerrainMesh* tm = (TerrainMesh*)calloc(1, sizeof(TerrainMesh));
    tm->model = LoadModelFromMesh(mesh);
    tm->vertex_count = (int)vert_count;
    tm->region_count = (int)region_count;
    tm->min_world_x = min_wx;
    tm->min_world_y = min_wy;
    tm->loaded = 1;

    /* read heightmap (appended after colors in the binary) */
    int32_t hm_min_x, hm_min_y;
    uint32_t hm_w, hm_h;
    int next = fgetc(f);
    if (next != EOF) {
        ungetc(next, f);
        if (!fc_read_exact(f, &hm_min_x, sizeof(hm_min_x), 1, path, "terrain heightmap min x") ||
            !fc_read_exact(f, &hm_min_y, sizeof(hm_min_y), 1, path, "terrain heightmap min y") ||
            !fc_read_exact(f, &hm_w, sizeof(hm_w), 1, path, "terrain heightmap width") ||
            !fc_read_exact(f, &hm_h, sizeof(hm_h), 1, path, "terrain heightmap height")) {
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        if (!(hm_w > 0 && hm_h > 0 && hm_w <= 4096 && hm_h <= 4096)) {
            fprintf(stderr, "%s: invalid terrain heightmap dimensions %ux%u\n",
                    path, hm_w, hm_h);
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        tm->hm_min_x = hm_min_x;
        tm->hm_min_y = hm_min_y;
        tm->hm_width = (int)hm_w;
        tm->hm_height = (int)hm_h;
        tm->heightmap = (float*)malloc(hm_w * hm_h * sizeof(float));
        if (!tm->heightmap ||
            !fc_read_exact(f, tm->heightmap, sizeof(float), hm_w * hm_h,
                           path, "terrain heightmap values")) {
            fc_asset_close(f);
            terrain_free(tm);
            return NULL;
        }
        fprintf(stderr, "terrain heightmap: %dx%d, origin (%d, %d)\n",
                tm->hm_width, tm->hm_height, tm->hm_min_x, tm->hm_min_y);
    }

    fc_asset_close(f);
    return tm;
}

/* shift terrain so world coordinates (wx, wy) become local (0, 0).
   offsets all mesh vertices and heightmap origin. must call before rendering. */
void terrain_offset(TerrainMesh* tm, int wx, int wy) {
    if (!tm || !tm->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;  /* Z = -world_y in our coord system */
    float* verts = tm->model.meshes[0].vertices;
    for (int i = 0; i < tm->vertex_count; i++) {
        verts[i * 3 + 0] -= dx;        /* X */
        verts[i * 3 + 2] += dz;        /* Z (negated world Y) */
    }
    UpdateMeshBuffer(tm->model.meshes[0], 0, verts,
                     tm->vertex_count * 3 * sizeof(float), 0);
    tm->min_world_x -= wx;
    tm->min_world_y -= wy;
    if (tm->heightmap) {
        tm->hm_min_x -= wx;
        tm->hm_min_y -= wy;
    }
    fprintf(stderr, "terrain_offset: shifted by (%d, %d), new origin (%d, %d)\n",
            wx, wy, tm->min_world_x, tm->min_world_y);
}

/* query terrain height at a world tile position (tile corner) */
float terrain_height_at(TerrainMesh* tm, int world_x, int world_y) {
    if (!tm || !tm->heightmap) return -2.0f;
    int lx = world_x - tm->hm_min_x;
    int ly = world_y - tm->hm_min_y;
    if (lx < 0 || lx >= tm->hm_width || ly < 0 || ly >= tm->hm_height)
        return -2.0f;
    return tm->heightmap[lx + ly * tm->hm_width];
}

void terrain_free(TerrainMesh* tm) {
    if (!tm) return;
    if (tm->loaded) {
        UnloadModel(tm->model);
    }
    free(tm->heightmap);
    free(tm);
}
