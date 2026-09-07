#ifndef FC_TERRAIN_LOADER_H
#define FC_TERRAIN_LOADER_H

#include "raylib.h"

typedef struct {
    Model model;
    int vertex_count;
    int region_count;
    int min_world_x;
    int min_world_y;
    int loaded;
    float *heightmap;
    int hm_min_x;
    int hm_min_y;
    int hm_width;
    int hm_height;
} TerrainMesh;

TerrainMesh *terrain_load(const char *path);
void terrain_offset(TerrainMesh *tm, int wx, int wy);
float terrain_height_at(TerrainMesh *tm, int world_x, int world_y);
void terrain_free(TerrainMesh *tm);

#endif
