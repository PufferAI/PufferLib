#ifndef FC_MINIMAP_H
#define FC_MINIMAP_H

#include "fc_types.h"
#include "raylib.h"

#define FC_MINIMAP_DISPLAY_SIZE 152
#define FC_MINIMAP_DISPLAY_CENTER 76.0f
#define FC_MINIMAP_DISPLAY_RADIUS 75.0f
#define FC_MINIMAP_DISPLAY_PIXELS_PER_TILE 3.5f
#define FC_MINIMAP_SCENE_SIZE 512
#define FC_MINIMAP_SCENE_PIXELS_PER_TILE 4.0f
#define FC_MINIMAP_SCENE_BORDER_TILES 32.0f

typedef struct FcMinimapScene {
    Color* pixels;
    int ready;
} FcMinimapScene;

int fc_minimap_scene_load_pixels(FcMinimapScene* scene,
                                 const Color* pixels, int width, int height);
void fc_minimap_scene_free(FcMinimapScene* scene);

void fc_minimap_render(const FcMinimapScene* scene, float player_x,
                       float player_y, float camera_yaw, Color* output);

Vector2 fc_minimap_rotate_offset(float dx, float dy, float camera_yaw);
int fc_minimap_click_to_tile(float map_x, float map_y, float player_x,
                             float player_y, float camera_yaw,
                             int* tile_x, int* tile_y);

#endif /* FC_MINIMAP_H */
