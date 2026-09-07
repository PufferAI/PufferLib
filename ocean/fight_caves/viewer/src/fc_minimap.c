#include "fc_minimap.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int fc_minimap_scene_load_pixels(FcMinimapScene* scene,
                                 const Color* pixels, int width, int height) {
    if (!scene || !pixels || width != FC_MINIMAP_SCENE_SIZE ||
        height != FC_MINIMAP_SCENE_SIZE) {
        return 0;
    }

    size_t count = (size_t)width * (size_t)height;
    Color* copy = (Color*)malloc(count * sizeof(*copy));
    if (!copy) return 0;
    memcpy(copy, pixels, count * sizeof(*copy));

    fc_minimap_scene_free(scene);
    scene->pixels = copy;
    scene->ready = 1;
    return 1;
}

void fc_minimap_scene_free(FcMinimapScene* scene) {
    if (!scene) return;
    free(scene->pixels);
    memset(scene, 0, sizeof(*scene));
}

Vector2 fc_minimap_rotate_offset(float dx, float dy, float camera_yaw) {
    float sine = sinf(camera_yaw);
    float cosine = cosf(camera_yaw);
    return (Vector2){
        dx * cosine + dy * sine,
        -dx * sine + dy * cosine,
    };
}

static Color sample_source(const FcMinimapScene* scene, float x, float y) {
    const Color blank = {0, 0, 0, 0};
    if (!scene || !scene->ready || !scene->pixels || x < 0.0f || y < 0.0f ||
        x >= (float)FC_MINIMAP_SCENE_SIZE ||
        y >= (float)FC_MINIMAP_SCENE_SIZE) {
        return blank;
    }

    int source_x = (int)floorf(x);
    int source_y = (int)floorf(y);
    return scene->pixels[source_x + source_y * FC_MINIMAP_SCENE_SIZE];
}

void fc_minimap_render(const FcMinimapScene* scene, float player_x,
                       float player_y, float camera_yaw, Color* output) {
    if (!output) return;
    const Color blank = {0, 0, 0, 0};
    float sine = sinf(camera_yaw);
    float cosine = cosf(camera_yaw);
    float player_source_x = 0.0f;
    float player_source_y = 0.0f;
    float source_scale =
        FC_MINIMAP_SCENE_PIXELS_PER_TILE /
        FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    if (scene && scene->ready) {
        player_source_x =
            (player_x + FC_MINIMAP_SCENE_BORDER_TILES) *
            FC_MINIMAP_SCENE_PIXELS_PER_TILE;
        player_source_y =
            (float)FC_MINIMAP_SCENE_SIZE -
            (player_y + FC_MINIMAP_SCENE_BORDER_TILES) *
            FC_MINIMAP_SCENE_PIXELS_PER_TILE;
    }

    for (int y = 0; y < FC_MINIMAP_DISPLAY_SIZE; y++) {
        for (int x = 0; x < FC_MINIMAP_DISPLAY_SIZE; x++) {
            float screen_x = (float)x - FC_MINIMAP_DISPLAY_CENTER;
            float screen_y = (float)y - FC_MINIMAP_DISPLAY_CENTER;
            int output_index = x + y * FC_MINIMAP_DISPLAY_SIZE;
            if (screen_x * screen_x + screen_y * screen_y >
                FC_MINIMAP_DISPLAY_RADIUS * FC_MINIMAP_DISPLAY_RADIUS) {
                output[output_index] = blank;
                continue;
            }

            float source_x = player_source_x +
                (screen_x * cosine + screen_y * sine) * source_scale;
            float source_y = player_source_y +
                (-screen_x * sine + screen_y * cosine) * source_scale;
            output[output_index] = sample_source(scene, source_x, source_y);
        }
    }
}

int fc_minimap_click_to_tile(float map_x, float map_y, float player_x,
                             float player_y, float camera_yaw,
                             int* tile_x, int* tile_y) {
    if (!tile_x || !tile_y) return 0;
    float screen_x = map_x - FC_MINIMAP_DISPLAY_CENTER;
    float screen_y = map_y - FC_MINIMAP_DISPLAY_CENTER;
    if (screen_x * screen_x + screen_y * screen_y >
        FC_MINIMAP_DISPLAY_RADIUS * FC_MINIMAP_DISPLAY_RADIUS) {
        return 0;
    }

    float sine = sinf(camera_yaw);
    float cosine = cosf(camera_yaw);
    float dx = (screen_x * cosine + screen_y * sine) /
               FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    float dy = (screen_x * sine - screen_y * cosine) /
               FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    int x = (int)floorf(player_x + dx);
    int y = (int)floorf(player_y + dy);
    if (x < 0 || y < 0 || x >= FC_ARENA_WIDTH || y >= FC_ARENA_HEIGHT)
        return 0;
    *tile_x = x;
    *tile_y = y;
    return 1;
}
