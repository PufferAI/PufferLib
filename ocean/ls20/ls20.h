// MIT License
//
// Copyright (c) 2026 ARC Prize Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Ported from ARC-AGI-3 ls20 revision 9607627b.

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define LS20_WIDTH 64
#define LS20_HEIGHT 64
#define LS20_BACKGROUND 3
#define LS20_DEFAULT_FPS 30
#define LS20_ACTION_COUNT 5
#define LS20_OBS_SIZE (LS20_WIDTH * LS20_HEIGHT)
#define LS20_LEVEL_COUNT 7
#define LS20_GRID_SIZE 12
#define LS20_MAX_ENERGY 42
#define LS20_MAX_SPRITES 160
#define LS20_MAX_GOALS 2
#define LS20_MAX_REFILLS 6
#define LS20_MAX_CONTROLS 3
#define LS20_MAX_MOVERS 3
#define LS20_MAX_PUSHERS 8
#define LS20_MAX_EXTRA_WALLS 1
#define LS20_SHAPE_COUNT 6
#define LS20_COLOR_COUNT 4
#define LS20_ROTATION_COUNT 4

enum {
    RESET = 0,
    ACTION1 = 1,  // Up
    ACTION2 = 2,  // Down
    ACTION3 = 3,  // Left
    ACTION4 = 4,  // Right
};

typedef enum {
    TAG_NONE,
    TAG_WALL,
    TAG_GOAL,
    TAG_REFILL,
    TAG_SHAPE_CONTROL,
    TAG_COLOR_CONTROL,
    TAG_ROTATION_CONTROL,
    TAG_PUSHER,
} SpriteTag;

typedef struct {
    const int8_t* pixels;
    int x;
    int y;
    int width;
    int height;
    int layer;
    int scale;
    int rotation;
    int recolor;
    int data;
    SpriteTag tag;
    bool visible;
    bool active;
} Sprite;

typedef struct {
    Sprite sprites[LS20_MAX_SPRITES];
    int count;
} Level;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int x;
    int y;
} Position;

typedef struct {
    int x;
    int y;
    int shape_index;
    int color_index;
    int rotation_index;
    int halo_rotation_index;
    bool hide_frame_when_complete;
} GoalSpec;

typedef enum {
    CONTROL_SHAPE,
    CONTROL_COLOR,
    CONTROL_ROTATION,
} ControlKind;

typedef enum {
    PATH_NONE,
    PATH_HORIZONTAL,
    PATH_VERTICAL,
    PATH_BOX,
} PathKind;

typedef struct {
    ControlKind kind;
    int x;
    int y;
    PathKind path;
    int path_x;
    int path_y;
    int path_width;
    int path_height;
} ControlSpec;

typedef struct {
    int x;
    int y;
    int dx;
    int dy;
} PusherSpec;

typedef struct {
    const char* walls[LS20_GRID_SIZE];
    Position player;
    int energy_decrement;
    int start_shape_index;
    int start_color_index;
    int start_rotation_index;
    bool fog;
    int goal_count;
    GoalSpec goals[LS20_MAX_GOALS];
    int refill_count;
    Position refills[LS20_MAX_REFILLS];
    int control_count;
    ControlSpec controls[LS20_MAX_CONTROLS];
    int pusher_count;
    PusherSpec pushers[LS20_MAX_PUSHERS];
    int extra_wall_count;
    Position extra_walls[LS20_MAX_EXTRA_WALLS];
} LevelSpec;

typedef struct {
    int base_sprite;
    int glyph_sprite;
    int frame_sprite;
    bool complete;
} Goal;

typedef struct {
    int sprite;
    int direction;
    PathKind path;
    int path_x;
    int path_y;
    int path_width;
    int path_height;
    bool can_undo;
    int undo_x;
    int undo_y;
    int undo_direction;
} MovingControl;

typedef struct {
    int sprite;
    int start_x;
    int start_y;
    int dx;
    int dy;
} Pusher;

typedef struct {
    Log log;
    int num_agents;
    int fps;
    bool reset_enabled;
    unsigned int rng;
    unsigned char* observations;
    unsigned char* action_mask;
    float* actions;
    float* rewards;
    float* terminals;

    Level level;
    int level_index;
    int levels_completed;
    int lives;
    int energy;
    int energy_decrement;
    int key_shape_index;
    int key_color_index;
    int key_rotation_index;
    int player_sprite;
    int goal_count;
    Goal goals[LS20_MAX_GOALS];
    int mover_count;
    MovingControl movers[LS20_MAX_MOVERS];
    int pusher_count;
    Pusher pushers[LS20_MAX_PUSHERS];
    int level_action_count;
    int episode_length;
    float episode_return;
    bool fog;
} Ls20;

static const int8_t WALL[25] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
static const int8_t WALL_OPEN_RIGHT[25] = {
    4, 4, 4, 4, -2, 4, 4, 4, 4, -2, 4, 4, 4,
    4, -2, 4, 4, 4, 4, -2, 4, 4, 4, 4, -2,
};
static const int8_t WALL_OPEN_TOP[25] = {
    -2, -2, -2, -2, -2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
static const int8_t WALL_OPEN_BOTTOM[25] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, -2, -2, -2, -2, -2,
};
static const int8_t WALL_OPEN_LEFT[25] = {
    -2, 4, 4, 4, 4, -2, 4, 4, 4, 4, -2, 4, 4,
    4, 4, -2, 4, 4, 4, 4, -2, 4, 4, 4, 4,
};

static const int8_t KEY_SHAPE_0[9] = {0, 0, -1, -1, 0, 0, 0, -1, 0};
static const int8_t KEY_SHAPE_1[9] = {-1, 0, -1, -1, 0, -1, 0, 0, 0};
static const int8_t KEY_SHAPE_2[9] = {0, -1, 0, 0, -1, 0, 0, 0, 0};
static const int8_t KEY_SHAPE_3[9] = {-1, 0, 0, 0, -1, 0, -1, 0, -1};
static const int8_t KEY_SHAPE_4[9] = {-1, 0, -1, 0, 0, -1, -1, 0, 0};
static const int8_t KEY_SHAPE_5[9] = {0, 0, 0, -1, -1, 0, 0, -1, 0};
static const int8_t* const KEY_SHAPES[LS20_SHAPE_COUNT] = {
    KEY_SHAPE_0, KEY_SHAPE_1, KEY_SHAPE_2,
    KEY_SHAPE_3, KEY_SHAPE_4, KEY_SHAPE_5,
};
static const int KEY_COLORS[LS20_COLOR_COUNT] = {12, 9, 14, 8};
static const int KEY_ROTATIONS[LS20_ROTATION_COUNT] = {0, 90, 180, 270};

static const int8_t GOAL_HALO[81] = {
    3, 3, -1, -1, -1, -1, -1, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
};
static const int8_t GOAL_FRAME[49] = {
    5, 5, 5, 5, 5, 5, 5,
    5, -1, -1, -1, -1, -1, 5,
    5, -1, -1, -1, -1, -1, 5,
    5, -1, -1, -1, -1, -1, 5,
    5, -1, -1, -1, -1, -1, 5,
    5, -1, -1, -1, -1, -1, 5,
    5, 5, 5, 5, 5, 5, 5,
};
static const int8_t GOAL_BASE[25] = {
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
};
static const int8_t PLAYER[25] = {
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
};
static const int8_t REFILL[9] = {11, 11, 11, 11, -1, 11, 11, 11, 11};
static const int8_t SHAPE_CONTROL[25] = {
    -2, -2, -2, -2, -2, -2, 0, -2, -2, -2,
    -2, -2, 0, 0, -2, -2, -2, 0, -2, -2,
    -2, -2, -2, -2, -2,
};
static const int8_t COLOR_CONTROL[25] = {
    -2, -2, -2, -2, -2, -2, 9, 14, 14, -2,
    -2, 9, 0, 8, -2, -2, 12, 12, 8, -2,
    -2, -2, -2, -2, -2,
};
static const int8_t ROTATION_CONTROL[25] = {
    -2, -2, -2, -2, -2, -2, -2, 0, -2, -2,
    -2, 1, 0, 0, -2, -2, -2, 1, -2, -2,
    -2, -2, -2, -2, -2,
};
static const int8_t PUSH_DOWN[25] = {
    1, 1, 1, 1, 1, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2,
};
static const int8_t PUSH_UP[25] = {
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    1, 1, 1, 1, 1,
};
static const int8_t PUSH_LEFT[25] = {
    -2, -2, -2, -2, 1, -2, -2, -2, -2, 1,
    -2, -2, -2, -2, 1, -2, -2, -2, -2, 1,
    -2, -2, -2, -2, 1,
};
static const int8_t PUSH_RIGHT[25] = {
    1, -2, -2, -2, -2, 1, -2, -2, -2, -2,
    1, -2, -2, -2, -2, 1, -2, -2, -2, -2,
    1, -2, -2, -2, -2,
};

// Coordinates are pixel anchors; shape, color, and rotation values are indices.
static const LevelSpec LEVELS[LS20_LEVEL_COUNT] = {
    {
        .walls = {
            "############", "############", "######.#####", "######.#####",
            "######.#####", "##........##", "##...#....##", "##...#....##",
            "###.##....##", "###.......##", "############", "############",
        },
        .player = {34, 45}, .energy_decrement = 1,
        .start_shape_index = 5, .start_color_index = 1, .start_rotation_index = 3,
        .goal_count = 1,
        .goals = {{34, 10, 5, 1, 0, 2, false}},
        .control_count = 1,
        .controls = {{CONTROL_ROTATION, 19, 30, PATH_NONE, 0, 0, 0, 0}},
    },
    {
        .walls = {
            "############", "###.......##", "#.........##", "#...#..#..##",
            "#...#..##..#", "##.###..#..#", "##.###..#.##", "##.##..##..#",
            "##.##..#...#", "########...#", "#######....#", "############",
        },
        .player = {29, 40}, .energy_decrement = 2,
        .start_shape_index = 5, .start_color_index = 1, .start_rotation_index = 0,
        .goal_count = 1,
        .goals = {{14, 40, 5, 1, 3, 0, false}},
        .refill_count = 2, .refills = {{15, 16}, {40, 51}},
        .control_count = 1,
        .controls = {{CONTROL_ROTATION, 49, 45, PATH_NONE, 0, 0, 0, 0}},
        .extra_wall_count = 1, .extra_walls = {{54, 34}},
    },
    {
        .walls = {
            "##########v#", ">......#...#", "#.###.##...#", "#.#....#...#",
            "#.#....###.#", "#.#........#", "#.#........#", "#.###.####.#",
            "#..#...###.#", "#..#...###.#", "####...###.#", "############",
        },
        .player = {9, 45}, .energy_decrement = 2,
        .start_shape_index = 5, .start_color_index = 0, .start_rotation_index = 0,
        .goal_count = 1,
        .goals = {{54, 50, 5, 1, 2, 0, false}},
        .refill_count = 2, .refills = {{35, 16}, {20, 31}},
        .control_count = 2,
        .controls = {
            {CONTROL_ROTATION, 49, 10, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_COLOR, 29, 45, PATH_NONE, 0, 0, 0, 0},
        },
        .pusher_count = 2,
        .pushers = {{54, 4, 0, 1}, {8, 5, 1, 0}},
    },
    {
        .walls = {
            "############", "#....##....#", "####.#..####", "###.....v..#",
            "##...>.....#", "#..###...<.#", "#..v.#.^...#", ">....###...#",
            "#....<...<.#", "##.....#...#", "####...##..#", "############",
        },
        .player = {54, 5}, .energy_decrement = 1,
        .start_shape_index = 4, .start_color_index = 2, .start_rotation_index = 0,
        .goal_count = 1,
        .goals = {{9, 5, 5, 1, 0, 1, false}},
        .refill_count = 2, .refills = {{35, 51}, {20, 16}},
        .control_count = 2,
        .controls = {
            {CONTROL_SHAPE, 24, 30, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_COLOR, 34, 30, PATH_NONE, 0, 0, 0, 0},
        },
        .pusher_count = 8,
        .pushers = {
            {19, 34, 0, 1}, {44, 19, 0, 1}, {39, 26, 0, -1},
            {45, 25, -1, 0}, {25, 40, -1, 0}, {45, 40, -1, 0},
            {33, 20, 1, 0}, {8, 35, 1, 0},
        },
    },
    {
        .walls = {
            "######v#####", "##...#...#.#", "#....#..##.#", "##.......#.#",
            "###.#>...#.#", "#...<....v.#", "#.#####^...<", "#....#.....#",
            "#......>...#", "##...#####.#", "###........#", "##########^#",
        },
        .player = {49, 40}, .energy_decrement = 2,
        .start_shape_index = 4, .start_color_index = 0, .start_rotation_index = 0,
        .goal_count = 1,
        .goals = {{54, 5, 0, 3, 2, 2, false}},
        .refill_count = 3, .refills = {{15, 46}, {45, 6}, {10, 11}},
        .control_count = 3,
        .controls = {
            {CONTROL_SHAPE, 19, 10, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_COLOR, 29, 25, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_ROTATION, 14, 35, PATH_HORIZONTAL, 14, 35, 11, 1},
        },
        .pusher_count = 8,
        .pushers = {
            {34, 4, 0, 1}, {49, 29, 0, 1}, {54, 51, 0, -1},
            {39, 26, 0, -1}, {55, 30, -1, 0}, {20, 25, -1, 0},
            {33, 20, 1, 0}, {43, 40, 1, 0},
        },
    },
    {
        .walls = {
            "#########v##", "#.......#..#", "#.......#..#", "#.#####...##",
            "#.#...#...<#", "#.......#..#", "#.#...#.##.#", "#.#####.##.#",
            "#.......##.#", "#.......##.#", "##.....###.#", "############",
        },
        .player = {24, 50}, .energy_decrement = 1,
        .start_shape_index = 0, .start_color_index = 2, .start_rotation_index = 0,
        .goal_count = 2,
        .goals = {
            {54, 50, 5, 1, 1, 0, false},
            {54, 35, 0, 3, 2, 0, true},
        },
        .refill_count = 3, .refills = {{40, 6}, {10, 46}, {10, 6}},
        .control_count = 3,
        .controls = {
            {CONTROL_SHAPE, 14, 10, PATH_HORIZONTAL, 14, 10, 21, 1},
            {CONTROL_COLOR, 24, 30, PATH_BOX, 19, 20, 11, 11},
            {CONTROL_ROTATION, 34, 40, PATH_HORIZONTAL, 14, 40, 21, 1},
        },
        .pusher_count = 2,
        .pushers = {{49, 4, 0, 1}, {50, 20, -1, 0}},
    },
    {
        .walls = {
            "############", "#.....#.#..#", "#...#...#..#", "#...##.v#..#",
            "#.###...#..#", "#.......#..#", "#.###...<..#", "#...#.^....#",
            "#...#.#....#", "##.##.###.##", "#####.##...#", "############",
        },
        .player = {19, 15}, .energy_decrement = 2,
        .start_shape_index = 1, .start_color_index = 0, .start_rotation_index = 0,
        .fog = true,
        .goal_count = 1,
        .goals = {{29, 50, 0, 3, 2, 0, false}},
        .refill_count = 6,
        .refills = {{30, 21}, {50, 6}, {15, 46}, {40, 6}, {55, 51}, {10, 6}},
        .control_count = 3,
        .controls = {
            {CONTROL_SHAPE, 19, 40, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_COLOR, 9, 40, PATH_NONE, 0, 0, 0, 0},
            {CONTROL_ROTATION, 54, 10, PATH_VERTICAL, 54, 5, 1, 29},
        },
        .pusher_count = 3,
        .pushers = {{39, 19, 0, 1}, {34, 31, 0, -1}, {40, 30, -1, 0}},
    },
};

static Sprite make_sprite(
    const int8_t* pixels, int width, int height, int x, int y, int layer,
    SpriteTag tag
) {
    return (Sprite) {
        .pixels = pixels,
        .x = x,
        .y = y,
        .width = width,
        .height = height,
        .layer = layer,
        .scale = 1,
        .rotation = 0,
        .recolor = -1,
        .tag = tag,
        .visible = true,
        .active = true,
    };
}

static int level_add(Level* level, Sprite sprite) {
    assert(level->count < LS20_MAX_SPRITES);
    int index = level->count;
    level->sprites[level->count++] = sprite;
    return index;
}

static int sprite_width(const Sprite* sprite) {
    int width = sprite->rotation % 180 == 0 ? sprite->width : sprite->height;
    return width * sprite->scale;
}

static int sprite_height(const Sprite* sprite) {
    int height = sprite->rotation % 180 == 0 ? sprite->height : sprite->width;
    return height * sprite->scale;
}

static int sprite_pixel(const Sprite* sprite, int x, int y) {
    x /= sprite->scale;
    y /= sprite->scale;

    int base_x = x;
    int base_y = y;
    if (sprite->rotation == 90) {
        base_x = y;
        base_y = sprite->height - 1 - x;
    } else if (sprite->rotation == 180) {
        base_x = sprite->width - 1 - x;
        base_y = sprite->height - 1 - y;
    } else if (sprite->rotation == 270) {
        base_x = sprite->width - 1 - y;
        base_y = x;
    }

    int pixel = sprite->pixels[base_y * sprite->width + base_x];
    if (pixel == 0 && sprite->recolor >= 0) {
        return sprite->recolor;
    }
    return pixel;
}

static bool sprites_overlap(const Sprite* first, const Sprite* second) {
    return first->active && second->active
        && first->x < second->x + sprite_width(second)
        && second->x < first->x + sprite_width(first)
        && first->y < second->y + sprite_height(second)
        && second->y < first->y + sprite_height(first);
}

static const int8_t* wall_pixels(char wall) {
    if (wall == '>') return WALL_OPEN_RIGHT;
    if (wall == '^') return WALL_OPEN_TOP;
    if (wall == 'v') return WALL_OPEN_BOTTOM;
    if (wall == '<') return WALL_OPEN_LEFT;
    return WALL;
}

static const int8_t* control_pixels(ControlKind kind) {
    if (kind == CONTROL_SHAPE) return SHAPE_CONTROL;
    if (kind == CONTROL_COLOR) return COLOR_CONTROL;
    return ROTATION_CONTROL;
}

static SpriteTag control_tag(ControlKind kind) {
    if (kind == CONTROL_SHAPE) return TAG_SHAPE_CONTROL;
    if (kind == CONTROL_COLOR) return TAG_COLOR_CONTROL;
    return TAG_ROTATION_CONTROL;
}

static const int8_t* pusher_pixels(int dx, int dy) {
    if (dy > 0) return PUSH_DOWN;
    if (dy < 0) return PUSH_UP;
    if (dx < 0) return PUSH_LEFT;
    return PUSH_RIGHT;
}

static void load_level(Ls20* env, int level_index) {
    assert(level_index >= 0 && level_index < LS20_LEVEL_COUNT);
    const LevelSpec* spec = &LEVELS[level_index];
    assert(spec->goal_count >= 0 && spec->goal_count <= LS20_MAX_GOALS);
    assert(spec->refill_count >= 0 && spec->refill_count <= LS20_MAX_REFILLS);
    assert(spec->control_count >= 0 && spec->control_count <= LS20_MAX_CONTROLS);
    assert(spec->pusher_count >= 0 && spec->pusher_count <= LS20_MAX_PUSHERS);
    assert(spec->extra_wall_count >= 0
        && spec->extra_wall_count <= LS20_MAX_EXTRA_WALLS);
    assert(spec->start_shape_index >= 0 && spec->start_shape_index < LS20_SHAPE_COUNT);
    assert(spec->start_color_index >= 0 && spec->start_color_index < LS20_COLOR_COUNT);
    assert(spec->start_rotation_index >= 0
        && spec->start_rotation_index < LS20_ROTATION_COUNT);
    memset(&env->level, 0, sizeof(env->level));
    memset(env->goals, 0, sizeof(env->goals));
    memset(env->movers, 0, sizeof(env->movers));
    memset(env->pushers, 0, sizeof(env->pushers));
    env->mover_count = 0;

    env->level_index = level_index;
    env->lives = 3;
    env->energy = LS20_MAX_ENERGY;
    env->energy_decrement = spec->energy_decrement;
    env->key_shape_index = spec->start_shape_index;
    env->key_color_index = spec->start_color_index;
    env->key_rotation_index = spec->start_rotation_index;
    env->fog = spec->fog;
    env->level_action_count = 0;

    for (int row = 0; row < LS20_GRID_SIZE; row++) {
        assert(strlen(spec->walls[row]) == LS20_GRID_SIZE);
        for (int col = 0; col < LS20_GRID_SIZE; col++) {
            char wall = spec->walls[row][col];
            assert(strchr(".#>^v<", wall) != NULL);
            if (wall == '.') continue;
            level_add(&env->level, make_sprite(
                wall_pixels(wall), 5, 5, 4 + 5 * col, 5 * row, -5,
                TAG_WALL
            ));
        }
    }
    for (int i = 0; i < spec->extra_wall_count; i++) {
        level_add(&env->level, make_sprite(
            WALL, 5, 5, spec->extra_walls[i].x, spec->extra_walls[i].y, -5,
            TAG_WALL
        ));
    }

    env->goal_count = spec->goal_count;
    for (int i = 0; i < spec->goal_count; i++) {
        const GoalSpec* goal_spec = &spec->goals[i];
        assert(goal_spec->shape_index >= 0 && goal_spec->shape_index < LS20_SHAPE_COUNT);
        assert(goal_spec->color_index >= 0 && goal_spec->color_index < LS20_COLOR_COUNT);
        assert(goal_spec->rotation_index >= 0
            && goal_spec->rotation_index < LS20_ROTATION_COUNT);
        assert(goal_spec->halo_rotation_index >= 0
            && goal_spec->halo_rotation_index < LS20_ROTATION_COUNT);
        Sprite halo = make_sprite(
            GOAL_HALO, 9, 9, goal_spec->x - 2, goal_spec->y - 2, -4,
            TAG_NONE
        );
        halo.rotation = KEY_ROTATIONS[goal_spec->halo_rotation_index];
        level_add(&env->level, halo);

        Sprite frame = make_sprite(
            GOAL_FRAME, 7, 7, goal_spec->x - 1, goal_spec->y - 1, -3,
            TAG_NONE
        );
        env->goals[i].frame_sprite = level_add(&env->level, frame);

        Sprite base = make_sprite(
            GOAL_BASE, 5, 5, goal_spec->x, goal_spec->y, -3,
            TAG_GOAL
        );
        base.data = i;
        env->goals[i].base_sprite = level_add(&env->level, base);

        Sprite glyph = make_sprite(
            KEY_SHAPES[goal_spec->shape_index], 3, 3,
            goal_spec->x + 1, goal_spec->y + 1, 0,
            TAG_NONE
        );
        glyph.rotation = KEY_ROTATIONS[goal_spec->rotation_index];
        glyph.recolor = KEY_COLORS[goal_spec->color_index];
        env->goals[i].glyph_sprite = level_add(&env->level, glyph);
    }

    for (int i = 0; i < spec->refill_count; i++) {
        Sprite refill = make_sprite(
            REFILL, 3, 3, spec->refills[i].x, spec->refills[i].y, -1,
            TAG_REFILL
        );
        level_add(&env->level, refill);
    }

    for (int i = 0; i < spec->control_count; i++) {
        const ControlSpec* control = &spec->controls[i];
        Sprite sprite = make_sprite(
            control_pixels(control->kind), 5, 5, control->x, control->y, -1,
            control_tag(control->kind)
        );
        int sprite_index = level_add(&env->level, sprite);
        if (control->path != PATH_NONE) {
            assert(env->mover_count < LS20_MAX_MOVERS);
            env->movers[env->mover_count++] = (MovingControl) {
                .sprite = sprite_index,
                .path = control->path,
                .path_x = control->path_x,
                .path_y = control->path_y,
                .path_width = control->path_width,
                .path_height = control->path_height,
            };
        }
    }

    env->pusher_count = spec->pusher_count;
    for (int i = 0; i < spec->pusher_count; i++) {
        const PusherSpec* pusher_spec = &spec->pushers[i];
        Sprite sprite = make_sprite(
            pusher_pixels(pusher_spec->dx, pusher_spec->dy), 5, 5,
            pusher_spec->x, pusher_spec->y, 0,
            TAG_PUSHER
        );
        int sprite_index = level_add(&env->level, sprite);
        env->pushers[i] = (Pusher) {
            .sprite = sprite_index,
            .start_x = pusher_spec->x,
            .start_y = pusher_spec->y,
            .dx = pusher_spec->dx,
            .dy = pusher_spec->dy,
        };
    }

    env->player_sprite = level_add(&env->level, make_sprite(
        PLAYER, 5, 5, spec->player.x, spec->player.y, 0,
        TAG_NONE
    ));
}

static void draw_sprite(unsigned char* frame, const Sprite* sprite) {
    if (!sprite->active || !sprite->visible) return;
    int width = sprite_width(sprite);
    int height = sprite_height(sprite);
    for (int y = 0; y < height; y++) {
        int frame_y = sprite->y + y;
        if (frame_y < 0 || frame_y >= LS20_HEIGHT) continue;
        for (int x = 0; x < width; x++) {
            int frame_x = sprite->x + x;
            if (frame_x < 0 || frame_x >= LS20_WIDTH) continue;
            int pixel = sprite_pixel(sprite, x, y);
            if (pixel >= 0) {
                frame[frame_y * LS20_WIDTH + frame_x] = (unsigned char)pixel;
            }
        }
    }
}

static void fill_rect(
    unsigned char* frame, int x, int y, int width, int height, unsigned char color
) {
    for (int row = y; row < y + height; row++) {
        for (int col = x; col < x + width; col++) {
            frame[row * LS20_WIDTH + col] = color;
        }
    }
}

static void draw_hud_frame(unsigned char* frame) {
    fill_rect(frame, 0, 0, 4, 52, 5);
    fill_rect(frame, 0, 52, 12, 1, 4);
    fill_rect(frame, 0, 53, 1, 10, 4);
    fill_rect(frame, 11, 53, 1, 10, 4);
    fill_rect(frame, 0, 63, 12, 1, 4);
    fill_rect(frame, 12, 60, 52, 4, 5);
}

static void draw_current_key(Ls20* env) {
    Sprite key = make_sprite(
        KEY_SHAPES[env->key_shape_index], 3, 3, 3, 55, 10,
        TAG_NONE
    );
    key.scale = 2;
    key.rotation = KEY_ROTATIONS[env->key_rotation_index];
    key.recolor = KEY_COLORS[env->key_color_index];
    draw_sprite(env->observations, &key);
}

static void draw_status(Ls20* env) {
    for (int i = 0; i < LS20_MAX_ENERGY; i++) {
        unsigned char color = LS20_MAX_ENERGY - i - 1 < env->energy ? 11 : 3;
        fill_rect(env->observations, 13 + i, 61, 1, 2, color);
    }
    for (int i = 0; i < 3; i++) {
        unsigned char color = env->lives > i ? 8 : 3;
        fill_rect(env->observations, 56 + 3 * i, 61, 2, 2, color);
    }
}

static void compute_observations(Ls20* env) {
    memset(env->observations, LS20_BACKGROUND, LS20_OBS_SIZE);
    for (int layer = -5; layer <= 10; layer++) {
        for (int i = 0; i < env->level.count; i++) {
            if (env->level.sprites[i].layer == layer) {
                draw_sprite(env->observations, &env->level.sprites[i]);
            }
        }
        if (layer == 0) draw_hud_frame(env->observations);
        if (layer == 1) fill_rect(env->observations, 1, 53, 10, 10, 5);
        if (layer == 10) draw_current_key(env);
    }

    if (env->fog) {
        const Sprite* player = &env->level.sprites[env->player_sprite];
        int center_x = 2 * player->x + 3;
        int center_y = 2 * player->y + 3;
        for (int y = 0; y < LS20_HEIGHT; y++) {
            for (int x = 0; x < LS20_WIDTH; x++) {
                int dx = 2 * x - center_x;
                int dy = 2 * y - center_y;
                if (dx * dx + dy * dy > 1600) {
                    env->observations[y * LS20_WIDTH + x] = 5;
                }
            }
        }
        draw_current_key(env);
    }
    draw_status(env);
}

static bool key_matches_goal(const Ls20* env, int goal_index) {
    const GoalSpec* goal = &LEVELS[env->level_index].goals[goal_index];
    return env->key_shape_index == goal->shape_index
        && env->key_color_index == goal->color_index
        && env->key_rotation_index == goal->rotation_index;
}

static bool any_unfinished_goal_matches(const Ls20* env) {
    for (int i = 0; i < env->goal_count; i++) {
        if (!env->goals[i].complete && key_matches_goal(env, i)) return true;
    }
    return false;
}

typedef struct {
    bool blocked;
    bool picked_up;
    bool feedback;
} Interaction;

static Interaction interact_at_destination(Ls20* env, int x, int y) {
    Interaction result = {0};
    for (int i = 0; i < env->level.count; i++) {
        Sprite* sprite = &env->level.sprites[i];
        if (!sprite->active || sprite->x < x || sprite->x >= x + 5
                || sprite->y < y || sprite->y >= y + 5) {
            continue;
        }

        if (sprite->tag == TAG_WALL) {
            result.blocked = true;
            break;
        }
        if (sprite->tag == TAG_GOAL) {
            if (!key_matches_goal(env, sprite->data)) {
                result.blocked = true;
                result.feedback = true;
            }
        } else if (sprite->tag == TAG_REFILL) {
            result.picked_up = true;
            env->energy = LS20_MAX_ENERGY;
            sprite->active = false;
        } else if (sprite->tag == TAG_SHAPE_CONTROL) {
            env->key_shape_index = (env->key_shape_index + 1) % LS20_SHAPE_COUNT;
            if (env->level_index == 0 && any_unfinished_goal_matches(env)) {
                result.feedback = true;
            }
        } else if (sprite->tag == TAG_COLOR_CONTROL) {
            env->key_color_index = (env->key_color_index + 1) % LS20_COLOR_COUNT;
            if (env->level_index == 0 && any_unfinished_goal_matches(env)) {
                result.feedback = true;
            }
        } else if (sprite->tag == TAG_ROTATION_CONTROL) {
            env->key_rotation_index = (env->key_rotation_index + 1) % LS20_ROTATION_COUNT;
            if (env->level_index == 0 && any_unfinished_goal_matches(env)) {
                result.feedback = true;
            }
        }
    }
    return result;
}

static bool path_contains(const MovingControl* mover, int x, int y) {
    int local_x = x - mover->path_x;
    int local_y = y - mover->path_y;
    if (local_x < 0 || local_y < 0
            || local_x >= mover->path_width || local_y >= mover->path_height) {
        return false;
    }
    if (mover->path == PATH_HORIZONTAL || mover->path == PATH_VERTICAL) {
        return true;
    }
    return local_x == 0 || local_y == 0
        || local_x == mover->path_width - 1
        || local_y == mover->path_height - 1;
}

static void step_moving_control(Ls20* env, MovingControl* mover) {
    static const int DX[4] = {0, 1, 0, -1};
    static const int DY[4] = {1, 0, -1, 0};
    Sprite* sprite = &env->level.sprites[mover->sprite];
    int directions[4] = {
        mover->direction,
        (mover->direction + 3) % 4,
        (mover->direction + 1) % 4,
        (mover->direction + 2) % 4,
    };

    mover->can_undo = false;
    for (int i = 0; i < 4; i++) {
        int direction = directions[i];
        int next_x = sprite->x + 5 * DX[direction];
        int next_y = sprite->y + 5 * DY[direction];
        if (!path_contains(mover, next_x, next_y)) continue;

        mover->undo_x = sprite->x;
        mover->undo_y = sprite->y;
        mover->undo_direction = mover->direction;
        mover->can_undo = true;
        mover->direction = direction;
        sprite->x = next_x;
        sprite->y = next_y;
        return;
    }
}

static void step_moving_controls(Ls20* env) {
    for (int i = 0; i < env->mover_count; i++) {
        step_moving_control(env, &env->movers[i]);
    }
}

static void undo_moving_controls(Ls20* env) {
    for (int i = 0; i < env->mover_count; i++) {
        MovingControl* mover = &env->movers[i];
        if (!mover->can_undo) continue;
        Sprite* sprite = &env->level.sprites[mover->sprite];
        sprite->x = mover->undo_x;
        sprite->y = mover->undo_y;
        mover->direction = mover->undo_direction;
        mover->can_undo = false;
    }
}

static bool wall_anchor_exists(const Ls20* env, int x, int y) {
    for (int i = 0; i < env->level.count; i++) {
        const Sprite* sprite = &env->level.sprites[i];
        if (sprite->tag == TAG_WALL && sprite->x == x && sprite->y == y) return true;
    }
    const LevelSpec* spec = &LEVELS[env->level_index];
    for (int i = 0; i < spec->goal_count; i++) {
        if (spec->goals[i].x == x && spec->goals[i].y == y) return true;
    }
    return false;
}

static int pusher_distance(const Ls20* env, const Pusher* pusher) {
    int wall_x = pusher->start_x + pusher->dx;
    int wall_y = pusher->start_y + pusher->dy;
    for (int distance = 1; distance < 12; distance++) {
        int x = wall_x + 5 * pusher->dx * distance;
        int y = wall_y + 5 * pusher->dy * distance;
        if (wall_anchor_exists(env, x, y)) return distance - 1;
    }
    return 0;
}

static bool apply_pushers(Ls20* env) {
    Sprite* player = &env->level.sprites[env->player_sprite];
    for (int i = 0; i < env->pusher_count; i++) {
        Pusher* pusher = &env->pushers[i];
        Sprite* sprite = &env->level.sprites[pusher->sprite];
        if (!sprites_overlap(sprite, player)) continue;
        int distance = pusher_distance(env, pusher);
        if (distance <= 0) continue;
        player->x += 5 * pusher->dx * distance;
        player->y += 5 * pusher->dy * distance;
        return true;
    }
    return false;
}

static bool complete_matching_goals(Ls20* env) {
    Sprite* player = &env->level.sprites[env->player_sprite];
    const LevelSpec* spec = &LEVELS[env->level_index];
    for (int i = 0; i < env->goal_count; i++) {
        Goal* goal = &env->goals[i];
        const GoalSpec* goal_spec = &spec->goals[i];
        if (goal->complete || player->x != goal_spec->x || player->y != goal_spec->y
                || !key_matches_goal(env, i)) {
            continue;
        }
        goal->complete = true;
        env->level.sprites[goal->base_sprite].active = false;
        env->level.sprites[goal->glyph_sprite].active = false;
        if (goal_spec->hide_frame_when_complete) {
            env->level.sprites[goal->frame_sprite].visible = false;
        }
    }
    for (int i = 0; i < env->goal_count; i++) {
        if (!env->goals[i].complete) return false;
    }
    return true;
}

static void reset_life(Ls20* env) {
    int lives = env->lives;
    int level_action_count = env->level_action_count;
    load_level(env, env->level_index);
    env->lives = lives;
    env->level_action_count = level_action_count;
}

static void reset_gameplay(Ls20* env, bool reset_episode) {
    env->levels_completed = 0;
    if (reset_episode) {
        env->episode_length = 0;
        env->episode_return = 0.0f;
    }
    load_level(env, 0);
}

static void add_log(Ls20* env) {
    env->log.perf += (float)env->levels_completed / (float)LS20_LEVEL_COUNT;
    env->log.score += (float)env->levels_completed;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.n += 1.0f;
}

void c_reset(Ls20* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    if (env->action_mask != NULL) {
        memset(env->action_mask, 1, LS20_ACTION_COUNT);
        env->action_mask[RESET] = (unsigned char)env->reset_enabled;
    }
    reset_gameplay(env, true);
    compute_observations(env);
}

static void handle_reset_action(Ls20* env) {
    if (env->level_action_count == 0) {
        reset_gameplay(env, false);
    } else {
        load_level(env, env->level_index);
    }
}

static void finish_episode(Ls20* env) {
    env->terminals[0] = 1.0f;
    add_log(env);
    reset_gameplay(env, true);
    compute_observations(env);
}

void c_step(Ls20* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->episode_length++;

    float raw_action = env->actions[0];
    if (!(raw_action >= (float)RESET && raw_action <= (float)ACTION4)) {
        compute_observations(env);
        return;
    }
    int action = (int)raw_action;
    if (raw_action != (float)action) {
        compute_observations(env);
        return;
    }
    if (action == RESET) {
        if (env->reset_enabled) handle_reset_action(env);
        compute_observations(env);
        return;
    }

    env->level_action_count++;
    int dx = 0;
    int dy = 0;
    if (action == ACTION1) dy = -1;
    else if (action == ACTION2) dy = 1;
    else if (action == ACTION3) dx = -1;
    else if (action == ACTION4) dx = 1;
    else {
        compute_observations(env);
        return;
    }

    step_moving_controls(env);
    Sprite* player = &env->level.sprites[env->player_sprite];
    int destination_x = player->x + 5 * dx;
    int destination_y = player->y + 5 * dy;
    Interaction interaction = interact_at_destination(env, destination_x, destination_y);
    if (interaction.blocked) {
        undo_moving_controls(env);
    } else {
        player->x = destination_x;
        player->y = destination_y;
    }

    if (interaction.feedback) {
        compute_observations(env);
        return;
    }

    bool exhausted = false;
    if (!interaction.picked_up) {
        env->energy -= env->energy_decrement;
        exhausted = env->energy < 0;
    }

    if (!exhausted && apply_pushers(env)) {
        (void)interact_at_destination(env, player->x, player->y);
        compute_observations(env);
        return;
    }

    if (complete_matching_goals(env)) {
        env->levels_completed++;
        env->rewards[0] = 1.0f;
        env->episode_return += 1.0f;
        if (env->level_index == LS20_LEVEL_COUNT - 1) {
            finish_episode(env);
        } else {
            load_level(env, env->level_index + 1);
            compute_observations(env);
        }
        return;
    }

    if (exhausted) {
        env->lives--;
        if (env->lives == 0) {
            finish_episode(env);
        } else {
            reset_life(env);
            compute_observations(env);
        }
        return;
    }

    compute_observations(env);
}

void c_render(Ls20* env) {
    static const Color PALETTE[16] = {
        {255, 255, 255, 255}, {204, 204, 204, 255}, {153, 153, 153, 255},
        {102, 102, 102, 255}, {51, 51, 51, 255},    {0, 0, 0, 255},
        {229, 58, 163, 255},  {255, 123, 204, 255}, {249, 60, 49, 255},
        {30, 147, 255, 255},  {136, 216, 241, 255}, {255, 220, 0, 255},
        {255, 133, 27, 255},  {146, 18, 49, 255},   {79, 204, 48, 255},
        {163, 86, 214, 255},
    };
    const int scale = 10;
    if (!IsWindowReady()) {
        InitWindow(LS20_WIDTH * scale, LS20_HEIGHT * scale, "PufferLib LS20");
        SetTargetFPS(env->fps > 0 ? env->fps : LS20_DEFAULT_FPS);
    }
    if (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }
    BeginDrawing();
    for (int y = 0; y < LS20_HEIGHT; y++) {
        for (int x = 0; x < LS20_WIDTH; x++) {
            unsigned char color = env->observations[y * LS20_WIDTH + x];
            Color pixel = color < 16 ? PALETTE[color] : PALETTE[5];
            DrawRectangle(x * scale, y * scale, scale, scale, pixel);
        }
    }
    EndDrawing();
}

void c_close(Ls20* env) {
    (void)env;
    if (IsWindowReady()) CloseWindow();
}
