/* Overcooked Types: Constants, enums, and struct definitions.
 */

#ifndef OVERCOOKED_TYPES_H
#define OVERCOOKED_TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "raylib.h"

// Tile types
#define EMPTY 0
#define COUNTER 1
#define STOVE 2
#define CUTTING_BOARD 3
#define INGREDIENT_BOX 4
#define SERVING_AREA 5
#define WALL 6
#define PLATE_BOX 7
#define AGENT 8

// Item types
#define NO_ITEM 10
#define TOMATO 11
#define ONION 12
#define PLATE 13
#define SOUP 14
#define PLATED_SOUP 15

// Cooking states
#define NOT_COOKING 0
#define COOKING 1
#define COOKED 2

// Cooking parameters
#define COOKING_TIME 20
#define MAX_INGREDIENTS 3

// Actions
#define ACTION_NOOP 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_INTERACT 5

// Agent states
#define AGENT_EMPTY_HANDED 0
#define AGENT_HOLDING_ITEM 1

#define MAX_SPAWN_POSITIONS 8

typedef enum {
    LAYOUT_CRAMPED_ROOM = 0,
    LAYOUT_ASYMMETRIC_ADVANTAGES = 1,
    LAYOUT_FORCED_COORDINATION = 2,
    LAYOUT_COORDINATION_RING = 3,
    LAYOUT_COUNTER_CIRCUIT = 4,
    LAYOUT_COUNT
} LayoutType;

typedef struct {
    const char* name;
    int width;
    int height;
    const char* grid;
    int spawn_positions[MAX_SPAWN_POSITIONS];
    int num_spawns;
} LayoutInfo;

typedef struct {
    float dish_served_whole_team;
    float dish_served_agent;
    float pot_started;
    float ingredient_added;
    float ingredient_picked;
    float plate_picked;
    float soup_plated;
    float wrong_dish_served;
    float step_penalty;
} RewardConfig;

typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float dishes_served; // Number of dishes successfully served
    float correct_dishes; // Number of correct 3-onion dishes
    float wrong_dishes; // Number of wrong dishes submitted
    float ingredients_picked; // Total ingredients picked up
    float pots_started; // Number of cooking sessions started
    float items_dropped; // Number of items dropped/placed
    float agent_collisions; // Number of times agents tried to move to same spot
    float n; // Required as the last field
} Log;

typedef struct {
    Texture2D floor;
    Texture2D counter;
    Texture2D pot;
    Texture2D serve;
    Texture2D onions_box;
    Texture2D tomatoes_box;
    Texture2D dishes_box;
    Texture2D wall;

    Texture2D onion;
    Texture2D tomato;
    Texture2D dish;
    Texture2D soup_onion;
    Texture2D soup_tomato;

    Texture2D soup_onion_cooking_1;
    Texture2D soup_onion_cooking_2;
    Texture2D soup_onion_cooking_3;
    Texture2D soup_onion_cooked;
    Texture2D soup_tomato_cooking_1;
    Texture2D soup_tomato_cooking_2;
    Texture2D soup_tomato_cooking_3;
    Texture2D soup_tomato_cooked;

    Texture2D chef_north;
    Texture2D chef_south;
    Texture2D chef_east;
    Texture2D chef_west;
    Texture2D chef_north_onion;
    Texture2D chef_south_onion;
    Texture2D chef_east_onion;
    Texture2D chef_west_onion;
    Texture2D chef_north_tomato;
    Texture2D chef_south_tomato;
    Texture2D chef_east_tomato;
    Texture2D chef_west_tomato;
    Texture2D chef_north_dish;
    Texture2D chef_south_dish;
    Texture2D chef_east_dish;
    Texture2D chef_west_dish;
    Texture2D chef_north_soup_onion;
    Texture2D chef_south_soup_onion;
    Texture2D chef_east_soup_onion;
    Texture2D chef_west_soup_onion;
    Texture2D chef_north_soup_tomato;
    Texture2D chef_south_soup_tomato;
    Texture2D chef_east_soup_tomato;
    Texture2D chef_west_soup_tomato;

    Texture2D soup_onion_dish;
    Texture2D soup_tomato_dish;
} Client;

typedef struct __attribute__((aligned(32))) {
    float x;
    float y;
    int facing_direction;
    int held_item;
    int held_soup_onions;
    int held_soup_tomatoes;
    int held_soup_total;
    int ticks_since_reward;
} Agent;

typedef struct __attribute__((aligned(32))) {
    int x;
    int y;
    int type;
    int state;
    int num_onions;
    int num_tomatoes;
    int total_ingredients;
} Item;

typedef struct {
    int cooking_state;      // NOT_COOKING, COOKING, COOKED
    int cooking_progress;   // Steps since cooking started
    int ingredient_types[MAX_INGREDIENTS];  // Types of ingredients added
    int ingredient_count;   // Number of ingredients in pot
    int num_onions;        // Count of onions
    int num_tomatoes;      // Count of tomatoes
} CookingPot;

// Cache for static tile positions (computed once at init, never changes)
typedef struct {
    // Static tile positions stored as x,y pairs: [x0, y0, x1, y1, ...]
    int ingredient_box_positions[20];  // Max 10 ingredient boxes
    int ingredient_box_count;
    int plate_box_positions[20];       // Max 10 plate boxes
    int plate_box_count;
    int serving_area_positions[20];    // Max 10 serving areas
    int serving_area_count;
    int stove_positions[20];           // Max 10 stoves
    int stove_count;
    int counter_positions[100];        // Max 50 counters
    int counter_count;

    // Precomputed normalization factors
    float inv_width;   // 1.0f / width
    float inv_height;  // 1.0f / height
} StaticCache;

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    LayoutType layout_id;
    char* grid;
    Item* items;  // Dynamic items in the kitchen
    int num_items;
    int max_items;
    Agent* agents;  // Array of agents
    int num_agents;
    uint64_t agent_position_mask;  // Bit (y * width + x) set if agent present
    CookingPot* cooking_pots;  // Array of cooking pots (one per stove)
    int num_stoves;
    int* pot_index_grid;  // Maps grid cell to pot index (-1 if not a stove)
    int* item_grid;       // Maps grid cell to item index (-1 if empty)
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    float* terminals; // Required. We don't yet have truncations as standard yet
    int width;
    int height;
    int grid_size;
    RewardConfig rewards_config;
    int observation_size;
    StaticCache cache;  // Cached static tile positions for O(1) lookup
    unsigned int rng;
} Overcooked;

// Grid layout
static const char CRAMPED_ROOM[5][5] = {
    {'6', '1', '2', '1', '6'},
    {'4', ' ', ' ', ' ', '4'},
    {'1', ' ', ' ', ' ', '1'},
    {'1', ' ', ' ', ' ', '1'},
    {'6', '7', '1', '5', '6'}
};

static const char ASYMMETRIC_ADVANTAGES[5][9] = {
    {'6','1','6','6','6','6','6','1','6'},
    {'4',' ','1','5','6','4','1',' ','5'},
    {'1',' ',' ',' ','2',' ',' ',' ','1'},
    {'1',' ',' ',' ','2',' ',' ',' ','1'},
    {'6','1','1','7','6','7','1','1','6'}
};

static const char COORDINATION_RING[5][5] = {
    {'6', '1', '1', '2', '6'},
    {'1', ' ', ' ', ' ', '2'},
    {'7', ' ', '1', ' ', '1'},
    {'4', ' ', ' ', ' ', '1'},
    {'6', '4', '5', '1', '6'}
};

static const char FORCED_COORDINATION[5][5] = {
    {'6', '1', '6', '2', '6'},
    {'4', ' ', '1', ' ', '2'},
    {'4', ' ', '1', ' ', '1'},
    {'7', ' ', '1', ' ', '1'},
    {'6', '1', '6', '5', '6'}
};

static const char COUNTER_CIRCUIT[5][8] = {
    {'6','1','1','2','2','1','1','6'},
    {'1',' ',' ',' ',' ',' ',' ','1'},
    {'7',' ','1','1','1','1',' ','5'},
    {'1',' ',' ',' ',' ',' ',' ','1'},
    {'6','1','1','4','4','1','1','6'}
};

static const LayoutInfo LAYOUTS[LAYOUT_COUNT] = {
    {
        "cramped_room",
        5, 5,
        (const char*)CRAMPED_ROOM,
        {1, 2, 3, 2},
        2
    },
    {
        "asymmetric_advantages",
        9, 5,
        (const char*)ASYMMETRIC_ADVANTAGES,
        {1, 2, 7, 2},
        2
    },
    {
        "forced_coordination",
        5, 5,
        (const char*)FORCED_COORDINATION,
        {1, 2, 3, 2},
        2
    },
    {
        "coordination_ring",
        5, 5,
        (const char*)COORDINATION_RING,
        {1, 2, 3, 2},
        2
    },
    {
        "counter_circuit",
        8, 5,
        (const char*)COUNTER_CIRCUIT,
        {1, 1, 6, 3},
        2
    }
};

static inline const LayoutInfo* get_layout_info(LayoutType id) {
    if (id < 0 || id >= LAYOUT_COUNT) return &LAYOUTS[0];
    return &LAYOUTS[id];
}

static inline char get_layout_tile(const LayoutInfo* info, int x, int y) {
    return info->grid[y * info->width + x];
}

static inline LayoutType get_layout_by_name(const char* name) {
    for (int i = 0; i < LAYOUT_COUNT; i++) {
        if (strcmp(LAYOUTS[i].name, name) == 0) return (LayoutType)i;
    }
    return LAYOUT_CRAMPED_ROOM;
}

#endif // OVERCOOKED_TYPES_H
