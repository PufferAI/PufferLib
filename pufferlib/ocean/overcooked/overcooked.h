/* Overcooked: a single-agent cooking coordination environment.
 * Agent can walk around, pick up items, and put down items.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Grid tile types
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
#define NO_ITEM 0
#define TOMATO 1
#define ONION 2
#define PLATE 3
#define SOUP 4

// Agent actions
#define ACTION_NOOP 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_INTERACT 5  // Pick up / Put down / Use station

// Agent states
#define AGENT_EMPTY_HANDED 0
#define AGENT_HOLDING_ITEM 1

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float dishes_served; // Number of dishes successfully served
    float n; // Required as the last field 
} Log;

typedef struct {
    Texture2D tiles;
    Texture2D items;
    Texture2D agent;
} Client;

typedef struct {
    float x;
    float y;
    int held_item;  // Item type the agent is holding (NO_ITEM if empty)
    int facing_direction;  // 0=up, 1=down, 2=left, 3=right
} Agent;

typedef struct {
    int type;
    float x;
    float y;
    int state;  // For items that can change state (e.g., cooking progress)
} Item;

// Required that you have some struct for your env
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    
    // Game state
    char* grid;  // Kitchen layout (static tiles)
    Item* items;  // Dynamic items in the kitchen
    int num_items;
    int max_items;
    Agent agent;
    
    // Required arrays
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    
    // Environment parameters
    int width;
    int height;
    int max_steps;
    int current_step;
    int grid_size;  // For rendering
    
    // Game parameters
    float reward_dish_served;
    float reward_step_penalty;
    int observation_size;
} Overcooked;


// From overcooked-ai repo; 5x5
static const char CRAMPED_ROOM[5][5] = {
    {'1', '1', '2', '1', '1'},
    {'4', ' ', ' ', ' ', '4'},
    {'1', ' ', ' ', ' ', '1'},
    {'1', ' ', ' ', ' ', '1'},
    {'1', '7', '1', '5', '1'}
};

static void parse_grid(Overcooked* env) {
    for (int y = 0; y < env->height && y < 5; y++) {
        for (int x = 0; x < env->width && x < 5; x++) {
            char tile = CRAMPED_ROOM[y][x];
            int idx = y * env->width + x;
            switch (tile) {
                case '#': env->grid[idx] = WALL; break;
                case '1': env->grid[idx] = COUNTER; break;
                case '2': env->grid[idx] = STOVE; break;
                case '3': env->grid[idx] = CUTTING_BOARD; break;
                case '4': env->grid[idx] = INGREDIENT_BOX; break;
                case '5': env->grid[idx] = SERVING_AREA; break;
                case '7': env->grid[idx] = PLATE_BOX; break;
                default: env->grid[idx] = EMPTY; break;
            }
        }
    }
}

static void init(Overcooked* env) {
    env->grid = calloc(env->width * env->height, sizeof(char));
    env->max_items = 20;
    env->items = calloc(env->max_items, sizeof(Item));
    env->num_items = 0;
    parse_grid(env);
    env->client = NULL;
}

static void compute_observations(Overcooked* env) {
    for (int i = 0; i < env->observation_size; i++) {
        env->observations[i] = 0.0f;
    }
}

static void handle_interaction(Overcooked* env) {
    // Get position agent is facing
    int target_x = env->agent.x;
    int target_y = env->agent.y;
    
    switch (env->agent.facing_direction) {
        case 0: target_y -= 1; break; // Up
        case 1: target_y += 1; break; // Down
        case 2: target_x -= 1; break; // Left
        case 3: target_x += 1; break; // Right
    }
    
    // Check bounds
    if (target_x < 0 || target_x >= env->width || target_y < 0 || target_y >= env->height) {
        return;
    }
    
    int tile = env->grid[target_y * env->width + target_x];
    Item* item = get_item_at(env, target_x, target_y);
    
    // If agent is holding something
    if (env->agent.held_item != NO_ITEM) {
        // Can only put down on empty counters or specific stations
        if ((tile == COUNTER || tile == CUTTING_BOARD || tile == STOVE) && item == NULL) {
            // Put down the item
            add_item(env, env->agent.held_item, target_x, target_y);
            env->agent.held_item = NO_ITEM;
        }
    }
    // If agent is empty handed
    else {
        // Pick up item if there is one
        if (item != NULL) {
            env->agent.held_item = item->type;
            remove_item(env, target_x, target_y);
        }
        // Special case: get new ingredients from ingredient box
        else if (tile == INGREDIENT_BOX) {
            env->agent.held_item = ONION; // Always gives onions for now
        }
        // Special case: get plates from plate box
        else if (tile == PLATE_BOX) {
            env->agent.held_item = PLATE;
        }
    }
}

static int is_valid_position(Overcooked* env, int x, int y) {
    if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
        return 0;
    }
    return env->grid[y * env->width + x] == EMPTY;
}

static Item* get_item_at(Overcooked* env, int x, int y) {
    for (int i = 0; i < env->num_items; i++) {
        if ((int)env->items[i].x == x && (int)env->items[i].y == y) {
            return &env->items[i];
        }
    }
    return NULL;
}

static void add_item(Overcooked* env, int type, int x, int y) {
    if (env->num_items < env->max_items) {
        env->items[env->num_items].type = type;
        env->items[env->num_items].x = x;
        env->items[env->num_items].y = y;
        env->items[env->num_items].state = 0;
        env->num_items++;
    }
}

static void remove_item(Overcooked* env, int x, int y) {
    for (int i = 0; i < env->num_items; i++) {
        if ((int)env->items[i].x == x && (int)env->items[i].y == y) {
            for (int j = i; j < env->num_items - 1; j++) {
                env->items[j] = env->items[j + 1];
            }
            env->num_items--;
            break;
        }
    }
}

void c_reset(Overcooked* env) {
    env->current_step = 0;
    env->num_items = 0;
    parse_grid(env);
    
    env->agent.x = 2;
    env->agent.y = 2;
    env->agent.held_item = NO_ITEM;
    env->agent.facing_direction = 0;
    
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    
    compute_observations(env);
    
    env->log.episode_length = 0;
    env->log.episode_return = 0;
    env->log.dishes_served = 0;
}

void c_step(Overcooked* env) {
    int action = env->actions[0];
    env->rewards[0] = env->reward_step_penalty;
    
    int new_x = env->agent.x;
    int new_y = env->agent.y;
    
    switch (action) {
        case ACTION_UP:    new_y -= 1; env->agent.facing_direction = 0; break;
        case ACTION_DOWN:  new_y += 1; env->agent.facing_direction = 1; break;
        case ACTION_LEFT:  new_x -= 1; env->agent.facing_direction = 2; break;
        case ACTION_RIGHT: new_x += 1; env->agent.facing_direction = 3; break;
        case ACTION_INTERACT: handle_interaction(env); break;
    }
    
    if (action != ACTION_INTERACT && action != ACTION_NOOP) {
        if (is_valid_position(env, new_x, new_y)) {
            env->agent.x = new_x;
            env->agent.y = new_y;
        }
    }
    
    env->current_step++;
    env->log.episode_length++;
    
    if (env->current_step >= env->max_steps) {
        env->terminals[0] = 1;
    }
    
    env->log.episode_return += env->rewards[0];
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Overcooked* env) {
    if (env->client == NULL) {
        InitWindow(env->width * env->grid_size, env->height * env->grid_size, "PufferLib Overcooked");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        
        // Textures can be loaded here if available
    }
    
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    
    BeginDrawing();
    ClearBackground((Color){240, 240, 240, 255});
    
    // Draw grid tiles
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            Color tile_color = WHITE;
            
            switch (env->grid[idx]) {
                case WALL:
                    tile_color = DARKGRAY;
                    break;
                case COUNTER:
                    tile_color = BROWN;
                    break;
                case STOVE:
                    tile_color = RED;
                    break;
                case CUTTING_BOARD:
                    tile_color = BEIGE;
                    break;
                case INGREDIENT_BOX:
                    tile_color = GREEN;
                    break;
                case SERVING_AREA:
                    tile_color = GOLD;
                    break;
                case PLATE_BOX:
                    tile_color = SKYBLUE;
                    break;
                case EMPTY:
                    tile_color = WHITE;
                    break;
            }
            
            DrawRectangle(
                x * env->grid_size,
                y * env->grid_size,
                env->grid_size,
                env->grid_size,
                tile_color
            );
            
            // Draw grid lines for better visibility
            DrawRectangleLines(
                x * env->grid_size,
                y * env->grid_size,
                env->grid_size,
                env->grid_size,
                LIGHTGRAY
            );
        }
    }
    
    // Draw items
    for (int i = 0; i < env->num_items; i++) {
        Color item_color = GRAY;
        switch (env->items[i].type) {
            case TOMATO:
                item_color = RED;
                break;
            case ONION:
                item_color = YELLOW;
                break;
            case PLATE:
                item_color = WHITE;
                break;
            case SOUP:
                item_color = ORANGE;
                break;
        }
        
        DrawCircle(
            env->items[i].x * env->grid_size + env->grid_size/2,
            env->items[i].y * env->grid_size + env->grid_size/2,
            env->grid_size/4,
            item_color
        );
    }
    
    // Draw agent
    DrawRectangle(
        env->agent.x * env->grid_size + env->grid_size/4,
        env->agent.y * env->grid_size + env->grid_size/4,
        env->grid_size/2,
        env->grid_size/2,
        BLUE
    );
    
    // Draw held item above agent
    if (env->agent.held_item != NO_ITEM) {
        Color held_color = GRAY;
        switch (env->agent.held_item) {
            case TOMATO:
                held_color = RED;
                break;
            case ONION:
                held_color = YELLOW;
                break;
            case PLATE:
                held_color = WHITE;
                break;
            case SOUP:
                held_color = ORANGE;
                break;
        }
        
        DrawCircle(
            env->agent.x * env->grid_size + env->grid_size/2,
            env->agent.y * env->grid_size,
            env->grid_size/6,
            held_color
        );
    }
    
    EndDrawing();
}

void c_close(Overcooked* env) {
    free(env->grid);
    free(env->items);
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
    }
}