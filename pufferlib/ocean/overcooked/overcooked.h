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

// Simple kitchen layout for testing (10x10)
static const char DEFAULT_KITCHEN[10][10] = {
    {'#','#','#','#','#','#','#','#','#','#'},
    {'#','4','1','1','1','1','1','1','5','#'},
    {'#',' ',' ',' ',' ',' ',' ',' ',' ','#'},
    {'#','1','1','2','1','1','3','1','1','#'},
    {'#',' ',' ',' ',' ',' ',' ',' ',' ','#'},
    {'#',' ',' ',' ',' ',' ',' ',' ',' ','#'},
    {'#','1','1','1','1','1','1','1','1','#'},
    {'#',' ',' ',' ',' ',' ',' ',' ',' ','#'},
    {'#',' ',' ',' ',' ',' ',' ',' ',' ','#'},
    {'#','#','#','#','#','#','#','#','#','#'}
};

// Initialize environment (allocate memory)
static void init(Overcooked* env) {
    // TODO: Allocate memory for grid
    env->grid = calloc(env->width * env->height, sizeof(char));
    
    // TODO: Allocate memory for items array
    env->max_items = 20;
    env->items = calloc(env->max_items, sizeof(Item));
    env->num_items = 0;
    
    // TODO: Copy default kitchen layout to grid
    // Implement grid initialization
    
    // TODO: Initialize other components
    env->client = NULL;
}

// Compute observations for the agent
static void compute_observations(Overcooked* env) {
    // TODO: Fill observations array with:
    // - Grid layout around agent (e.g., 7x7 window)
    // - Agent position
    // - Agent held item
    // - Items on the map
    // - Other relevant state
    
    // For now, just zero out observations
    int obs_idx = 0;
    for (int i = 0; i < env->observation_size; i++) {
        env->observations[obs_idx++] = 0.0f;
    }
}

// Handle agent interaction with the environment
static void handle_interaction(Overcooked* env) {
    // TODO: Handle agent interaction based on what they're facing
    // - Pick up item if empty-handed
    // - Put down item if holding something
    // - Use station (stove, cutting board, etc.)
}

// Check if position is valid (not wall, within bounds)
static int is_valid_position(Overcooked* env, int x, int y) {
    // TODO: Check if position is within bounds and not a wall
    if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
        return 0;
    }
    
    int idx = y * env->width + x;
    return env->grid[idx] != WALL;
}

// Get item at position
static Item* get_item_at(Overcooked* env, int x, int y) {
    // TODO: Find and return item at given position
    for (int i = 0; i < env->num_items; i++) {
        if ((int)env->items[i].x == x && (int)env->items[i].y == y) {
            return &env->items[i];
        }
    }
    return NULL;
}

// Add item to the environment
static void add_item(Overcooked* env, int type, int x, int y) {
    // TODO: Add new item to the environment
    if (env->num_items < env->max_items) {
        env->items[env->num_items].type = type;
        env->items[env->num_items].x = x;
        env->items[env->num_items].y = y;
        env->items[env->num_items].state = 0;
        env->num_items++;
    }
}

// Remove item from position
static void remove_item(Overcooked* env, int x, int y) {
    // TODO: Remove item at given position
    for (int i = 0; i < env->num_items; i++) {
        if ((int)env->items[i].x == x && (int)env->items[i].y == y) {
            // Shift remaining items
            for (int j = i; j < env->num_items - 1; j++) {
                env->items[j] = env->items[j + 1];
            }
            env->num_items--;
            break;
        }
    }
}

// Required function
void c_reset(Overcooked* env) {
    // TODO: Reset environment state
    env->current_step = 0;
    env->num_items = 0;
    
    // TODO: Initialize grid from default kitchen
    // Copy DEFAULT_KITCHEN to env->grid
    
    // TODO: Place agent at starting position
    env->agent.x = 5;
    env->agent.y = 5;
    env->agent.held_item = NO_ITEM;
    env->agent.facing_direction = 0;
    
    // TODO: Place initial items (ingredients)
    // add_item(env, TOMATO, x, y);
    
    // TODO: Reset rewards and terminals
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    
    // TODO: Compute initial observations
    compute_observations(env);
    
    // Reset log
    env->log.episode_length = 0;
    env->log.episode_return = 0;
    env->log.dishes_served = 0;
}

// Required function
void c_step(Overcooked* env) {
    // TODO: Get action from actions array
    int action = env->actions[0];
    
    // Reset reward for this step
    env->rewards[0] = env->reward_step_penalty;
    
    // TODO: Process movement actions
    int new_x = env->agent.x;
    int new_y = env->agent.y;
    
    switch (action) {
        case ACTION_UP:
            new_y -= 1;
            env->agent.facing_direction = 0;
            break;
        case ACTION_DOWN:
            new_y += 1;
            env->agent.facing_direction = 1;
            break;
        case ACTION_LEFT:
            new_x -= 1;
            env->agent.facing_direction = 2;
            break;
        case ACTION_RIGHT:
            new_x += 1;
            env->agent.facing_direction = 3;
            break;
        case ACTION_INTERACT:
            handle_interaction(env);
            break;
    }
    
    // TODO: Check if new position is valid and update
    if (action != ACTION_INTERACT && action != ACTION_NOOP) {
        if (is_valid_position(env, new_x, new_y)) {
            env->agent.x = new_x;
            env->agent.y = new_y;
        }
    }
    
    // TODO: Update game state
    // - Check if dish was served
    // - Update cooking timers
    // - etc.
    
    // Update step counter
    env->current_step++;
    env->log.episode_length++;
    
    // TODO: Check terminal conditions
    if (env->current_step >= env->max_steps) {
        env->terminals[0] = 1;
    }
    
    // Update log
    env->log.episode_return += env->rewards[0];
    
    // TODO: Compute new observations
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Overcooked* env) {
    if (env->client == NULL) {
        InitWindow(env->width * env->grid_size, env->height * env->grid_size, "PufferLib Overcooked");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        
        // TODO: Load textures after InitWindow
        // env->client->tiles = LoadTexture("resources/overcooked/tiles.png");
        // env->client->items = LoadTexture("resources/overcooked/items.png");
        // env->client->agent = LoadTexture("resources/overcooked/agent.png");
    }
    
    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground((Color){240, 240, 240, 255});
    
    // TODO: Draw grid tiles
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
            }
            
            DrawRectangle(
                x * env->grid_size,
                y * env->grid_size,
                env->grid_size,
                env->grid_size,
                tile_color
            );
        }
    }
    
    // TODO: Draw items
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
    
    // TODO: Draw agent
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

// Required function. Should clean up anything you allocated
void c_close(Overcooked* env) {
    free(env->grid);
    free(env->items);
    
    if (env->client != NULL) {
        Client* client = env->client;
        // TODO: Unload textures if loaded
        // UnloadTexture(client->tiles);
        // UnloadTexture(client->items);
        // UnloadTexture(client->agent);
        CloseWindow();
        free(client);
    }
}