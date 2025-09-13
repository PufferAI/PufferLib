/* Overcooked: a multi-agent cooking coordination environment.
 * Agents can walk around, pick up items, and put down items.
 */

#include <stdio.h>
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
#define SOUP 4  // Generic soup (deprecated)
#define PLATED_SOUP 5  // Soup on a plate with ingredient info

// Cooking states
#define NOT_COOKING 0
#define COOKING 1
#define COOKED 2
#define BURNT 3

// Cooking parameters
#define COOKING_TIME 20  // Steps to cook
#define BURN_TIME 40     // Steps until burnt
#define MAX_INGREDIENTS 5 // Max ingredients per pot

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
    float cooperation_score; // Bonus for cooperative actions
    // User-defined stats
    float correct_dishes; // Number of correct 3-onion dishes
    float wrong_dishes; // Number of wrong dishes submitted
    float ingredients_picked; // Total ingredients picked up
    float pots_started; // Number of cooking sessions started
    float items_dropped; // Number of items dropped/placed
    float agent_collisions; // Number of times agents tried to move to same spot
    float cooking_time_efficiency; // Average cooking efficiency (0-1)
    float n; // Required as the last field 
} Log;

typedef struct {
    // Terrain textures
    Texture2D floor;
    Texture2D counter;
    Texture2D pot;
    Texture2D serve;
    Texture2D onions_box;
    Texture2D tomatoes_box;
    Texture2D dishes_box;
    
    // Object textures
    Texture2D onion;
    Texture2D tomato;
    Texture2D dish;
    Texture2D soup_onion;
    Texture2D soup_tomato;
    
    // Cooking stage textures
    Texture2D soup_onion_cooking_1;
    Texture2D soup_onion_cooking_2;
    Texture2D soup_onion_cooking_3;
    Texture2D soup_onion_cooked;
    Texture2D soup_tomato_cooking_1;
    Texture2D soup_tomato_cooking_2;
    Texture2D soup_tomato_cooking_3;
    Texture2D soup_tomato_cooked;
    
    // Chef sprites (4 directions)
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
    // Chef sprites holding soup
    Texture2D chef_north_soup_onion;
    Texture2D chef_south_soup_onion;
    Texture2D chef_east_soup_onion;
    Texture2D chef_west_soup_onion;
    Texture2D chef_north_soup_tomato;
    Texture2D chef_south_soup_tomato;
    Texture2D chef_east_soup_tomato;
    Texture2D chef_west_soup_tomato;
    
    // Plated soup textures
    Texture2D soup_onion_dish;
    Texture2D soup_tomato_dish;
} Client;

typedef struct {
    float x;
    float y;
    int held_item;  // Item type the agent is holding (NO_ITEM if empty)
    int facing_direction;  // 0=up, 1=down, 2=left, 3=right
    // Temporary storage for soup recipe when holding plated soup
    int held_soup_onions;
    int held_soup_tomatoes;
    int held_soup_total;
} Agent;

typedef struct {
    int type;
    float x;
    float y;
    int state;  // For items that can change state (e.g., cooking progress)
    // For plated soups, track the recipe
    int num_onions;     // Number of onions in the soup
    int num_tomatoes;   // Number of tomatoes in the soup
    int total_ingredients;  // Total ingredient count
} Item;

typedef struct {
    int cooking_state;      // NOT_COOKING, COOKING, COOKED, BURNT
    int cooking_progress;   // Steps since cooking started
    int ingredient_types[MAX_INGREDIENTS];  // Types of ingredients added
    int ingredient_count;   // Number of ingredients in pot
    int num_onions;        // Count of onions
    int num_tomatoes;      // Count of tomatoes
} CookingPot;

// Required that you have some struct for your env
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    
    // Game state
    char* grid;  // Kitchen layout (static tiles)
    Item* items;  // Dynamic items in the kitchen
    int num_items;
    int max_items;
    Agent* agents;  // Array of agents
    int num_agents;
    
    // Cooking state for each stove position
    CookingPot* cooking_pots;  // Array of cooking pots (one per stove)
    int num_stoves;
    
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


// Forward declarations for helper functions
static Item* get_item_at(Overcooked* env, int x, int y);
static void add_item(Overcooked* env, int type, int x, int y);
static void remove_item(Overcooked* env, int x, int y);
static CookingPot* get_pot_at(Overcooked* env, int x, int y);
static void init_cooking_pots(Overcooked* env);
static void update_cooking(Overcooked* env);
static void evaluate_dish_served(Overcooked* env, Agent* agent);

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
    env->agents = calloc(env->num_agents, sizeof(Agent));
    parse_grid(env);
    init_cooking_pots(env);
    env->client = NULL;
}

static void compute_observations(Overcooked* env) {
    // Simple flat array observation for each agent
    // Same structure for all agents but from their perspective
    int obs_idx = 0;
    
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        obs_idx = 0;  // Reset for each agent
        
        // 1. Grid tiles (5x5 = 25 values) - normalized tile types
        for (int y = 0; y < env->height; y++) {
            for (int x = 0; x < env->width; x++) {
                int tile = env->grid[y * env->width + x];
                env->observations[agent_idx * env->observation_size + obs_idx++] = tile / 7.0f;
            }
        }
        
        // 2. All agent states (4 values per agent)
        for (int a = 0; a < env->num_agents; a++) {
            env->observations[agent_idx * env->observation_size + obs_idx++] = env->agents[a].x / (float)env->width;
            env->observations[agent_idx * env->observation_size + obs_idx++] = env->agents[a].y / (float)env->height;
            env->observations[agent_idx * env->observation_size + obs_idx++] = env->agents[a].held_item / 5.0f;
            env->observations[agent_idx * env->observation_size + obs_idx++] = env->agents[a].facing_direction / 3.0f;
        }
        
        // 3. Item positions and types (max 10 items, 3 values each)
        for (int i = 0; i < 10; i++) {
            if (i < env->num_items) {
                env->observations[agent_idx * env->observation_size + obs_idx++] = env->items[i].x / (float)env->width;
                env->observations[agent_idx * env->observation_size + obs_idx++] = env->items[i].y / (float)env->height;
                env->observations[agent_idx * env->observation_size + obs_idx++] = env->items[i].type / 5.0f;
            } else {
                env->observations[agent_idx * env->observation_size + obs_idx++] = -1.0f;  // No item
                env->observations[agent_idx * env->observation_size + obs_idx++] = -1.0f;
                env->observations[agent_idx * env->observation_size + obs_idx++] = 0.0f;
            }
        }
        
        // 4. Cooking pot states (for each stove, 5 values each)
        for (int i = 0; i < env->num_stoves; i++) {
            CookingPot* pot = &env->cooking_pots[i];
            
            // Find stove position
            int stove_idx = 0;
            float stove_x = -1, stove_y = -1;
            for (int y = 0; y < env->height && stove_x < 0; y++) {
                for (int x = 0; x < env->width; x++) {
                    if (env->grid[y * env->width + x] == STOVE) {
                        if (stove_idx == i) {
                            stove_x = x / (float)env->width;
                            stove_y = y / (float)env->height;
                            break;
                        }
                        stove_idx++;
                    }
                }
            }
            
            env->observations[agent_idx * env->observation_size + obs_idx++] = stove_x;
            env->observations[agent_idx * env->observation_size + obs_idx++] = stove_y;
            env->observations[agent_idx * env->observation_size + obs_idx++] = pot->cooking_state / 3.0f;
            env->observations[agent_idx * env->observation_size + obs_idx++] = pot->cooking_progress / (float)BURN_TIME;
            env->observations[agent_idx * env->observation_size + obs_idx++] = pot->ingredient_count / (float)MAX_INGREDIENTS;
        }
        
        // 5. Global state
        env->observations[agent_idx * env->observation_size + obs_idx++] = env->current_step / (float)env->max_steps;
        env->observations[agent_idx * env->observation_size + obs_idx++] = env->log.dishes_served / 10.0f;
        env->observations[agent_idx * env->observation_size + obs_idx++] = env->rewards[agent_idx];  // Last reward
    }
}

static void handle_interaction(Overcooked* env, int agent_idx) {
    Agent* agent = &env->agents[agent_idx];
    // Get position agent is facing
    int target_x = agent->x;
    int target_y = agent->y;
    
    switch (agent->facing_direction) {
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
    CookingPot* pot = get_pot_at(env, target_x, target_y);
    
    // Special stove interaction
    if (tile == STOVE && pot != NULL) {
        // If agent is holding an ingredient and pot is not cooking yet
        if (agent->held_item == ONION || agent->held_item == TOMATO) {
            if (pot->cooking_state == NOT_COOKING && pot->ingredient_count < MAX_INGREDIENTS) {
                // Add ingredient to pot
                pot->ingredient_types[pot->ingredient_count] = agent->held_item;
                pot->ingredient_count++;
                if (agent->held_item == ONION) {
                    pot->num_onions++;
                } else if (agent->held_item == TOMATO) {
                    pot->num_tomatoes++;
                }
                agent->held_item = NO_ITEM;
            }
        }
        // If agent is empty handed and pot has ingredients, start cooking
        else if (agent->held_item == NO_ITEM && pot->ingredient_count > 0) {
            if (pot->cooking_state == NOT_COOKING) {
                pot->cooking_state = COOKING;
                pot->cooking_progress = 0;
                env->log.pots_started++;  // Track cooking sessions started
            }
            // Pick up cooked soup only with a plate
            else if (pot->cooking_state == COOKED) {
                // Can't pick up soup without a plate
                return;
            }
        }
        // If agent is holding a plate and pot has cooked soup
        else if (agent->held_item == PLATE && pot->cooking_state == COOKED) {
            // Create plated soup with ingredient info
            agent->held_item = PLATED_SOUP;
            // Store the soup's ingredient info in agent's temporary state
            // We'll need to track this when placing the soup down
            agent->held_soup_onions = pot->num_onions;
            agent->held_soup_tomatoes = pot->num_tomatoes;
            agent->held_soup_total = pot->ingredient_count;
            
            // Reset pot
            pot->cooking_state = NOT_COOKING;
            pot->cooking_progress = 0;
            pot->ingredient_count = 0;
            pot->num_onions = 0;
            pot->num_tomatoes = 0;
            for (int i = 0; i < MAX_INGREDIENTS; i++) {
                pot->ingredient_types[i] = NO_ITEM;
            }
        }
        return;
    }
    
    // Check for serving completed dish
    if (tile == SERVING_AREA && agent->held_item == PLATED_SOUP) {
        // Evaluate the dish and assign rewards based on rules
        evaluate_dish_served(env, agent);
        
        // Clear the plated soup from agent's hands (whether correct or not)
        agent->held_item = NO_ITEM;
        agent->held_soup_onions = 0;
        agent->held_soup_tomatoes = 0;
        agent->held_soup_total = 0;
        return;
    }
    
    // Normal interaction (non-stove)
    if (agent->held_item != NO_ITEM) {
        // Can only put down on empty counters or cutting boards
        if ((tile == COUNTER || tile == CUTTING_BOARD) && item == NULL) {
            // Special handling for plated soup to preserve recipe
            if (agent->held_item == PLATED_SOUP) {
                add_item(env, agent->held_item, target_x, target_y);
                // Transfer soup recipe to the placed item
                Item* placed_soup = get_item_at(env, target_x, target_y);
                if (placed_soup) {
                    placed_soup->num_onions = agent->held_soup_onions;
                    placed_soup->num_tomatoes = agent->held_soup_tomatoes;
                    placed_soup->total_ingredients = agent->held_soup_total;
                }
                // Clear agent's soup info
                agent->held_soup_onions = 0;
                agent->held_soup_tomatoes = 0;
                agent->held_soup_total = 0;
            } else {
                add_item(env, agent->held_item, target_x, target_y);
            }
            agent->held_item = NO_ITEM;
            env->log.items_dropped++;  // Track items dropped
        }
    }
    else {
        // Pick up item if there is one
        if (item != NULL) {
            // Special handling for plated soup to preserve recipe
            if (item->type == PLATED_SOUP) {
                agent->held_soup_onions = item->num_onions;
                agent->held_soup_tomatoes = item->num_tomatoes;
                agent->held_soup_total = item->total_ingredients;
            }
            agent->held_item = item->type;
            remove_item(env, target_x, target_y);
        }
        // Special case: get new ingredients from ingredient box
        else if (tile == INGREDIENT_BOX) {
            agent->held_item = ONION; // Always gives onions for now
            env->log.ingredients_picked++;  // Track ingredient pickups
        }
        // Special case: get plates from plate box
        else if (tile == PLATE_BOX) {
            agent->held_item = PLATE;
            env->log.items_dropped++;  // Track items picked up (reusing counter)
        }
    }
}

static int is_valid_position(Overcooked* env, int x, int y, int excluding_agent) {
    if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
        return 0;
    }
    if (env->grid[y * env->width + x] != EMPTY) {
        return 0;
    }
    // Check for collision with other agents
    for (int i = 0; i < env->num_agents; i++) {
        if (i != excluding_agent && (int)env->agents[i].x == x && (int)env->agents[i].y == y) {
            return 0;  // Position occupied by another agent
        }
    }
    return 1;
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
        env->items[env->num_items].num_onions = 0;
        env->items[env->num_items].num_tomatoes = 0;
        env->items[env->num_items].total_ingredients = 0;
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

static Color get_agent_color(int held_item) {
    switch (held_item) {
        case NO_ITEM:
            return BLUE;      // Blue when empty-handed
        case TOMATO:
            return (Color){200, 50, 50, 255};   // Dark red when holding tomato
        case ONION:
            return (Color){255, 200, 100, 255}; // Light orange when holding onion
        case PLATE:
            return (Color){200, 200, 220, 255}; // Light blue-gray when holding plate
        case SOUP:
            return (Color){255, 140, 0, 255};   // Orange when holding soup
        case PLATED_SOUP:
            return (Color){255, 165, 0, 255};   // Brighter orange when holding plated soup
        default:
            return BLUE;      // Default to blue
    }
}

static void init_cooking_pots(Overcooked* env) {
    // Count stoves in the grid
    env->num_stoves = 0;
    for (int i = 0; i < env->width * env->height; i++) {
        if (env->grid[i] == STOVE) {
            env->num_stoves++;
        }
    }
    
    // Allocate cooking pots
    env->cooking_pots = calloc(env->num_stoves, sizeof(CookingPot));
    
    // Initialize each pot
    int pot_index = 0;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (env->grid[y * env->width + x] == STOVE) {
                CookingPot* pot = &env->cooking_pots[pot_index];
                pot->cooking_state = NOT_COOKING;
                pot->cooking_progress = 0;
                pot->ingredient_count = 0;
                pot->num_onions = 0;
                pot->num_tomatoes = 0;
                for (int i = 0; i < MAX_INGREDIENTS; i++) {
                    pot->ingredient_types[i] = NO_ITEM;
                }
                pot_index++;
            }
        }
    }
}

static CookingPot* get_pot_at(Overcooked* env, int x, int y) {
    if (env->grid[y * env->width + x] != STOVE) {
        return NULL;
    }
    
    // Find which stove index this is
    int stove_index = 0;
    for (int sy = 0; sy < env->height; sy++) {
        for (int sx = 0; sx < env->width; sx++) {
            if (env->grid[sy * env->width + sx] == STOVE) {
                if (sx == x && sy == y) {
                    return &env->cooking_pots[stove_index];
                }
                stove_index++;
            }
        }
    }
    return NULL;
}

static void update_cooking(Overcooked* env) {
    for (int i = 0; i < env->num_stoves; i++) {
        CookingPot* pot = &env->cooking_pots[i];
        if (pot->cooking_state == COOKING) {
            pot->cooking_progress++;
            if (pot->cooking_progress >= COOKING_TIME) {
                pot->cooking_state = COOKED;
            } else if (pot->cooking_progress >= BURN_TIME) {
                pot->cooking_state = BURNT;
            }
        }
    }
}

// Dedicated function to evaluate dish and assign rewards
// Easy to modify rules here without changing interaction logic
static void evaluate_dish_served(Overcooked* env, Agent* agent) {
    // Rule 1: Check if soup has exactly 3 onions
    int is_correct_recipe = (agent->held_soup_onions == 3);
    
    // You can add more rules here, e.g.:
    // int has_no_tomatoes = (agent->held_soup_tomatoes == 0);
    // int is_not_burnt = 1;  // Could track if soup was burnt
    // int served_quickly = (env->current_step < 100);
    
    if (is_correct_recipe) {
        // Correct dish - give full reward to all agents
        float reward = env->reward_dish_served;
        
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += reward;
        }
        
        env->log.dishes_served++;
        env->log.correct_dishes++;  // Track correct dishes
        env->log.score += reward;
        
        // Calculate cooking efficiency bonus
        if (env->current_step > 0) {
            float efficiency = (float)env->log.correct_dishes / (env->current_step / 100.0f);
            env->log.cooking_time_efficiency = efficiency > 1.0f ? 1.0f : efficiency;
        }
    } else {
        // Wrong recipe - no penalty, just track it
        env->log.wrong_dishes++;  // Track wrong dishes
    }
}

void c_reset(Overcooked* env) {
    env->current_step = 0;
    env->num_items = 0;
    parse_grid(env);
    
    // Reset cooking pots
    for (int i = 0; i < env->num_stoves; i++) {
        CookingPot* pot = &env->cooking_pots[i];
        pot->cooking_state = NOT_COOKING;
        pot->cooking_progress = 0;
        pot->ingredient_count = 0;
        pot->num_onions = 0;
        pot->num_tomatoes = 0;
        for (int j = 0; j < MAX_INGREDIENTS; j++) {
            pot->ingredient_types[j] = NO_ITEM;
        }
    }
    
    // Reset all agents with different starting positions
    for (int i = 0; i < env->num_agents; i++) {
        // Place agents at different starting positions
        if (i == 0) {
            env->agents[i].x = 1;
            env->agents[i].y = 2;
        } else if (i == 1) {
            env->agents[i].x = 3;
            env->agents[i].y = 2;
        } else {
            // For more than 2 agents, distribute them
            env->agents[i].x = 1 + (i % 3);
            env->agents[i].y = 1 + (i / 3);
        }
        env->agents[i].held_item = NO_ITEM;
        env->agents[i].facing_direction = 0;
        env->agents[i].held_soup_onions = 0;
        env->agents[i].held_soup_tomatoes = 0;
        env->agents[i].held_soup_total = 0;
        
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }
    
    compute_observations(env);
    
    env->log.episode_length = 0;
    env->log.episode_return = 0;
    env->log.dishes_served = 0;
    env->log.cooperation_score = 0;
    // Initialize user stats
    env->log.correct_dishes = 0;
    env->log.wrong_dishes = 0;
    env->log.ingredients_picked = 0;
    env->log.pots_started = 0;
    env->log.items_dropped = 0;
    env->log.agent_collisions = 0;
    env->log.cooking_time_efficiency = 0;
}

void c_step(Overcooked* env) {
    // Process actions for all agents
    for (int i = 0; i < env->num_agents; i++) {
        int action = env->actions[i];
        env->rewards[i] = env->reward_step_penalty;
        
        Agent* agent = &env->agents[i];
        int new_x = agent->x;
        int new_y = agent->y;
        
        switch (action) {
            case ACTION_UP:    new_y -= 1; agent->facing_direction = 0; break;
            case ACTION_DOWN:  new_y += 1; agent->facing_direction = 1; break;
            case ACTION_LEFT:  new_x -= 1; agent->facing_direction = 2; break;
            case ACTION_RIGHT: new_x += 1; agent->facing_direction = 3; break;
            case ACTION_INTERACT: handle_interaction(env, i); break;
        }
        
        if (action != ACTION_INTERACT && action != ACTION_NOOP) {
            if (is_valid_position(env, new_x, new_y, i)) {
                agent->x = new_x;
                agent->y = new_y;
            } else {
                // Check if collision was with another agent
                for (int j = 0; j < env->num_agents; j++) {
                    if (j != i && (int)env->agents[j].x == new_x && (int)env->agents[j].y == new_y) {
                        env->log.agent_collisions++;
                        break;
                    }
                }
            }
        }
    }
    
    // Update cooking progress
    update_cooking(env);
    
    env->current_step++;
    env->log.episode_length++;
    
    // Check for terminal condition
    if (env->current_step >= env->max_steps) {
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
        }
    }
    
    // Update episode return
    for (int i = 0; i < env->num_agents; i++) {
        env->log.episode_return += env->rewards[i];
    }
    
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Overcooked* env) {
    if (env->client == NULL) {
        // Add extra height for status display
        int window_width = env->width * env->grid_size;
        int window_height = env->height * env->grid_size + 80;  // Extra 80 pixels for status
        InitWindow(window_width, window_height, "PufferLib Overcooked");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        
        // Load terrain textures
        env->client->floor = LoadTexture("pufferlib/resources/overcooked/terrain/floor.png");
        env->client->counter = LoadTexture("pufferlib/resources/overcooked/terrain/counter.png");
        env->client->pot = LoadTexture("pufferlib/resources/overcooked/terrain/pot.png");
        env->client->serve = LoadTexture("pufferlib/resources/overcooked/terrain/serve.png");
        env->client->onions_box = LoadTexture("pufferlib/resources/overcooked/terrain/onions.png");
        env->client->tomatoes_box = LoadTexture("pufferlib/resources/overcooked/terrain/tomatoes.png");
        env->client->dishes_box = LoadTexture("pufferlib/resources/overcooked/terrain/dishes.png");
        
        // Load object textures
        env->client->onion = LoadTexture("pufferlib/resources/overcooked/objects/onion.png");
        env->client->tomato = LoadTexture("pufferlib/resources/overcooked/objects/tomato.png");
        env->client->dish = LoadTexture("pufferlib/resources/overcooked/objects/dish.png");
        env->client->soup_onion = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-cooked.png");
        env->client->soup_tomato = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-cooked.png");
        env->client->soup_onion_dish = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-dish.png");
        env->client->soup_tomato_dish = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-dish.png");
        
        // Load cooking stage textures
        env->client->soup_onion_cooking_1 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-1-cooking.png");
        env->client->soup_onion_cooking_2 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-2-cooking.png");
        env->client->soup_onion_cooking_3 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-3-cooking.png");
        env->client->soup_onion_cooked = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-cooked.png");
        env->client->soup_tomato_cooking_1 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-1-cooking.png");
        env->client->soup_tomato_cooking_2 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-2-cooking.png");
        env->client->soup_tomato_cooking_3 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-3-cooking.png");
        env->client->soup_tomato_cooked = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-cooked.png");
        
        // Load chef sprites
        env->client->chef_north = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH.png");
        env->client->chef_south = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH.png");
        env->client->chef_east = LoadTexture("pufferlib/resources/overcooked/chefs/EAST.png");
        env->client->chef_west = LoadTexture("pufferlib/resources/overcooked/chefs/WEST.png");
        env->client->chef_north_onion = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH-onion.png");
        env->client->chef_south_onion = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH-onion.png");
        env->client->chef_east_onion = LoadTexture("pufferlib/resources/overcooked/chefs/EAST-onion.png");
        env->client->chef_west_onion = LoadTexture("pufferlib/resources/overcooked/chefs/WEST-onion.png");
        env->client->chef_north_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH-tomato.png");
        env->client->chef_south_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH-tomato.png");
        env->client->chef_east_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/EAST-tomato.png");
        env->client->chef_west_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/WEST-tomato.png");
        env->client->chef_north_dish = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH-dish.png");
        env->client->chef_south_dish = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH-dish.png");
        env->client->chef_east_dish = LoadTexture("pufferlib/resources/overcooked/chefs/EAST-dish.png");
        env->client->chef_west_dish = LoadTexture("pufferlib/resources/overcooked/chefs/WEST-dish.png");
        
        // Load chef sprites holding soup
        env->client->chef_north_soup_onion = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH-soup-onion.png");
        env->client->chef_south_soup_onion = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH-soup-onion.png");
        env->client->chef_east_soup_onion = LoadTexture("pufferlib/resources/overcooked/chefs/EAST-soup-onion.png");
        env->client->chef_west_soup_onion = LoadTexture("pufferlib/resources/overcooked/chefs/WEST-soup-onion.png");
        env->client->chef_north_soup_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/NORTH-soup-tomato.png");
        env->client->chef_south_soup_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/SOUTH-soup-tomato.png");
        env->client->chef_east_soup_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/EAST-soup-tomato.png");
        env->client->chef_west_soup_tomato = LoadTexture("pufferlib/resources/overcooked/chefs/WEST-soup-tomato.png");
    }
    
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    
    BeginDrawing();
    ClearBackground((Color){240, 240, 240, 255});
    
    // Draw game status at the top
    DrawText(TextFormat("Step: %d / %d", env->current_step, env->max_steps), 10, 10, 20, BLACK);
    DrawText(TextFormat("Dishes Served: %d", (int)env->log.dishes_served), 10, 35, 20, BLACK);
    DrawText("Recipe: 3 Onions", 10, 60, 16, DARKGRAY);
    
    // Draw grid tiles with textures (offset by 80 pixels for status area)
    int grid_offset_y = 80;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            Rectangle dest = {x * env->grid_size, y * env->grid_size + grid_offset_y, env->grid_size, env->grid_size};
            
            // Draw floor for all tiles first
            if (env->client->floor.id != 0) {
                DrawTexturePro(env->client->floor, 
                    (Rectangle){0, 0, env->client->floor.width, env->client->floor.height},
                    dest, (Vector2){0, 0}, 0, WHITE);
            }
            
            // Draw specific tile overlays
            Texture2D* texture = NULL;
            switch (env->grid[idx]) {
                case COUNTER:
                    texture = &env->client->counter;
                    break;
                case STOVE:
                    texture = &env->client->pot;
                    break;
                case CUTTING_BOARD:
                    texture = &env->client->counter;  // Use counter for cutting board
                    break;
                case INGREDIENT_BOX:
                    // for now, we only use the onions box
                    texture = &env->client->onions_box;
                    break;
                case SERVING_AREA:
                    texture = &env->client->serve;
                    break;
                case PLATE_BOX:
                    texture = &env->client->dishes_box;
                    break;
                case WALL:
                    // Draw a dark rectangle for walls
                    DrawRectangle(x * env->grid_size, y * env->grid_size, 
                                  env->grid_size, env->grid_size, DARKGRAY);
                    continue;
            }
            
            // Draw the texture if available
            if (texture && texture->id != 0) {
                DrawTexturePro(*texture,
                    (Rectangle){0, 0, texture->width, texture->height},
                    dest, (Vector2){0, 0}, 0, WHITE);
            }
            
            // Draw cooking state on stoves
            if (env->grid[idx] == STOVE) {
                CookingPot* pot = get_pot_at(env, x, y);
                if (pot && pot->ingredient_count > 0) {
                    Texture2D* cooking_texture = NULL;
                    
                    // Determine if soup is primarily onion or tomato based
                    bool is_onion_soup = (pot->num_onions >= pot->num_tomatoes);
                    
                    if (pot->cooking_state == COOKING) {
                        // Select cooking stage texture based on progress
                        float progress = (float)pot->cooking_progress / COOKING_TIME;
                        if (is_onion_soup) {
                            if (progress < 0.33f) {
                                cooking_texture = &env->client->soup_onion_cooking_1;
                            } else if (progress < 0.66f) {
                                cooking_texture = &env->client->soup_onion_cooking_2;
                            } else {
                                cooking_texture = &env->client->soup_onion_cooking_3;
                            }
                        } else {
                            if (progress < 0.33f) {
                                cooking_texture = &env->client->soup_tomato_cooking_1;
                            } else if (progress < 0.66f) {
                                cooking_texture = &env->client->soup_tomato_cooking_2;
                            } else {
                                cooking_texture = &env->client->soup_tomato_cooking_3;
                            }
                        }
                        
                        // Draw progress bar below
                        DrawRectangle(x * env->grid_size + 5,
                                    y * env->grid_size + grid_offset_y + env->grid_size - 10,
                                    (env->grid_size - 10) * progress, 3, GREEN);
                        DrawRectangleLines(x * env->grid_size + 5,
                                         y * env->grid_size + grid_offset_y + env->grid_size - 10,
                                         env->grid_size - 10, 3, BLACK);
                    }
                    else if (pot->cooking_state == COOKED) {
                        cooking_texture = is_onion_soup ? &env->client->soup_onion_cooked : 
                                                          &env->client->soup_tomato_cooked;
                        // Small "READY!" text
                        DrawText("READY!", x * env->grid_size + 5,
                               y * env->grid_size + grid_offset_y + env->grid_size - 10,
                               8, GREEN);
                    }
                    else if (pot->cooking_state == BURNT) {
                        // Still use cooked texture but tint it darker
                        cooking_texture = is_onion_soup ? &env->client->soup_onion_cooked : 
                                                          &env->client->soup_tomato_cooked;
                        DrawText("BURNT!", x * env->grid_size + 5,
                               y * env->grid_size + grid_offset_y + env->grid_size - 10,
                               8, RED);
                    }
                    else if (pot->cooking_state == NOT_COOKING) {
                        // Show ingredients not cooking yet - use first cooking stage
                        cooking_texture = is_onion_soup ? &env->client->soup_onion_cooking_1 : 
                                                          &env->client->soup_tomato_cooking_1;
                    }
                    
                    // Draw the cooking texture on top of the pot
                    if (cooking_texture && cooking_texture->id != 0) {
                        Rectangle pot_dest = {
                            x * env->grid_size + env->grid_size/4,
                            y * env->grid_size + grid_offset_y + env->grid_size/4,
                            env->grid_size/2,
                            env->grid_size/2
                        };
                        Color tint = (pot->cooking_state == BURNT) ? DARKGRAY : WHITE;
                        DrawTexturePro(*cooking_texture,
                            (Rectangle){0, 0, cooking_texture->width, cooking_texture->height},
                            pot_dest, (Vector2){0, 0}, 0, tint);
                    }
                }
            }
        }
    }
    
    // Draw items with textures
    for (int i = 0; i < env->num_items; i++) {
        Texture2D* texture = NULL;
        switch (env->items[i].type) {
            case TOMATO:
                texture = &env->client->tomato;
                break;
            case ONION:
                texture = &env->client->onion;
                break;
            case PLATE:
                texture = &env->client->dish;
                break;
            case SOUP:
                texture = &env->client->soup_onion;
                break;
            case PLATED_SOUP:
                // Use the plated soup sprites
                if (env->items[i].num_onions >= env->items[i].num_tomatoes) {
                    texture = &env->client->soup_onion_dish;
                } else {
                    texture = &env->client->soup_tomato_dish;
                }
                break;
        }
        
        if (texture && texture->id != 0) {
            Rectangle dest = {
                env->items[i].x * env->grid_size + env->grid_size/4,
                env->items[i].y * env->grid_size + grid_offset_y + env->grid_size/4,
                env->grid_size/2,
                env->grid_size/2
            };
            DrawTexturePro(*texture,
                (Rectangle){0, 0, texture->width, texture->height},
                dest, (Vector2){0, 0}, 0, WHITE);
        } else {
            // Fallback to colored circle if texture not loaded
            Color item_color = GRAY;
            switch (env->items[i].type) {
                case TOMATO: item_color = RED; break;
                case ONION: item_color = YELLOW; break;
                case PLATE: item_color = WHITE; break;
                case SOUP: item_color = ORANGE; break;
                case PLATED_SOUP: item_color = ORANGE; break;
            }
            DrawCircle(
                env->items[i].x * env->grid_size + env->grid_size/2,
                env->items[i].y * env->grid_size + grid_offset_y + env->grid_size/2,
                env->grid_size/4,
                item_color
            );
        }
    }
    
    // Draw all agents with appropriate chef sprites
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        Texture2D* chef_texture = NULL;
        
        // Select chef texture based on direction and held item
        if (agent->held_item == NO_ITEM) {
            // Empty handed chef
            switch (agent->facing_direction) {
                case 0: chef_texture = &env->client->chef_north; break;
                case 1: chef_texture = &env->client->chef_south; break;
                case 2: chef_texture = &env->client->chef_west; break;
                case 3: chef_texture = &env->client->chef_east; break;
            }
        } else if (agent->held_item == ONION) {
            switch (agent->facing_direction) {
                case 0: chef_texture = &env->client->chef_north_onion; break;
                case 1: chef_texture = &env->client->chef_south_onion; break;
                case 2: chef_texture = &env->client->chef_west_onion; break;
                case 3: chef_texture = &env->client->chef_east_onion; break;
            }
        } else if (agent->held_item == TOMATO) {
            switch (agent->facing_direction) {
                case 0: chef_texture = &env->client->chef_north_tomato; break;
                case 1: chef_texture = &env->client->chef_south_tomato; break;
                case 2: chef_texture = &env->client->chef_west_tomato; break;
                case 3: chef_texture = &env->client->chef_east_tomato; break;
            }
        } else if (agent->held_item == PLATE) {
            switch (agent->facing_direction) {
                case 0: chef_texture = &env->client->chef_north_dish; break;
                case 1: chef_texture = &env->client->chef_south_dish; break;
                case 2: chef_texture = &env->client->chef_west_dish; break;
                case 3: chef_texture = &env->client->chef_east_dish; break;
            }
        } else if (agent->held_item == PLATED_SOUP) {
            // Use soup-specific sprites based on the soup type
            bool is_onion_soup = (agent->held_soup_onions >= agent->held_soup_tomatoes);
            if (is_onion_soup) {
                switch (agent->facing_direction) {
                    case 0: chef_texture = &env->client->chef_north_soup_onion; break;
                    case 1: chef_texture = &env->client->chef_south_soup_onion; break;
                    case 2: chef_texture = &env->client->chef_west_soup_onion; break;
                    case 3: chef_texture = &env->client->chef_east_soup_onion; break;
                }
            } else {
                switch (agent->facing_direction) {
                    case 0: chef_texture = &env->client->chef_north_soup_tomato; break;
                    case 1: chef_texture = &env->client->chef_south_soup_tomato; break;
                    case 2: chef_texture = &env->client->chef_west_soup_tomato; break;
                    case 3: chef_texture = &env->client->chef_east_soup_tomato; break;
                }
            }
        }
        
        // Draw the chef sprite if texture is loaded
        if (chef_texture && chef_texture->id != 0) {
            Rectangle dest = {
                agent->x * env->grid_size,
                agent->y * env->grid_size + grid_offset_y,
                env->grid_size,
                env->grid_size
            };
            // Tint agents with different colors to distinguish them
            Color tint = WHITE;
            if (agent_idx == 0) {
                tint = (Color){255, 255, 255, 255};  // White for player 1
            } else if (agent_idx == 1) {
                tint = (Color){200, 200, 255, 255};  // Light blue tint for player 2
            } else {
                tint = (Color){255, 200, 200, 255};  // Light red tint for other players
            }
            DrawTexturePro(*chef_texture,
                (Rectangle){0, 0, chef_texture->width, chef_texture->height},
                dest, (Vector2){0, 0}, 0, tint);
        } else {
            // Fallback to colored rectangle with item indicator
            Color agent_color = get_agent_color(agent->held_item);
            // Modify color based on agent index
            if (agent_idx == 1) {
                agent_color = (Color){agent_color.r * 0.8, agent_color.g * 0.8, agent_color.b, agent_color.a};
            }
            DrawRectangle(
                agent->x * env->grid_size + env->grid_size/4,
                agent->y * env->grid_size + grid_offset_y + env->grid_size/4,
                env->grid_size/2,
                env->grid_size/2,
                agent_color
            );
            
            // Draw direction indicator
            int dir_x = agent->x * env->grid_size + env->grid_size/2;
            int dir_y = agent->y * env->grid_size + grid_offset_y + env->grid_size/2;
            int end_x = dir_x, end_y = dir_y;
            switch (agent->facing_direction) {
                case 0: end_y -= env->grid_size/4; break; // Up
                case 1: end_y += env->grid_size/4; break; // Down
                case 2: end_x -= env->grid_size/4; break; // Left  
                case 3: end_x += env->grid_size/4; break; // Right
            }
            DrawLine(dir_x, dir_y, end_x, end_y, BLACK);
            
            // Draw agent number
            DrawText(TextFormat("%d", agent_idx + 1), 
                     agent->x * env->grid_size + 2,
                     agent->y * env->grid_size + grid_offset_y + 2,
                     10, BLACK);
        }
    }
    
    EndDrawing();
}

void c_close(Overcooked* env) {
    free(env->grid);
    free(env->items);
    free(env->agents);
    free(env->cooking_pots);
    if (env->client != NULL) {
        // Unload terrain textures
        UnloadTexture(env->client->floor);
        UnloadTexture(env->client->counter);
        UnloadTexture(env->client->pot);
        UnloadTexture(env->client->serve);
        UnloadTexture(env->client->onions_box);
        UnloadTexture(env->client->tomatoes_box);
        UnloadTexture(env->client->dishes_box);
        
        // Unload object textures
        UnloadTexture(env->client->onion);
        UnloadTexture(env->client->tomato);
        UnloadTexture(env->client->dish);
        UnloadTexture(env->client->soup_onion);
        UnloadTexture(env->client->soup_tomato);
        UnloadTexture(env->client->soup_onion_dish);
        UnloadTexture(env->client->soup_tomato_dish);
        
        // Unload cooking stage textures
        UnloadTexture(env->client->soup_onion_cooking_1);
        UnloadTexture(env->client->soup_onion_cooking_2);
        UnloadTexture(env->client->soup_onion_cooking_3);
        UnloadTexture(env->client->soup_onion_cooked);
        UnloadTexture(env->client->soup_tomato_cooking_1);
        UnloadTexture(env->client->soup_tomato_cooking_2);
        UnloadTexture(env->client->soup_tomato_cooking_3);
        UnloadTexture(env->client->soup_tomato_cooked);
        
        // Unload chef sprites
        UnloadTexture(env->client->chef_north);
        UnloadTexture(env->client->chef_south);
        UnloadTexture(env->client->chef_east);
        UnloadTexture(env->client->chef_west);
        UnloadTexture(env->client->chef_north_onion);
        UnloadTexture(env->client->chef_south_onion);
        UnloadTexture(env->client->chef_east_onion);
        UnloadTexture(env->client->chef_west_onion);
        UnloadTexture(env->client->chef_north_tomato);
        UnloadTexture(env->client->chef_south_tomato);
        UnloadTexture(env->client->chef_east_tomato);
        UnloadTexture(env->client->chef_west_tomato);
        UnloadTexture(env->client->chef_north_dish);
        UnloadTexture(env->client->chef_south_dish);
        UnloadTexture(env->client->chef_east_dish);
        UnloadTexture(env->client->chef_west_dish);
        UnloadTexture(env->client->chef_north_soup_onion);
        UnloadTexture(env->client->chef_south_soup_onion);
        UnloadTexture(env->client->chef_east_soup_onion);
        UnloadTexture(env->client->chef_west_soup_onion);
        UnloadTexture(env->client->chef_north_soup_tomato);
        UnloadTexture(env->client->chef_south_soup_tomato);
        UnloadTexture(env->client->chef_east_soup_tomato);
        UnloadTexture(env->client->chef_west_soup_tomato);
        
        CloseWindow();
        free(env->client);
    }
}