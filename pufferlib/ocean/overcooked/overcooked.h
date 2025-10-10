/* Overcooked: a multi-agent cooking coordination environment.
 * Agents can walk around, pick up items, and put down items.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define EMPTY 0
#define COUNTER 1
#define STOVE 2
#define CUTTING_BOARD 3
#define INGREDIENT_BOX 4
#define SERVING_AREA 5
#define WALL 6
#define PLATE_BOX 7
#define AGENT 8

#define NO_ITEM 10
#define TOMATO 11
#define ONION 12
#define PLATE 13
#define SOUP 14
#define PLATED_SOUP 15

#define NOT_COOKING 0
#define COOKING 1
#define COOKED 2
#define BURNT 3

#define COOKING_TIME 20
#define BURN_TIME 40
#define MAX_INGREDIENTS 3

#define ACTION_NOOP 0
#define ACTION_UP 1
#define ACTION_DOWN 2
#define ACTION_LEFT 3
#define ACTION_RIGHT 4
#define ACTION_INTERACT 5

#define AGENT_EMPTY_HANDED 0
#define AGENT_HOLDING_ITEM 1

typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float dishes_served; // Number of dishes successfully served
    float cooperation_score; // Bonus for cooperative actions
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

typedef struct {
    float x;
    float y;
    int held_item;  // Item type the agent is holding (NO_ITEM if empty)
    int facing_direction;  // 0=up, 1=down, 2=left, 3=right
    int held_soup_onions;
    int held_soup_tomatoes;
    int held_soup_total;
} Agent;

typedef struct {
    int type;
    int x;  // Changed from float to int for grid positioning
    int y;  // Changed from float to int for grid positioning
    int state;  // For items that can change state (e.g., cooking progress)
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
    char* grid;  // Kitchen layout (static tiles)
    Item* items;  // Dynamic items in the kitchen
    int num_items;
    int max_items;
    Agent* agents;  // Array of agents
    int num_agents;
    CookingPot* cooking_pots;  // Array of cooking pots (one per stove)
    int num_stoves;
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int width;
    int height;
    int max_steps;
    int current_step;
    int grid_size; 
    float reward_dish_served;
    float reward_step_penalty;
    int observation_size;
} Overcooked;


static Item* get_item_at(Overcooked* env, int x, int y);
static void add_item(Overcooked* env, int type, int x, int y);
static void remove_item(Overcooked* env, int x, int y);
static CookingPot* get_pot_at(Overcooked* env, int x, int y);
static void init_cooking_pots(Overcooked* env);
static void update_cooking(Overcooked* env);
static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx);

static const char CRAMPED_ROOM[5][5] = {
    {'6', '1', '2', '1', '6'},
    {'4', ' ', ' ', ' ', '4'},
    {'1', ' ', ' ', ' ', '1'},
    {'1', ' ', ' ', ' ', '1'},
    {'6', '7', '1', '5', '6'}
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
                case '6': env->grid[idx] = WALL; break;
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

    memset(&env->log, 0, sizeof(Log));
}

static void compute_proximity_feature(Overcooked* env, Agent* agent, int feature_type, float* dx, float* dy) {
    *dx = 0.0f;
    *dy = 0.0f;

    if (feature_type == INGREDIENT_BOX && agent->held_item == ONION) return;  // Holding onion
    if (feature_type == PLATE_BOX && agent->held_item == PLATE) return;       // Holding dish
    if (feature_type == PLATED_SOUP && agent->held_item == PLATED_SOUP) return; // Holding soup

    if (feature_type == PLATED_SOUP) {
        float min_dist = 1000.0f;
        for (int i = 0; i < env->num_items; i++) {
            if (env->items[i].type == PLATED_SOUP) {
                float dist = (float)(abs(env->items[i].x - (int)agent->x) + abs(env->items[i].y - (int)agent->y));
                if (dist < min_dist) {
                    min_dist = dist;
                    *dx = (env->items[i].x - agent->x) / (float)env->width;
                    *dy = (env->items[i].y - agent->y) / (float)env->height;
                }
            }
        }
    }
    else {
        float min_dist = 1000.0f;
        for (int y = 0; y < env->height; y++) {
            for (int x = 0; x < env->width; x++) {
                if (env->grid[y * env->width + x] == feature_type) {
                    float dist = (float)(abs(x - (int)agent->x) + abs(y - (int)agent->y));
                    if (dist < min_dist) {
                        min_dist = dist;
                        *dx = (x - agent->x) / (float)env->width;
                        *dy = (y - agent->y) / (float)env->height;
                    }
                }
            }
        }
    }
}


static void find_nearest_empty_counter(Overcooked* env, int agent_x, int agent_y, float* dx, float* dy) {
    float min_dist = 1000.0f;
    *dx = 0.0f;
    *dy = 0.0f;

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (env->grid[y * env->width + x] == COUNTER && get_item_at(env, x, y) == NULL) {
                float dist = (float)(abs(x - agent_x) + abs(y - agent_y));
                if (dist < min_dist) {
                    min_dist = dist;
                    *dx = (x - agent_x) / (float)env->width;
                    *dy = (y - agent_y) / (float)env->height;
                }
            }
        }
    }
}

static void compute_observations(Overcooked* env) {
    // 39-dimensional observation vector for each agent
    // Structure per agent:
    // - Player features: 34 dims (4 orientation + 4 held + 12 proximity + 2 nearest soup ingredients + 2 pot soup ingredients + 1 pot exist + 4 pot state + 1 cook time + 4 walls)
    // - Teammate relative position: 2 dims
    // - Absolute position: 2 dims
    // - Reward: 1 dim
    // Total: 39 dims
    // No tomatoes! Just onions for now...
    // TODO @mmbajo: Add tomatoes
    
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        float* obs = &env->observations[agent_idx * env->observation_size];
        int obs_idx = 0;
        
        // Clear observation
        for (int i = 0; i < env->observation_size; i++) {
            obs[i] = 0.0f;
        }
        
        // === PLAYER-SPECIFIC FEATURES (28 dims) ===
        
        // 1. Orientation (one-hot, 4 dims)
        obs[obs_idx + agent->facing_direction] = 1.0f;
        obs_idx += 4;
        
        // 2. Held object (one-hot: onion, soup, dish, tomato, empty - 5 dims but we use 4)
        if (agent->held_item == NO_ITEM) {
            obs[obs_idx + 3] = 1.0f;  // Empty
        } else if (agent->held_item == ONION) {
            obs[obs_idx + 0] = 1.0f;
        } else if (agent->held_item == PLATED_SOUP) {
            obs[obs_idx + 1] = 1.0f;  // Soup
        } else if (agent->held_item == PLATE) {
            obs[obs_idx + 2] = 1.0f;  // Dish
        }
        // Note: We don't use tomatoes in this version, keeping slot for compatibility
        obs_idx += 4;
        
        // 3. Proximity to key objects (dx, dy for each, 12 dims total)
        float dx, dy;

        // Nearest onion source (ingredient box) - returns (0,0) if holding onion
        compute_proximity_feature(env, agent, INGREDIENT_BOX, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest dish (plate box) - returns (0,0) if holding plate
        compute_proximity_feature(env, agent, PLATE_BOX, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest soup (plated soup) - returns (0,0) if holding soup or none exists
        compute_proximity_feature(env, agent, PLATED_SOUP, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest serving area
        compute_proximity_feature(env, agent, SERVING_AREA, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest empty counter - special case, needs custom handling
        find_nearest_empty_counter(env, agent->x, agent->y, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest pot (stove)
        compute_proximity_feature(env, agent, STOVE, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // 4. Nearest soup ingredients (2 dims: onions, tomatoes in nearest plated soup or held soup)
        // Check if agent is holding plated soup
        if (agent->held_item == PLATED_SOUP) {
            obs[obs_idx++] = agent->held_soup_onions / (float)MAX_INGREDIENTS;
            obs[obs_idx++] = agent->held_soup_tomatoes / (float)MAX_INGREDIENTS;
        } else {
            // Find nearest plated soup on counter
            Item* nearest_soup = NULL;
            float min_soup_dist = 1000.0f;
            for (int i = 0; i < env->num_items; i++) {
                if (env->items[i].type == PLATED_SOUP) {
                    float dist = (float)(abs((int)env->items[i].x - (int)agent->x) +
                                         abs((int)env->items[i].y - (int)agent->y));
                    if (dist < min_soup_dist) {
                        min_soup_dist = dist;
                        nearest_soup = &env->items[i];
                    }
                }
            }
            if (nearest_soup) {
                obs[obs_idx++] = nearest_soup->num_onions / (float)MAX_INGREDIENTS;
                obs[obs_idx++] = nearest_soup->num_tomatoes / (float)MAX_INGREDIENTS;
            } else {
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
            }
        }

        // 5. Pot soup ingredients (2 dims: onion count, always 0 for tomatoes in nearest pot)
        // Find nearest pot
        float min_pot_dist = 1000.0f;
        CookingPot* nearest_pot = NULL;
        
        for (int y = 0; y < env->height; y++) {
            for (int x = 0; x < env->width; x++) {
                if (env->grid[y * env->width + x] == STOVE) {
                    float dist = (float)(abs(x - (int)agent->x) + abs(y - (int)agent->y));
                    if (dist < min_pot_dist) {
                        min_pot_dist = dist;
                        nearest_pot = get_pot_at(env, x, y);
                    }
                }
            }
        }
        
        if (nearest_pot) {
            obs[obs_idx++] = nearest_pot->num_onions / (float)MAX_INGREDIENTS;
            obs[obs_idx++] = 0.0f;  // No tomatoes in our version
        } else {
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
        }
        
        // 6. Reachable pot existence (1 dim)
        obs[obs_idx++] = (nearest_pot != NULL) ? 1.0f : 0.0f;

        // 7. Pot state flags (4 dims: empty, full, cooking, ready)
        if (nearest_pot) {
            obs[obs_idx++] = (nearest_pot->ingredient_count == 0) ? 1.0f : 0.0f;  // Empty
            obs[obs_idx++] = (nearest_pot->ingredient_count == MAX_INGREDIENTS) ? 1.0f : 0.0f;  // Full (exactly MAX_INGREDIENTS)
            obs[obs_idx++] = (nearest_pot->cooking_state == COOKING) ? 1.0f : 0.0f;  // Cooking
            obs[obs_idx++] = (nearest_pot->cooking_state == COOKED) ? 1.0f : 0.0f;  // Ready
        } else {
            obs_idx += 4;  // Skip pot state if no pot found
        }

        // 8. Remaining cooking time (1 dim)
        if (nearest_pot && nearest_pot->cooking_state == COOKING) {
            float remaining = (COOKING_TIME - nearest_pot->cooking_progress) / (float)COOKING_TIME;
            obs[obs_idx++] = remaining;
        } else {
            obs[obs_idx++] = 0.0f;
        }

        // 9. Wall detection (4 dims: up, down, left, right)
        // Check each direction for walls, stoves, and counters (all are non-walkable)
        int wall_up = (agent->y > 0) ? env->grid[((int)agent->y - 1) * env->width + (int)agent->x] : WALL;
        int wall_down = (agent->y < env->height - 1) ? env->grid[((int)agent->y + 1) * env->width + (int)agent->x] : WALL;
        int wall_left = (agent->x > 0) ? env->grid[(int)agent->y * env->width + ((int)agent->x - 1)] : WALL;
        int wall_right = (agent->x < env->width - 1) ? env->grid[(int)agent->y * env->width + ((int)agent->x + 1)] : WALL;

        obs[obs_idx++] = (wall_up == WALL || wall_up == STOVE || wall_up == COUNTER) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_down == WALL || wall_down == STOVE || wall_down == COUNTER) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_left == WALL || wall_left == STOVE || wall_left == COUNTER) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_right == WALL || wall_right == STOVE || wall_right == COUNTER) ? 1.0f : 0.0f;

        // === TEAMMATE RELATIVE POSITION (2 dims) ===
        // Find teammate (other agent)
        int teammate_idx = (agent_idx == 0) ? 1 : 0;
        if (teammate_idx < env->num_agents) {
            Agent* teammate = &env->agents[teammate_idx];
            obs[obs_idx++] = (teammate->x - agent->x) / (float)env->width;
            obs[obs_idx++] = (teammate->y - agent->y) / (float)env->height;
        } else {
            // No teammate, set relative position to 0
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
        }

        // === ABSOLUTE POSITION (2 dims) ===
        obs[obs_idx++] = agent->x / (float)env->width;
        obs[obs_idx++] = agent->y / (float)env->height;

        // === REWARD (1 dim) ===
        obs[obs_idx++] = env->rewards[agent_idx];

        // Total should be 39 dims (34 player features + 2 teammate relative position + 2 absolute position + 1 reward)
        if (obs_idx != 39 && agent_idx == 0 && env->current_step == 0) {
            printf("Warning: Observation size mismatch! Expected 39, got %d\n", obs_idx);
        }
    }
}

static void handle_interaction(Overcooked* env, int agent_idx) {
    Agent* agent = &env->agents[agent_idx];
    int target_x = agent->x;
    int target_y = agent->y;
    
    switch (agent->facing_direction) {
        case 0: target_y -= 1; break; // Up
        case 1: target_y += 1; break; // Down
        case 2: target_x -= 1; break; // Left
        case 3: target_x += 1; break; // Right
    }
    
    if (target_x < 0 || target_x >= env->width || target_y < 0 || target_y >= env->height) {
        return;
    }
    
    int tile = env->grid[target_y * env->width + target_x];
    Item* item = get_item_at(env, target_x, target_y);
    CookingPot* pot = get_pot_at(env, target_x, target_y);
    
    if (tile == STOVE && pot != NULL) {
        if (agent->held_item == ONION || agent->held_item == TOMATO) {
            if (pot->cooking_state == NOT_COOKING && pot->ingredient_count < MAX_INGREDIENTS) {
                pot->ingredient_types[pot->ingredient_count] = agent->held_item;
                pot->ingredient_count++;
                if (agent->held_item == ONION) {
                    pot->num_onions++;
                    env->rewards[agent_idx] += 0.1f;
                } else if (agent->held_item == TOMATO) {
                    pot->num_tomatoes++;
                }
                agent->held_item = NO_ITEM;
            }
        }
        else if (agent->held_item == NO_ITEM && pot->ingredient_count > 0) {
            if (pot->cooking_state == NOT_COOKING) {
                pot->cooking_state = COOKING;
                pot->cooking_progress = 0;
                env->log.pots_started++;
                if (pot->num_onions == 3) {
                    env->rewards[agent_idx] += 0.1f;
                }
            }
            else if (pot->cooking_state == COOKED) {
                return;
            }
        }
        else if (agent->held_item == PLATE && pot->cooking_state == COOKED) {
            agent->held_item = PLATED_SOUP;
            agent->held_soup_onions = pot->num_onions;
            agent->held_soup_tomatoes = pot->num_tomatoes;
            agent->held_soup_total = pot->ingredient_count;

            env->rewards[agent_idx] += 0.1f;

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
    
    if (tile == SERVING_AREA && agent->held_item == PLATED_SOUP) {
        evaluate_dish_served(env, agent, agent_idx);
        
        agent->held_item = NO_ITEM;
        agent->held_soup_onions = 0;
        agent->held_soup_tomatoes = 0;
        agent->held_soup_total = 0;
        return;
    }
    
    if (agent->held_item != NO_ITEM) {
        if ((tile == COUNTER || tile == CUTTING_BOARD) && item == NULL) {
            if (agent->held_item == PLATED_SOUP) {
                add_item(env, agent->held_item, target_x, target_y);
                Item* placed_soup = get_item_at(env, target_x, target_y);
                if (placed_soup) {
                    placed_soup->num_onions = agent->held_soup_onions;
                    placed_soup->num_tomatoes = agent->held_soup_tomatoes;
                    placed_soup->total_ingredients = agent->held_soup_total;
                }
                agent->held_soup_onions = 0;
                agent->held_soup_tomatoes = 0;
                agent->held_soup_total = 0;
            } else {
                add_item(env, agent->held_item, target_x, target_y);
            }
            agent->held_item = NO_ITEM;
            env->log.items_dropped++;
        } else if ((tile == EMPTY) && item == NULL) {
            agent->held_item = NO_ITEM;
            env->log.items_dropped++;
        }
    }
    else {
        if (item != NULL) {
            if (item->type == PLATED_SOUP) {
                agent->held_soup_onions = item->num_onions;
                agent->held_soup_tomatoes = item->num_tomatoes;
                agent->held_soup_total = item->total_ingredients;
            }
            agent->held_item = item->type;
            remove_item(env, target_x, target_y);
        }
        else if (tile == INGREDIENT_BOX) {
            // TODO @mmbajo: What if we have Tomatoes as well?
            // Add logs for each ingredient type
            agent->held_item = ONION; // Always gives onions for now
            env->log.ingredients_picked++;
        }
        else if (tile == PLATE_BOX) {
            agent->held_item = PLATE;
            env->log.items_dropped++;
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
    for (int i = 0; i < env->num_agents; i++) {
        if (i != excluding_agent && (int)env->agents[i].x == x && (int)env->agents[i].y == y) {
            return 0;
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
    env->num_stoves = 0;
    for (int i = 0; i < env->width * env->height; i++) {
        if (env->grid[i] == STOVE) {
            env->num_stoves++;
        }
    }
    
    env->cooking_pots = calloc(env->num_stoves, sizeof(CookingPot));
    
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

static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx) {
    // Rule 1: Check if soup has exactly 3 onions -> actually the only rule atm
    int is_correct_recipe = (agent->held_soup_onions == 3);

    // You can add more rules here, e.g.:
    // int has_no_tomatoes = (agent->held_soup_tomatoes == 0);
    // int is_not_burnt = 1;  // Could track if soup was burnt
    // int served_quickly = (env->current_step < 100);
    
    if (is_correct_recipe) {
        float reward = env->reward_dish_served;
        // reward the particular agent for serving the dish
        env->rewards[agent_idx] += 5.0f;
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += reward; // reward all agents for serving the dish
        }
        
        env->log.dishes_served++;
        env->log.correct_dishes++;
        env->log.score += reward;
    } else {
        env->rewards[agent_idx] += 0.1f;
        for (int i = 0; i < env->num_agents; i++) {
            // TODO @mmbajo: Pls generalize this into a struct? for easy changing
            env->rewards[i] += 0.1f; // reward all agents for serving
        }
        env->log.wrong_dishes++;
    }
}

void c_reset(Overcooked* env) {
    env->current_step = 0;
    env->num_items = 0;
    parse_grid(env);

    env->log.episode_return = 0.0f;
    env->log.episode_length = 0.0f;
    env->log.dishes_served = 0.0f;
    env->log.cooperation_score = 0.0f;
    env->log.correct_dishes = 0.0f;
    env->log.wrong_dishes = 0.0f;
    env->log.ingredients_picked = 0.0f;
    env->log.pots_started = 0.0f;
    env->log.items_dropped = 0.0f;
    env->log.agent_collisions = 0.0f;
    env->log.cooking_time_efficiency = 0.0f;

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
    
    for (int i = 0; i < env->num_agents; i++) {
        if (i == 0) {
            env->agents[i].x = 1;
            env->agents[i].y = 2;
        } else if (i == 1) {
            env->agents[i].x = 3;
            env->agents[i].y = 2;
        } else {
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
    
}

void c_step(Overcooked* env) {
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
                for (int j = 0; j < env->num_agents; j++) {
                    if (j != i && (int)env->agents[j].x == new_x && (int)env->agents[j].y == new_y) {
                        env->log.agent_collisions++;
                        break;
                    }
                }
            }
        }
    }
    
    update_cooking(env);

    for (int i = 0; i < env->num_agents; i++) {
        env->log.episode_return += env->rewards[i];
    }

    env->current_step++;
    env->log.episode_length++;

    if (env->current_step >= env->max_steps) {
        for (int i = 0; i < env->num_agents; i++) {
            env->terminals[i] = 1;
        }
        env->log.perf += env->log.dishes_served / 20.0f;  // Normalize to 0-1 (20 dishes would be excellent)
        env->log.score += env->log.episode_return;
        env->log.n += 1;
    }
    
    compute_observations(env);
}

void c_render(Overcooked* env) {
    if (env->client == NULL) {
        int window_width = env->width * env->grid_size + 350;
        int window_height = env->height * env->grid_size + 80;
        InitWindow(window_width, window_height, "PufferLib Overcooked");
        SetTargetFPS(4);
        env->client = (Client*)calloc(1, sizeof(Client));
        
        env->client->floor = LoadTexture("pufferlib/resources/overcooked/terrain/floor.png");
        env->client->counter = LoadTexture("pufferlib/resources/overcooked/terrain/counter.png");
        env->client->pot = LoadTexture("pufferlib/resources/overcooked/terrain/pot.png");
        env->client->serve = LoadTexture("pufferlib/resources/overcooked/terrain/serve.png");
        env->client->onions_box = LoadTexture("pufferlib/resources/overcooked/terrain/onions.png");
        env->client->tomatoes_box = LoadTexture("pufferlib/resources/overcooked/terrain/tomatoes.png");
        env->client->dishes_box = LoadTexture("pufferlib/resources/overcooked/terrain/dishes.png");
        env->client->wall = LoadTexture("pufferlib/resources/overcooked/terrain/counter.png");

        env->client->onion = LoadTexture("pufferlib/resources/overcooked/objects/onion.png");
        env->client->tomato = LoadTexture("pufferlib/resources/overcooked/objects/tomato.png");
        env->client->dish = LoadTexture("pufferlib/resources/overcooked/objects/dish.png");
        env->client->soup_onion = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-cooked.png");
        env->client->soup_tomato = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-cooked.png");
        env->client->soup_onion_dish = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-dish.png");
        env->client->soup_tomato_dish = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-dish.png");
        
        env->client->soup_onion_cooking_1 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-1-cooking.png");
        env->client->soup_onion_cooking_2 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-2-cooking.png");
        env->client->soup_onion_cooking_3 = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-3-cooking.png");
        env->client->soup_onion_cooked = LoadTexture("pufferlib/resources/overcooked/objects/soup-onion-cooked.png");
        env->client->soup_tomato_cooking_1 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-1-cooking.png");
        env->client->soup_tomato_cooking_2 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-2-cooking.png");
        env->client->soup_tomato_cooking_3 = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-3-cooking.png");
        env->client->soup_tomato_cooked = LoadTexture("pufferlib/resources/overcooked/objects/soup-tomato-cooked.png");
        
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
    
    DrawText(TextFormat("Step: %d / %d", env->current_step, env->max_steps), 10, 10, 20, BLACK);
    DrawText(TextFormat("Dishes Served: %d", (int)env->log.dishes_served), 10, 35, 20, BLACK);
    DrawText("Recipe: 3 Onions", 10, 60, 16, DARKGRAY);
    
    int grid_offset_y = 80;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            Rectangle dest = {x * env->grid_size, y * env->grid_size + grid_offset_y, env->grid_size, env->grid_size};
            
            if (env->client->floor.id != 0) {
                DrawTexturePro(env->client->floor, 
                    (Rectangle){0, 0, env->client->floor.width, env->client->floor.height},
                    dest, (Vector2){0, 0}, 0, WHITE);
            }
            
            Texture2D* texture = NULL;
            switch (env->grid[idx]) {
                case COUNTER:
                    texture = &env->client->counter;
                    break;
                case STOVE:
                    texture = &env->client->pot;
                    break;
                case CUTTING_BOARD:
                    texture = &env->client->counter;
                    break;
                case INGREDIENT_BOX:
                    texture = &env->client->onions_box;
                    break;
                case SERVING_AREA:
                    texture = &env->client->serve;
                    break;
                case PLATE_BOX:
                    texture = &env->client->dishes_box;
                    break;
                case WALL:
                    texture = &env->client->wall;
                    break;
            }
            
            if (texture && texture->id != 0) {
                DrawTexturePro(*texture,
                    (Rectangle){0, 0, texture->width, texture->height},
                    dest, (Vector2){0, 0}, 0, WHITE);
            }
            
            if (env->grid[idx] == STOVE) {
                CookingPot* pot = get_pot_at(env, x, y);
                if (pot && pot->ingredient_count > 0) {
                    Texture2D* cooking_texture = NULL;
                    
                    bool is_onion_soup = (pot->num_onions >= pot->num_tomatoes);
                    
                    if (pot->cooking_state == COOKING) {
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
                        DrawText("READY!", x * env->grid_size + 5,
                               y * env->grid_size + grid_offset_y + env->grid_size - 10,
                               8, GREEN);
                    }
                    else if (pot->cooking_state == BURNT) {
                        cooking_texture = is_onion_soup ? &env->client->soup_onion_cooked : 
                                                          &env->client->soup_tomato_cooked;
                        DrawText("BURNT!", x * env->grid_size + 5,
                               y * env->grid_size + grid_offset_y + env->grid_size - 10,
                               8, RED);
                    }
                    else if (pot->cooking_state == NOT_COOKING) {
                        cooking_texture = is_onion_soup ? &env->client->soup_onion_cooking_1 : 
                                                          &env->client->soup_tomato_cooking_1;
                    }
                    
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
    
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        Texture2D* chef_texture = NULL;
        
        if (agent->held_item == NO_ITEM) {
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
        
        if (chef_texture && chef_texture->id != 0) {
            Rectangle dest = {
                agent->x * env->grid_size,
                agent->y * env->grid_size + grid_offset_y,
                env->grid_size,
                env->grid_size
            };
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
            Color agent_color = get_agent_color(agent->held_item);
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
            
            DrawText(TextFormat("%d", agent_idx + 1), 
                     agent->x * env->grid_size + 2,
                     agent->y * env->grid_size + grid_offset_y + 2,
                     10, BLACK);
        }
    }

    int obs_panel_x = env->width * env->grid_size + 10;
    int obs_panel_y = grid_offset_y;

    if (env->num_agents > 0) {
        float* obs = &env->observations[0];

        DrawText("=== OBSERVATION ARRAY (39 dims) ===", obs_panel_x, obs_panel_y, 11, BLACK);
        obs_panel_y += 18;

        DrawText("-- PLAYER (0-33) --", obs_panel_x, obs_panel_y, 10, DARKGREEN);
        obs_panel_y += 13;

        DrawText(TextFormat("[0-3] Orient: %.0f %.0f %.0f %.0f",
                 obs[0], obs[1], obs[2], obs[3]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 11;

        DrawText(TextFormat("[4-7] Held: %.0f %.0f %.0f %.0f",
                 obs[4], obs[5], obs[6], obs[7]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 11;

        DrawText(TextFormat("[8-9] Onion: %.2f, %.2f", obs[8], obs[9]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
        DrawText(TextFormat("[10-11] Dish: %.2f, %.2f", obs[10], obs[11]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
        DrawText(TextFormat("[12-13] Soup: %.2f, %.2f", obs[12], obs[13]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
        DrawText(TextFormat("[14-15] Serve: %.2f, %.2f", obs[14], obs[15]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
        DrawText(TextFormat("[16-17] Empty: %.2f, %.2f", obs[16], obs[17]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
        DrawText(TextFormat("[18-19] Pot: %.2f, %.2f", obs[18], obs[19]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 11;

        DrawText(TextFormat("[20-21] SoupIngr: %.2f, %.2f", obs[20], obs[21]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[22-23] PotIngr: %.2f, %.2f", obs[22], obs[23]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[24] PotExists: %.0f", obs[24]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[25-28] PotState: %.0f %.0f %.0f %.0f",
                 obs[25], obs[26], obs[27], obs[28]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[29] CookTime: %.2f", obs[29]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[30-33] Walls: %.0f %.0f %.0f %.0f",
                 obs[30], obs[31], obs[32], obs[33]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 13;

        DrawText("-- TEAMMATE (34-35) --", obs_panel_x, obs_panel_y, 10, DARKBLUE);
        obs_panel_y += 13;

        if (env->num_agents > 1) {
            DrawText(TextFormat("[34-35] T.RelPos: %.2f, %.2f", obs[34], obs[35]),
                     obs_panel_x, obs_panel_y, 9, BLACK);
            obs_panel_y += 10;
        } else {
            DrawText("No teammate", obs_panel_x, obs_panel_y, 9, GRAY);
            obs_panel_y += 10;
        }

        obs_panel_y += 3;
        DrawText("-- MISC (36-38) --", obs_panel_x, obs_panel_y, 10, DARKGRAY);
        obs_panel_y += 13;

        DrawText(TextFormat("[36-37] AbsPos: %.3f, %.3f", obs[36], obs[37]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;

        DrawText(TextFormat("[38] Reward: %.2f", obs[38]),
                 obs_panel_x, obs_panel_y, 9, BLACK);
        obs_panel_y += 10;
    }

    if (env->num_agents > 0) {
        Agent* agent = &env->agents[0];
        float* obs = &env->observations[0];

        int agent_screen_x = agent->x * env->grid_size + env->grid_size/2;
        int agent_screen_y = agent->y * env->grid_size + grid_offset_y + env->grid_size/2;

        float dx_onion = obs[8] * env->width;
        float dy_onion = obs[9] * env->height;
        if (dx_onion != 0 || dy_onion != 0) {
            DrawLine(agent_screen_x, agent_screen_y,
                    agent_screen_x + dx_onion * env->grid_size,
                    agent_screen_y + dy_onion * env->grid_size,
                    (Color){0, 200, 0, 100});
        }

        float dx_serve = obs[14] * env->width;
        float dy_serve = obs[15] * env->height;
        if (dx_serve != 0 || dy_serve != 0) {
            DrawLine(agent_screen_x, agent_screen_y,
                    agent_screen_x + dx_serve * env->grid_size,
                    agent_screen_y + dy_serve * env->grid_size,
                    (Color){0, 0, 200, 100});
        }

        float dx_pot = obs[18] * env->width;
        float dy_pot = obs[19] * env->height;
        if (dx_pot != 0 || dy_pot != 0) {
            DrawLine(agent_screen_x, agent_screen_y,
                    agent_screen_x + dx_pot * env->grid_size,
                    agent_screen_y + dy_pot * env->grid_size,
                    (Color){200, 0, 0, 100});
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
        UnloadTexture(env->client->floor);
        UnloadTexture(env->client->counter);
        UnloadTexture(env->client->pot);
        UnloadTexture(env->client->serve);
        UnloadTexture(env->client->onions_box);
        UnloadTexture(env->client->tomatoes_box);
        UnloadTexture(env->client->dishes_box);
        UnloadTexture(env->client->wall);
        
        UnloadTexture(env->client->onion);
        UnloadTexture(env->client->tomato);
        UnloadTexture(env->client->dish);
        UnloadTexture(env->client->soup_onion);
        UnloadTexture(env->client->soup_tomato);
        UnloadTexture(env->client->soup_onion_dish);
        UnloadTexture(env->client->soup_tomato_dish);
        
        UnloadTexture(env->client->soup_onion_cooking_1);
        UnloadTexture(env->client->soup_onion_cooking_2);
        UnloadTexture(env->client->soup_onion_cooking_3);
        UnloadTexture(env->client->soup_onion_cooked);
        UnloadTexture(env->client->soup_tomato_cooking_1);
        UnloadTexture(env->client->soup_tomato_cooking_2);
        UnloadTexture(env->client->soup_tomato_cooking_3);
        UnloadTexture(env->client->soup_tomato_cooked);
        
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