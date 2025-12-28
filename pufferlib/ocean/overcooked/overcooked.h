/* Overcooked: a multi-agent cooking coordination environment.
 * Agents can walk around, pick up items, and put down items.
 */

#include "overcooked_types.h"
#include "overcooked_items.h"
#include "overcooked_obs.h"

// Forward declarations
static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx);

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
                    env->rewards[agent_idx] += env->rewards_config.ingredient_added;
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
                    env->rewards[agent_idx] += env->rewards_config.pot_started;
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

            env->rewards[agent_idx] += env->rewards_config.soup_plated;

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
            env->rewards[agent_idx] += env->rewards_config.ingredient_picked;
        }
        else if (tile == PLATE_BOX) {
            agent->held_item = PLATE;
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

static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx) {
    // Rule 1: Check if soup has exactly 3 onions -> actually the only rule atm
    int is_correct_recipe = (agent->held_soup_onions == 3);

    // You can add more rules here, e.g.:
    // int has_no_tomatoes = (agent->held_soup_tomatoes == 0);
    // int served_quickly = (env->current_step < 100);

    if (is_correct_recipe) {
        // reward the particular agent for serving the dish
        env->rewards[agent_idx] += env->rewards_config.dish_served_agent;
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += env->rewards_config.dish_served_whole_team; // reward all agents for serving the dish
        }

        env->log.episode_length += agent->ticks_since_reward;
        agent->ticks_since_reward = 0;

        env->log.episode_return += env->rewards_config.dish_served_whole_team;
        env->log.correct_dishes++;
        env->log.score += 1.0f;
        env->log.perf += 1.0f;
        env->log.n++;
    } else {
        env->rewards[agent_idx] += env->rewards_config.wrong_dish_served;
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += env->rewards_config.wrong_dish_served; // reward all agents for serving
        }
        env->log.episode_return += env->rewards_config.wrong_dish_served;
        env->log.wrong_dishes++;
    }
    env->log.dishes_served++;
}

void c_reset(Overcooked* env) {
    env->num_items = 0;
    parse_grid(env);


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
        env->agents[i].ticks_since_reward = 0;

        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }

    compute_observations(env);
}

void c_step(Overcooked* env) {
    for (int i = 0; i < env->num_agents; i++) {
        int action = env->actions[i];
        env->rewards[i] = env->rewards_config.step_penalty;
        env->agents[i].ticks_since_reward++;

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
        if (env->agents[i].ticks_since_reward % 512 == 0 && env->agents[i].ticks_since_reward > 0) {
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
            env->agents[i].held_soup_onions = 0;
            env->agents[i].held_soup_tomatoes = 0;
            env->agents[i].held_soup_total = 0;
        }
    }

    compute_observations(env);
}

void c_render(Overcooked* env) {
    if (env->client == NULL) {
        int window_width = env->width * env->grid_size + 350;
        int window_height = env->height * env->grid_size + 80;
        InitWindow(window_width, window_height, "PufferLib Overcooked");
        SetTargetFPS(16);
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
    
    DrawText(TextFormat("Correct Dishes: %d", (int)env->log.n), 10, 10, 20, BLACK);
    DrawText(TextFormat("Total Dishes: %d", (int)env->log.dishes_served), 10, 35, 20, BLACK);
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
                        DrawTexturePro(*cooking_texture,
                            (Rectangle){0, 0, cooking_texture->width, cooking_texture->height},
                            pot_dest, (Vector2){0, 0}, 0, WHITE);
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