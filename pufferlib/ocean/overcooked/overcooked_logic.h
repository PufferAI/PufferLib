/* Overcooked Logic: Game logic functions (parsing, interaction, movement).
 */

#ifndef OVERCOOKED_LOGIC_H
#define OVERCOOKED_LOGIC_H

#include "overcooked_types.h"
#include "overcooked_items.h"

// Forward declaration for circular dependency
static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx);

static void parse_grid(Overcooked* env) {
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            char tile = get_layout_tile(layout, x, y);
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

static void init_static_cache(Overcooked* env) {
    // Precompute normalization factors
    env->cache.inv_width = 1.0f / env->width;
    env->cache.inv_height = 1.0f / env->height;

    // Reset counts
    env->cache.ingredient_box_count = 0;
    env->cache.plate_box_count = 0;
    env->cache.serving_area_count = 0;
    env->cache.stove_count = 0;
    env->cache.counter_count = 0;

    // Scan grid once and cache all static tile positions
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int tile = env->grid[y * env->width + x];
            switch (tile) {
                case INGREDIENT_BOX:
                    env->cache.ingredient_box_positions[env->cache.ingredient_box_count * 2] = x;
                    env->cache.ingredient_box_positions[env->cache.ingredient_box_count * 2 + 1] = y;
                    env->cache.ingredient_box_count++;
                    break;
                case PLATE_BOX:
                    env->cache.plate_box_positions[env->cache.plate_box_count * 2] = x;
                    env->cache.plate_box_positions[env->cache.plate_box_count * 2 + 1] = y;
                    env->cache.plate_box_count++;
                    break;
                case SERVING_AREA:
                    env->cache.serving_area_positions[env->cache.serving_area_count * 2] = x;
                    env->cache.serving_area_positions[env->cache.serving_area_count * 2 + 1] = y;
                    env->cache.serving_area_count++;
                    break;
                case STOVE:
                    env->cache.stove_positions[env->cache.stove_count * 2] = x;
                    env->cache.stove_positions[env->cache.stove_count * 2 + 1] = y;
                    env->cache.stove_count++;
                    break;
                case COUNTER:
                    env->cache.counter_positions[env->cache.counter_count * 2] = x;
                    env->cache.counter_positions[env->cache.counter_count * 2 + 1] = y;
                    env->cache.counter_count++;
                    break;
            }
        }
    }
}

static inline void set_agent_position(Overcooked* env, int x, int y) {
    env->agent_position_mask |= (1ULL << (y * env->width + x));
}

static inline void clear_agent_position(Overcooked* env, int x, int y) {
    env->agent_position_mask &= ~(1ULL << (y * env->width + x));
}

static inline int is_agent_at(Overcooked* env, int x, int y) {
    return (env->agent_position_mask >> (y * env->width + x)) & 1;
}

static int is_valid_position(Overcooked* env, int x, int y, int excluding_agent) {
    (void)excluding_agent;
    if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
        return 0;
    }
    if (env->grid[y * env->width + x] != EMPTY) {
        return 0;
    }
    if (is_agent_at(env, x, y)) {
        return 0;
    }
    return 1;
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
            env->rewards[agent_idx] += env->rewards_config.plate_picked;
        }
    }
}

static void evaluate_dish_served(Overcooked* env, Agent* agent, int agent_idx) {
    int is_correct_recipe = (agent->held_soup_onions == 3);

    if (is_correct_recipe) {
        env->rewards[agent_idx] += env->rewards_config.dish_served_agent;
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += env->rewards_config.dish_served_whole_team;
        }
        env->log.episode_length += agent->ticks_since_reward;
        env->log.score += 25.0 / agent->ticks_since_reward;
        env->log.perf += 25.0 / agent->ticks_since_reward;
        agent->ticks_since_reward = 0;
        env->log.correct_dishes++;
        env->log.n++;
    } else {
        env->rewards[agent_idx] += env->rewards_config.wrong_dish_served;
        for (int i = 0; i < env->num_agents; i++) {
            env->rewards[i] += env->rewards_config.wrong_dish_served;
        }
        env->log.wrong_dishes++;
    }
    env->log.dishes_served++;
}

#endif // OVERCOOKED_LOGIC_H
