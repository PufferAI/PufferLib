/* Overcooked Logic: Game logic functions (parsing, interaction, movement).
 */

#ifndef OVERCOOKED_LOGIC_H
#define OVERCOOKED_LOGIC_H

#include "overcooked_types.h"
#include "overcooked_items.h"

// Forward declaration for circular dependency
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

#endif // OVERCOOKED_LOGIC_H
