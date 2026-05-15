/* Overcooked Observations: Observation computation functions.
 */

#ifndef OVERCOOKED_OBS_H
#define OVERCOOKED_OBS_H

#include "overcooked_types.h"
#include "overcooked_items.h"

static Item* find_nearest_plated_soup(Overcooked* env, Agent* agent, float* dx, float* dy) {
    *dx = 0.0f;
    *dy = 0.0f;

    if (agent->held_item == PLATED_SOUP) return NULL;

    Item* nearest = NULL;
    float min_dist = 1000.0f;
    for (int i = 0; i < env->num_items; i++) {
        if (env->items[i].type == PLATED_SOUP) {
            float dist = (float)(abs(env->items[i].x - (int)agent->x) + abs(env->items[i].y - (int)agent->y));
            if (dist < min_dist) {
                min_dist = dist;
                nearest = &env->items[i];
                *dx = (env->items[i].x - agent->x) * env->cache.inv_width;
                *dy = (env->items[i].y - agent->y) * env->cache.inv_height;
            }
        }
    }
    return nearest;
}

static void find_nearest_item_by_type(Overcooked* env, Agent* agent,
                                       int item_type, float* dx, float* dy) {
    *dx = 0.0f;
    *dy = 0.0f;

    if (agent->held_item == item_type) return;

    float min_dist = 1000.0f;
    for (int i = 0; i < env->num_items; i++) {
        if (env->items[i].type == item_type) {
            float dist = (float)(abs(env->items[i].x - (int)agent->x) +
                                 abs(env->items[i].y - (int)agent->y));
            if (dist < min_dist) {
                min_dist = dist;
                *dx = (env->items[i].x - agent->x) * env->cache.inv_width;
                *dy = (env->items[i].y - agent->y) * env->cache.inv_height;
            }
        }
    }
}

// Cached version: iterate over precomputed tile positions instead of scanning grid
static void compute_tile_proximity_cached(Overcooked* env, Agent* agent,
                                          int* positions, int count,
                                          float* dx, float* dy) {
    *dx = 0.0f;
    *dy = 0.0f;

    int min_dist = 1000;
    int best_x = 0, best_y = 0;

    for (int i = 0; i < count; i++) {
        int x = positions[i * 2];
        int y = positions[i * 2 + 1];
        int dist = abs(x - (int)agent->x) + abs(y - (int)agent->y);
        if (dist < min_dist) {
            min_dist = dist;
            best_x = x;
            best_y = y;
        }
    }

    if (min_dist < 1000) {
        *dx = (best_x - agent->x) * env->cache.inv_width;
        *dy = (best_y - agent->y) * env->cache.inv_height;
    }
}


static void find_nearest_empty_counter(Overcooked* env, int agent_x, int agent_y, float* dx, float* dy) {
    *dx = 0.0f;
    *dy = 0.0f;
    int min_dist = 1000;

    // Iterate cached counter positions instead of scanning entire grid
    for (int i = 0; i < env->cache.counter_count; i++) {
        int x = env->cache.counter_positions[i * 2];
        int y = env->cache.counter_positions[i * 2 + 1];

        if (env->item_grid[y * env->width + x] < 0) {
            int dist = abs(x - agent_x) + abs(y - agent_y);
            if (dist < min_dist) {
                min_dist = dist;
                *dx = (x - agent_x) * env->cache.inv_width;
                *dy = (y - agent_y) * env->cache.inv_height;
            }
        }
    }
}

static void compute_observations(Overcooked* env) {
    // 43-dimensional observation vector for each agent
    // Structure per agent:
    // - Player features: 38 dims (4 orientation + 4 held + 16 proximity + 2 nearest soup ingredients + 2 pot soup ingredients + 1 pot exist + 4 pot state + 1 cook time + 4 walls)
    // - Teammate relative position: 2 dims
    // - Absolute position: 2 dims
    // - Reward: 1 dim
    // Total: 43 dims
    // Proximity: onion box, plate box, plated soup, serving, empty counter, pot, pickable onion, pickable plate

    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        float* obs = &env->observations[agent_idx * env->observation_size];
        int obs_idx = 0;

        memset(obs, 0, env->observation_size * sizeof(float));

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

        // 3. Proximity to key objects (dx, dy for each, 16 dims total)
        float dx, dy;

        // Nearest onion source (ingredient box) - returns (0,0) if holding onion
        if (agent->held_item == ONION) {
            dx = 0.0f;
            dy = 0.0f;
        } else {
            compute_tile_proximity_cached(env, agent,
                env->cache.ingredient_box_positions, env->cache.ingredient_box_count,
                &dx, &dy);
        }
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest dish (plate box) - returns (0,0) if holding plate
        if (agent->held_item == PLATE) {
            dx = 0.0f;
            dy = 0.0f;
        } else {
            compute_tile_proximity_cached(env, agent,
                env->cache.plate_box_positions, env->cache.plate_box_count,
                &dx, &dy);
        }
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest soup (plated soup) - returns (0,0) if holding soup or none exists
        Item* nearest_soup = find_nearest_plated_soup(env, agent, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest serving area
        compute_tile_proximity_cached(env, agent,
            env->cache.serving_area_positions, env->cache.serving_area_count,
            &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest empty counter - special case, needs custom handling
        find_nearest_empty_counter(env, agent->x, agent->y, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest pot (stove)
        compute_tile_proximity_cached(env, agent,
            env->cache.stove_positions, env->cache.stove_count,
            &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest pickable onion on counter (not in box)
        find_nearest_item_by_type(env, agent, ONION, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // Nearest pickable plate on counter (not in box)
        find_nearest_item_by_type(env, agent, PLATE, &dx, &dy);
        obs[obs_idx++] = dx;
        obs[obs_idx++] = dy;

        // 4. Nearest soup ingredients (2 dims: onions, tomatoes in nearest plated soup or held soup)
        if (agent->held_item == PLATED_SOUP) {
            obs[obs_idx++] = agent->held_soup_onions / (float)MAX_INGREDIENTS;
            obs[obs_idx++] = agent->held_soup_tomatoes / (float)MAX_INGREDIENTS;
        } else if (nearest_soup) {
            obs[obs_idx++] = nearest_soup->num_onions / (float)MAX_INGREDIENTS;
            obs[obs_idx++] = nearest_soup->num_tomatoes / (float)MAX_INGREDIENTS;
        } else {
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
        }

        // 5. Pot soup ingredients (2 dims: onion count, always 0 for tomatoes in nearest pot)
        // Find nearest pot using cached stove positions
        int min_pot_dist = 1000;
        CookingPot* nearest_pot = NULL;

        for (int i = 0; i < env->cache.stove_count; i++) {
            int x = env->cache.stove_positions[i * 2];
            int y = env->cache.stove_positions[i * 2 + 1];
            int dist = abs(x - (int)agent->x) + abs(y - (int)agent->y);
            if (dist < min_pot_dist) {
                min_pot_dist = dist;
                nearest_pot = get_pot_at(env, x, y);
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
        // Check each direction for any non-EMPTY tile (walls, stoves, counters, serving area, ingredient box, plate box, cutting board - all are non-walkable)
        int wall_up = (agent->y > 0) ? env->grid[((int)agent->y - 1) * env->width + (int)agent->x] : WALL;
        int wall_down = (agent->y < env->height - 1) ? env->grid[((int)agent->y + 1) * env->width + (int)agent->x] : WALL;
        int wall_left = (agent->x > 0) ? env->grid[(int)agent->y * env->width + ((int)agent->x - 1)] : WALL;
        int wall_right = (agent->x < env->width - 1) ? env->grid[(int)agent->y * env->width + ((int)agent->x + 1)] : WALL;

        obs[obs_idx++] = (wall_up != EMPTY) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_down != EMPTY) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_left != EMPTY) ? 1.0f : 0.0f;
        obs[obs_idx++] = (wall_right != EMPTY) ? 1.0f : 0.0f;

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

        // Total should be 43 dims (38 player features + 2 teammate relative position + 2 absolute position + 1 reward)
        // Debug check removed - was only useful on first step
    }
}

#endif // OVERCOOKED_OBS_H
