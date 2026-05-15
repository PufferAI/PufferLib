/* Overcooked: a multi-agent cooking coordination environment.
 * Agents can walk around, pick up items, and put down items.
 */

#ifndef OVERCOOKED_H
#define OVERCOOKED_H

#include "overcooked_types.h"
#include "overcooked_items.h"
#include "overcooked_obs.h"
#include "overcooked_logic.h"
#include "overcooked_render.h"

static void init(Overcooked* env) {
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    env->width = layout->width;
    env->height = layout->height;
    env->grid = calloc(env->width * env->height, sizeof(char));
    env->max_items = 20;
    env->items = calloc(env->max_items, sizeof(Item));
    env->num_items = 0;
    env->agents = calloc(env->num_agents, sizeof(Agent));
    parse_grid(env);
    init_static_cache(env);
    init_cooking_pots(env);
    init_pot_indices(env);
    init_item_grid(env);
    env->client = NULL;

    memset(&env->log, 0, sizeof(Log));
}

void c_reset(Overcooked* env) {
    env->num_items = 0;
    reset_item_grid(env);
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
    
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    for (int i = 0; i < env->num_agents; i++) {
        if (i < layout->num_spawns) {
            env->agents[i].x = layout->spawn_positions[i * 2];
            env->agents[i].y = layout->spawn_positions[i * 2 + 1];
        } else {
            env->agents[i].x = 1 + (i % (env->width - 2));
            env->agents[i].y = 1 + (i / (env->width - 2));
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

    env->agent_position_mask = 0;
    for (int i = 0; i < env->num_agents; i++) {
        set_agent_position(env, env->agents[i].x, env->agents[i].y);
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
                clear_agent_position(env, agent->x, agent->y);
                agent->x = new_x;
                agent->y = new_y;
                set_agent_position(env, new_x, new_y);
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

    const LayoutInfo* layout = get_layout_info(env->layout_id);
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].ticks_since_reward % 512 == 0 && env->agents[i].ticks_since_reward > 0) {
            clear_agent_position(env, env->agents[i].x, env->agents[i].y);
            if (i < layout->num_spawns) {
                env->agents[i].x = layout->spawn_positions[i * 2];
                env->agents[i].y = layout->spawn_positions[i * 2 + 1];
            } else {
                env->agents[i].x = 1 + (i % (env->width - 2));
                env->agents[i].y = 1 + (i / (env->width - 2));
            }
            set_agent_position(env, env->agents[i].x, env->agents[i].y);
            env->agents[i].held_item = NO_ITEM;
            env->agents[i].held_soup_onions = 0;
            env->agents[i].held_soup_tomatoes = 0;
            env->agents[i].held_soup_total = 0;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        env->log.episode_return += env->rewards[i];
    }

    compute_observations(env);
}

void c_close(Overcooked* env) {
    free(env->grid);
    free(env->items);
    free(env->agents);
    free(env->cooking_pots);
    free(env->pot_index_grid);
    free(env->item_grid);
    if (env->client != NULL) {
        unload_textures(env->client);
        free(env->client);
    }
}

#endif // OVERCOOKED_H