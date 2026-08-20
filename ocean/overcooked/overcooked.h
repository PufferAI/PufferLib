/* Overcooked: a multi-agent cooking coordination environment.
 * Agents can walk around, pick up items, and put down items.
 */

#ifndef OVERCOOKED_H
#define OVERCOOKED_H

typedef float obs_t;
#ifndef OBS_SIZE
#define ACT_SIZES {6}
#define OBS_SIZE 43
#define NUM_ATNS 1
#endif

#include "overcooked_types.h"
// ACT_SIZES defined in overcooked_types.h
#include "overcooked_items.h"
#include "overcooked_obs.h"
#include "overcooked_logic.h"
#include "overcooked_render.h"

static void init(Overcooked* env) {
    const LayoutInfo* layout = get_layout_info(env->layout_id);
    env->width = layout->width;
    env->height = layout->height;
    env->grid = (char*)calloc(env->width * env->height, sizeof(char));
    env->max_items = 20;
    env->items = (Item*)calloc(env->max_items, sizeof(Item));
    env->num_items = 0;
    env->chefs = (Chef*)calloc(env->num_agents, sizeof(Chef));
    parse_grid(env);
    init_static_cache(env);
    init_cooking_pots(env);
    init_pot_indices(env);
    init_item_grid(env);
    env->client = NULL;

    memset(&env->log, 0, sizeof(Log));
}

void puf_reset(Overcooked* env) {
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
            env->chefs[i].x = layout->spawn_positions[i * 2];
            env->chefs[i].y = layout->spawn_positions[i * 2 + 1];
        } else {
            env->chefs[i].x = 1 + (i % (env->width - 2));
            env->chefs[i].y = 1 + (i / (env->width - 2));
        }
        env->chefs[i].held_item = NO_ITEM;
        env->chefs[i].facing_direction = 0;
        env->chefs[i].held_soup_onions = 0;
        env->chefs[i].held_soup_tomatoes = 0;
        env->chefs[i].held_soup_total = 0;
        env->chefs[i].ticks_since_reward = 0;

        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0;
    }

    env->agent_position_mask = 0;
    for (int i = 0; i < env->num_agents; i++) {
        set_agent_position(env, env->chefs[i].x, env->chefs[i].y);
    }

    compute_observations(env);
}

void puf_step(Overcooked* env) {
    // Team serve reward is written onto every agent; must not re-init mid-loop.
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = env->rewards_config.step_penalty;
        env->chefs[i].ticks_since_reward++;
    }

    for (int i = 0; i < env->num_agents; i++) {
        int action = (int)env->agents[i].actions[0];

        Chef* agent = &env->chefs[i];
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
                    if (j != i && (int)env->chefs[j].x == new_x && (int)env->chefs[j].y == new_y) {
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
        if (env->chefs[i].ticks_since_reward % 512 == 0 && env->chefs[i].ticks_since_reward > 0) {
            clear_agent_position(env, env->chefs[i].x, env->chefs[i].y);
            if (i < layout->num_spawns) {
                env->chefs[i].x = layout->spawn_positions[i * 2];
                env->chefs[i].y = layout->spawn_positions[i * 2 + 1];
            } else {
                env->chefs[i].x = 1 + (i % (env->width - 2));
                env->chefs[i].y = 1 + (i / (env->width - 2));
            }
            set_agent_position(env, env->chefs[i].x, env->chefs[i].y);
            env->chefs[i].held_item = NO_ITEM;
            env->chefs[i].held_soup_onions = 0;
            env->chefs[i].held_soup_tomatoes = 0;
            env->chefs[i].held_soup_total = 0;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        env->log.episode_return += env->agents[i].rewards[0];
    }

    compute_observations(env);
}

void puf_close(Overcooked* env) {
    free(env->grid);
    free(env->items);
    free(env->chefs);
    free(env->cooking_pots);
    free(env->pot_index_grid);
    free(env->item_grid);
    if (env->client != NULL) {
        unload_textures(env->client);
        free(env->client);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->layout_id = (LayoutType)dict_get(kwargs, "layout");
    env->num_agents = dict_get(kwargs, "num_agents");
    env->grid_size = dict_get(kwargs, "grid_size");
    env->observation_size = OBS_SIZE;
    if (env->num_agents > OVERCOOKED_MAX_AGENTS) {
        fprintf(stderr, "overcooked: num_agents too large\n");
        exit(1);
    }
    env->rewards_config.dish_served_whole_team = dict_get(kwargs, "reward_dish_served_whole_team");
    env->rewards_config.dish_served_agent = dict_get(kwargs, "reward_dish_served_agent");
    env->rewards_config.pot_started = dict_get(kwargs, "reward_pot_started");
    env->rewards_config.ingredient_added = dict_get(kwargs, "reward_ingredient_added");
    env->rewards_config.ingredient_picked = dict_get(kwargs, "reward_ingredient_picked");
    env->rewards_config.plate_picked = dict_get(kwargs, "reward_plate_picked");
    env->rewards_config.soup_plated = dict_get(kwargs, "reward_soup_plated");
    env->rewards_config.wrong_dish_served = dict_get(kwargs, "reward_wrong_dish_served");
    env->rewards_config.step_penalty = dict_get(kwargs, "reward_step_penalty");
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "dishes_served", log->dishes_served);
    dict_set(out, "correct_dishes", log->correct_dishes);
    dict_set(out, "wrong_dishes", log->wrong_dishes);
    dict_set(out, "ingredients_picked", log->ingredients_picked);
    dict_set(out, "pots_started", log->pots_started);
    dict_set(out, "items_dropped", log->items_dropped);
    dict_set(out, "agent_collisions", log->agent_collisions);
    dict_set(out, "n", log->n);
}

#endif // OVERCOOKED_H
