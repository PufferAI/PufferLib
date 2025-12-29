/* Overcooked Items: Item and cooking pot management functions.
 */

#ifndef OVERCOOKED_ITEMS_H
#define OVERCOOKED_ITEMS_H

#include "overcooked_types.h"

static inline Item* get_item_at(Overcooked* env, int x, int y) {
    int idx = env->item_grid[y * env->width + x];
    return (idx >= 0) ? &env->items[idx] : NULL;
}

static void add_item(Overcooked* env, int type, int x, int y) {
    if (env->num_items < env->max_items) {
        int idx = env->num_items;
        env->items[idx].type = type;
        env->items[idx].x = x;
        env->items[idx].y = y;
        env->items[idx].state = 0;
        env->items[idx].num_onions = 0;
        env->items[idx].num_tomatoes = 0;
        env->items[idx].total_ingredients = 0;
        env->item_grid[y * env->width + x] = idx;
        env->num_items++;
    }
}

static void remove_item(Overcooked* env, int x, int y) {
    int idx = env->item_grid[y * env->width + x];
    if (idx < 0) return;

    env->item_grid[y * env->width + x] = -1;

    if (idx < env->num_items - 1) {
        Item* last = &env->items[env->num_items - 1];
        env->items[idx] = *last;
        env->item_grid[last->y * env->width + last->x] = idx;
    }
    env->num_items--;
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

static void init_pot_indices(Overcooked* env) {
    // Allocate pot index grid (same size as main grid)
    env->pot_index_grid = calloc(env->width * env->height, sizeof(int));

    // Initialize all cells to -1 (not a stove)
    for (int i = 0; i < env->width * env->height; i++) {
        env->pot_index_grid[i] = -1;
    }

    // Map stove cells to their pot indices (same order as init_cooking_pots)
    int pot_idx = 0;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (env->grid[y * env->width + x] == STOVE) {
                env->pot_index_grid[y * env->width + x] = pot_idx++;
            }
        }
    }
}

// O(1) pot lookup using precomputed index grid
static inline CookingPot* get_pot_at(Overcooked* env, int x, int y) {
    int idx = env->pot_index_grid[y * env->width + x];
    return (idx >= 0) ? &env->cooking_pots[idx] : NULL;
}

static void init_item_grid(Overcooked* env) {
    env->item_grid = calloc(env->width * env->height, sizeof(int));
    for (int i = 0; i < env->width * env->height; i++) {
        env->item_grid[i] = -1;
    }
}

static void reset_item_grid(Overcooked* env) {
    for (int i = 0; i < env->width * env->height; i++) {
        env->item_grid[i] = -1;
    }
}

static void update_cooking(Overcooked* env) {
    for (int i = 0; i < env->num_stoves; i++) {
        CookingPot* pot = &env->cooking_pots[i];
        if (pot->cooking_state == COOKING) {
            pot->cooking_progress++;
            if (pot->cooking_progress >= COOKING_TIME) {
                pot->cooking_state = COOKED;
            }
        }
    }
}

#endif // OVERCOOKED_ITEMS_H
