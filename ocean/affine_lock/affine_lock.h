#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef AFFINE_LOCK_NO_RENDER
#include "raylib.h"
#endif

#include "affine_lock_visible_targets.h"

#define AFFINE_LOCK_BITS 16
#define AFFINE_LOCK_TIMER_INDEX (2 * AFFINE_LOCK_BITS)
#define AFFINE_LOCK_OBS_SIZE (AFFINE_LOCK_TIMER_INDEX + 1)
// PufferLib uses one action slot for this single-discrete-action env.
#define AFFINE_LOCK_NUM_ATNS 1
#define AFFINE_LOCK_NUM_ACTIONS 8
#define AFFINE_LOCK_MAX_SCRAMBLE_DEPTH 16
#define AFFINE_LOCK_MAX_SOLUTION_DEPTH 16
#define AFFINE_LOCK_STEP_REWARD (-0.01f)
#ifndef AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH
#define AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH \
    "ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin"
#endif

typedef enum AffineLockInitializationMode {
    AFFINE_LOCK_INIT_EXACT_DISTANCE = 1,
    AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE = 2,
} AffineLockInitializationMode;

typedef enum AffineLockAction {
    AFFINE_LOCK_ACTION_SHIFT_LEFT = 0,
    AFFINE_LOCK_ACTION_SHIFT_RIGHT = 1,
    AFFINE_LOCK_ACTION_INVERT_RIGHT_7 = 2,
    AFFINE_LOCK_ACTION_SWAP_ADJACENT_BITS = 3,
    AFFINE_LOCK_ACTION_SWAP_ADJACENT_PAIRS = 4,
    AFFINE_LOCK_ACTION_SWAP_NIBBLES_EACH_BYTE = 5,
    AFFINE_LOCK_ACTION_REVERSE_EACH_NIBBLE = 6,
    AFFINE_LOCK_ACTION_REVERSE_EACH_BYTE = 7,
} AffineLockAction;

typedef struct Log {
    float perf;
    float score;
    float solve_rate;
    float scramble_depth;
    float at_max_depth;
    float max_depth_solve;
    float episode_return;
    float episode_length;
    float solve_steps;
    float timeout_rate;
    float invalid_rate;
    float start_mismatches;
    float final_mismatches;
    float one_action_target_rate;
    float two_action_target_rate;
    float short_solve_rate;
    float solve_efficiency;
    float reward_state_mismatch;
    float target_distance;
    float solved_target_distance;
    float depth_2_rate;
    float depth_2_solve_rate;
    float depth_4_rate;
    float depth_4_solve_rate;
    float depth_8_rate;
    float depth_8_solve_rate;
    float depth_16_rate;
    float depth_16_solve_rate;
    float n;
} Log;

typedef struct AffineLockShared {
    int start_depth;
    int max_depth;
    int depth_multiplier;
    int step_grace;
    int initialization_mode;
    int num_states;
    uint32_t mask;
    uint32_t* next;
    int visible_target_table_loaded;
    AffineLockVisibleTargetTable visible_target_table;
    float observation_bit_patterns[256][8];
} AffineLockShared;

typedef struct AffineLockBfsScratch {
    uint16_t* seen_generation;
    uint8_t* distances;
    uint16_t* parents;
    int8_t* parent_actions;
    uint16_t* queue;
    uint16_t generation;
    int num_states;
} AffineLockBfsScratch;

typedef struct Client {
    int screen_width;
    int screen_height;
} Client;

typedef struct AffineLock {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    uint32_t state;
    uint32_t target;
    int step_count;
    int max_steps;
    int scramble_depth;
    int curriculum_depth;
    int solution_length;
    int solution_actions[AFFINE_LOCK_MAX_SOLUTION_DEPTH];
    int known_solution;
    int target_distance;
    int start_mismatches;
    int one_action_target;
    int two_action_target;
    float episode_return;
    float last_reward;
    int last_terminal;
    int last_solved;
    int hint_visible;
    int hint_action;
    unsigned int rng;
    int env_id;
    int episode_id;
    int num_agents;
    AffineLockShared* shared;
    Client* client;
} AffineLock;

static float affine_lock_solve_credit(const AffineLockShared* shared, int depth) {
    return shared->max_depth > 0 ? (float)depth / (float)shared->max_depth : 0.0f;
}

static int affine_lock_log_depth(const AffineLock* env) {
    return env->target_distance > 0 ? env->target_distance : env->scramble_depth;
}

static const char* affine_lock_action_name(int action) {
    switch (action) {
        case AFFINE_LOCK_ACTION_SHIFT_LEFT: return "shift_left";
        case AFFINE_LOCK_ACTION_SHIFT_RIGHT: return "shift_right";
        case AFFINE_LOCK_ACTION_INVERT_RIGHT_7: return "invert_right_7";
        case AFFINE_LOCK_ACTION_SWAP_ADJACENT_BITS:
            return "swap_adjacent_bits";
        case AFFINE_LOCK_ACTION_SWAP_ADJACENT_PAIRS:
            return "swap_adjacent_pairs";
        case AFFINE_LOCK_ACTION_SWAP_NIBBLES_EACH_BYTE:
            return "swap_nibbles_each_byte";
        case AFFINE_LOCK_ACTION_REVERSE_EACH_NIBBLE:
            return "reverse_each_nibble";
        case AFFINE_LOCK_ACTION_REVERSE_EACH_BYTE: return "reverse_each_byte";
        default: return "invalid";
    }
}

static int affine_lock_count_bits(uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(value & ((1u << AFFINE_LOCK_BITS) - 1u));
#else
    int count = 0;
    for (int bit = 0; bit < AFFINE_LOCK_BITS; bit++) {
        count += (value >> bit) & 1u;
    }
    return count;
#endif
}

static void affine_lock_init_observation_bit_patterns(AffineLockShared* shared) {
    for (uint32_t value = 0; value < 256u; value++) {
        for (int bit = 0; bit < 8; bit++) {
            shared->observation_bit_patterns[value][bit] =
                (value & (1u << bit)) ? 1.0f : -1.0f;
        }
    }
}

static uint32_t affine_lock_shift_left(uint32_t state) {
    uint32_t first = state & 1u;
    return (state >> 1) | (first << (AFFINE_LOCK_BITS - 1));
}

static uint32_t affine_lock_shift_right(uint32_t state) {
    uint32_t last = (state >> (AFFINE_LOCK_BITS - 1)) & 1u;
    return ((state << 1) & ((1u << AFFINE_LOCK_BITS) - 1u)) | last;
}

static uint32_t affine_lock_swap_adjacent_bits(uint32_t state) {
    return ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
}

static uint32_t affine_lock_swap_adjacent_pairs(uint32_t state) {
    return ((state & 0x3333u) << 2) | ((state & 0xccccu) >> 2);
}

static uint32_t affine_lock_swap_nibbles_each_byte(uint32_t state) {
    return ((state & 0x0f0fu) << 4) | ((state & 0xf0f0u) >> 4);
}

static uint32_t affine_lock_reverse_each_nibble(uint32_t state) {
    return affine_lock_swap_adjacent_pairs(
        affine_lock_swap_adjacent_bits(state));
}

static uint32_t affine_lock_reverse_each_byte(uint32_t state) {
    return affine_lock_swap_nibbles_each_byte(
        affine_lock_reverse_each_nibble(state));
}

static int affine_lock_init_shared(
        AffineLockShared* shared,
        int start_depth,
        int max_depth,
        int depth_multiplier,
        int step_grace) {
    memset(shared, 0, sizeof(*shared));

    shared->start_depth = start_depth;
    shared->max_depth = max_depth;
    shared->depth_multiplier = depth_multiplier;
    shared->step_grace = step_grace;
    shared->initialization_mode = AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE;
    shared->num_states = 1 << AFFINE_LOCK_BITS;
    shared->mask = (1u << AFFINE_LOCK_BITS) - 1u;
    affine_lock_init_observation_bit_patterns(shared);

    size_t transition_count =
        (size_t)shared->num_states * AFFINE_LOCK_NUM_ACTIONS;
    shared->next = (uint32_t*)calloc(transition_count, sizeof(uint32_t));
    if (shared->next == NULL) {
        fprintf(stderr, "affine_lock: failed to allocate action table\n");
        return -1;
    }

    for (uint32_t state = 0; state < (uint32_t)shared->num_states; state++) {
        for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
            uint32_t next = state;
            switch (action) {
                case AFFINE_LOCK_ACTION_SHIFT_LEFT:
                    next = affine_lock_shift_left(state);
                    break;
                case AFFINE_LOCK_ACTION_SHIFT_RIGHT:
                    next = affine_lock_shift_right(state);
                    break;
                case AFFINE_LOCK_ACTION_INVERT_RIGHT_7:
                    next = state ^ 0xfe00u;
                    break;
                case AFFINE_LOCK_ACTION_SWAP_ADJACENT_BITS:
                    next = affine_lock_swap_adjacent_bits(state);
                    break;
                case AFFINE_LOCK_ACTION_SWAP_ADJACENT_PAIRS:
                    next = affine_lock_swap_adjacent_pairs(state);
                    break;
                case AFFINE_LOCK_ACTION_SWAP_NIBBLES_EACH_BYTE:
                    next = affine_lock_swap_nibbles_each_byte(state);
                    break;
                case AFFINE_LOCK_ACTION_REVERSE_EACH_NIBBLE:
                    next = affine_lock_reverse_each_nibble(state);
                    break;
                case AFFINE_LOCK_ACTION_REVERSE_EACH_BYTE:
                    next = affine_lock_reverse_each_byte(state);
                    break;
            }
            shared->next[state * AFFINE_LOCK_NUM_ACTIONS + action] =
                next & shared->mask;
        }
    }

    return 0;
}

static int affine_lock_prepare_visible_targets(AffineLockShared* shared) {
    if (shared->visible_target_table_loaded) {
        return 0;
    }

    char error[256];
    if (affine_lock_visible_targets_load(
            AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH,
            AFFINE_LOCK_VISIBLE_TARGET_8ACTION_V1_HASH,
            &shared->visible_target_table,
            error,
            sizeof(error)) != 0) {
        fprintf(stderr, "affine_lock: %s\n", error);
        return -1;
    }

    shared->visible_target_table_loaded = 1;
    return 0;
}

static int affine_lock_configure_initialization(
        AffineLockShared* shared,
        int initialization_mode) {
    if (initialization_mode == AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE &&
            affine_lock_prepare_visible_targets(shared) != 0) {
        return -1;
    }
    shared->initialization_mode = initialization_mode;
    return 0;
}

static void affine_lock_cleanup_thread_scratch(void);

static void affine_lock_free_shared(AffineLockShared* shared) {
    if (shared == NULL) {
        return;
    }
    free(shared->next);
    affine_lock_visible_targets_free(&shared->visible_target_table);
    affine_lock_cleanup_thread_scratch();
    memset(shared, 0, sizeof(*shared));
}

static _Thread_local AffineLockBfsScratch affine_lock_bfs_scratch = {0};

static void affine_lock_free_bfs_scratch(AffineLockBfsScratch* scratch) {
    free(scratch->seen_generation);
    free(scratch->distances);
    free(scratch->parents);
    free(scratch->parent_actions);
    free(scratch->queue);
    memset(scratch, 0, sizeof(*scratch));
}

static void affine_lock_cleanup_thread_scratch(void) {
    affine_lock_free_bfs_scratch(&affine_lock_bfs_scratch);
}

static AffineLockBfsScratch* affine_lock_get_bfs_scratch(
        const AffineLockShared* shared) {
    AffineLockBfsScratch* scratch = &affine_lock_bfs_scratch;
    if (scratch->num_states == shared->num_states) {
        return scratch;
    }

    affine_lock_free_bfs_scratch(scratch);
    scratch->num_states = shared->num_states;
    scratch->seen_generation =
        (uint16_t*)calloc((size_t)shared->num_states, sizeof(uint16_t));
    scratch->distances =
        (uint8_t*)malloc((size_t)shared->num_states * sizeof(uint8_t));
    scratch->parents =
        (uint16_t*)malloc((size_t)shared->num_states * sizeof(uint16_t));
    scratch->parent_actions =
        (int8_t*)malloc((size_t)shared->num_states * sizeof(int8_t));
    scratch->queue =
        (uint16_t*)malloc((size_t)shared->num_states * sizeof(uint16_t));

    if (scratch->seen_generation == NULL || scratch->distances == NULL ||
            scratch->parents == NULL || scratch->parent_actions == NULL ||
            scratch->queue == NULL) {
        affine_lock_free_bfs_scratch(scratch);
        return NULL;
    }

    return scratch;
}

static AffineLockBfsScratch* affine_lock_begin_bfs_scratch(
        const AffineLockShared* shared) {
    AffineLockBfsScratch* scratch = affine_lock_get_bfs_scratch(shared);
    if (scratch == NULL) {
        return NULL;
    }

    scratch->generation += 1;
    if (scratch->generation == 0) {
        memset(scratch->seen_generation, 0,
            (size_t)shared->num_states * sizeof(uint16_t));
        scratch->generation = 1;
    }
    return scratch;
}

static int affine_lock_bfs_seen(
        const AffineLockBfsScratch* scratch, uint32_t state) {
    return scratch->seen_generation[state] == scratch->generation;
}

static void affine_lock_bfs_visit(
        AffineLockBfsScratch* scratch,
        uint32_t state,
        int distance,
        uint32_t parent,
        int parent_action) {
    scratch->seen_generation[state] = scratch->generation;
    scratch->distances[state] = (uint8_t)distance;
    scratch->parents[state] = (uint16_t)parent;
    scratch->parent_actions[state] = (int8_t)parent_action;
}

static uint32_t affine_lock_apply_action(
        const AffineLockShared* shared, uint32_t rel, int action) {
    return shared->next[(rel & shared->mask) * AFFINE_LOCK_NUM_ACTIONS + action];
}

static int affine_lock_target_reachable_in_one(
        const AffineLockShared* shared, uint32_t state, uint32_t target) {
    for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
        if (affine_lock_apply_action(shared, state, action) == target) {
            return 1;
        }
    }
    return 0;
}

static int affine_lock_target_reachable_in_two(
        const AffineLockShared* shared, uint32_t state, uint32_t target) {
    if (affine_lock_target_reachable_in_one(shared, state, target)) {
        return 1;
    }
    for (int first = 0; first < AFFINE_LOCK_NUM_ACTIONS; first++) {
        uint32_t mid = affine_lock_apply_action(shared, state, first);
        for (int second = 0; second < AFFINE_LOCK_NUM_ACTIONS; second++) {
            if (affine_lock_apply_action(shared, mid, second) == target) {
                return 1;
            }
        }
    }
    return 0;
}

static int affine_lock_hint_action(
        const AffineLockShared* shared, uint32_t start, uint32_t target) {
    start &= shared->mask;
    target &= shared->mask;
    if (start == target) {
        return -1;
    }

    AffineLockBfsScratch* scratch = affine_lock_begin_bfs_scratch(shared);
    if (scratch == NULL) {
        return -1;
    }

    int head = 0;
    int tail = 0;
    affine_lock_bfs_visit(scratch, start, 0, start, -1);
    scratch->queue[tail++] = (uint16_t)start;

    while (head < tail) {
        uint32_t state = scratch->queue[head++];
        int next_distance = (int)scratch->distances[state] + 1;
        for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
            uint32_t next = affine_lock_apply_action(shared, state, action);
            if (affine_lock_bfs_seen(scratch, next)) {
                continue;
            }

            affine_lock_bfs_visit(scratch, next, next_distance, state, action);
            if (next == target) {
                uint32_t cursor = next;
                int first_action = (int)scratch->parent_actions[cursor];
                while (scratch->parents[cursor] != start) {
                    cursor = scratch->parents[cursor];
                    first_action = (int)scratch->parent_actions[cursor];
                }
                return first_action;
            }
            scratch->queue[tail++] = (uint16_t)next;
        }
    }

    return -1;
}

static inline void affine_lock_show_hint(AffineLock* env) {
    env->hint_action = affine_lock_hint_action(
        env->shared, env->state, env->target);
    env->hint_visible = 1;
}

static uint32_t affine_lock_random_u32(AffineLock* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    return env->rng;
}

// Keep RNG fully local to each env so sweep runs differ only by hyperparams.
// The mixer avoids weak low bits from the LCG when sampling bounded actions or
// bit states. Do not replace this with global rand()/srand().
static uint32_t affine_lock_random_mixed_u32(AffineLock* env) {
    uint32_t x = affine_lock_random_u32(env);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static int affine_lock_random_bounded(AffineLock* env, int bound) {
    uint32_t limit = UINT32_MAX - (UINT32_MAX % (uint32_t)bound);
    uint32_t value = affine_lock_random_mixed_u32(env);
    while (value >= limit) {
        value = affine_lock_random_mixed_u32(env);
    }
    return (int)(value % (uint32_t)bound);
}

static uint32_t affine_lock_random_state_bits(
        AffineLock* env, const AffineLockShared* shared) {
    return affine_lock_random_mixed_u32(env) & shared->mask;
}

static int affine_lock_parse_action(float raw_action, int* action_out) {
    if (!isfinite(raw_action) ||
            raw_action < 0.0f ||
            raw_action > (float)(AFFINE_LOCK_NUM_ACTIONS - 1)) {
        return 0;
    }

    int action = (int)raw_action;
    if ((float)action != raw_action) {
        return 0;
    }

    *action_out = action;
    return 1;
}

static void affine_lock_clear_generated_path(AffineLock* env) {
    env->solution_length = 0;
    for (int i = 0; i < AFFINE_LOCK_MAX_SOLUTION_DEPTH; i++) {
        env->solution_actions[i] = -1;
    }
}

static void affine_lock_store_solution_path(
        AffineLock* env,
        const uint16_t* parents,
        const int8_t* parent_actions,
        uint32_t target) {
    AffineLockShared* shared = env->shared;
    int reversed[AFFINE_LOCK_MAX_SOLUTION_DEPTH];
    int length = 0;
    uint32_t state = target & shared->mask;
    while (state != env->state) {
        if (length >= AFFINE_LOCK_MAX_SOLUTION_DEPTH ||
                parent_actions[state] < 0) {
            fprintf(stderr, "affine_lock: failed to reconstruct solution path\n");
            abort();
        }
        reversed[length++] = parent_actions[state];
        state = parents[state];
    }

    env->solution_length = length;
    for (int i = 0; i < length; i++) {
        env->solution_actions[i] = reversed[length - 1 - i];
    }
}

static const AffineLockVisibleTargetDepth* affine_lock_visible_target_depth(
        const AffineLockShared* shared,
        int requested_depth) {
    const AffineLockVisibleTargetTable* table = &shared->visible_target_table;
    for (uint32_t i = 0; i < table->depth_count; i++) {
        if (table->depths[i].depth == (uint32_t)requested_depth) {
            return &table->depths[i];
        }
    }
    return NULL;
}

static void affine_lock_store_visible_solution_path(
        AffineLock* env,
        const AffineLockVisibleTargetRecord* record) {
    int length = (int)record->solution_length;
    if (length <= 0 || length > AFFINE_LOCK_MAX_SOLUTION_DEPTH) {
        fprintf(stderr, "affine_lock: invalid visible target solution length\n");
        abort();
    }

    env->solution_length = length;
    for (int i = 0; i < length; i++) {
        int action = (int)((record->packed_actions >> (3u * i)) & 7ull);
        if (action < 0 || action >= AFFINE_LOCK_NUM_ACTIONS) {
            fprintf(stderr, "affine_lock: invalid visible target solution action\n");
            abort();
        }
        env->solution_actions[i] = action;
    }
}

static void affine_lock_generate_exact_distance_target(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    int desired_distance = env->scramble_depth;
    affine_lock_clear_generated_path(env);

    AffineLockBfsScratch* scratch = affine_lock_begin_bfs_scratch(shared);
    if (scratch == NULL) {
        fprintf(stderr, "affine_lock: failed to allocate exact-distance BFS scratch\n");
        abort();
    }

    int head = 0;
    int tail = 0;
    affine_lock_bfs_visit(scratch, env->state, 0, env->state, -1);
    scratch->queue[tail++] = (uint16_t)env->state;

    int exact_count = 0;
    uint32_t exact_target = env->state;
    int farthest_distance = 0;
    int farthest_count = 1;
    uint32_t farthest_target = env->state;

    while (head < tail) {
        uint32_t state = scratch->queue[head++];
        int distance = (int)scratch->distances[state];
        if (distance >= desired_distance) {
            continue;
        }

        for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
            uint32_t next = affine_lock_apply_action(shared, state, action);
            if (affine_lock_bfs_seen(scratch, next)) {
                continue;
            }

            int next_distance = distance + 1;
            affine_lock_bfs_visit(scratch, next, next_distance, state, action);
            scratch->queue[tail++] = (uint16_t)next;

            if (next_distance == desired_distance) {
                exact_count += 1;
                if (affine_lock_random_bounded(env, exact_count) == 0) {
                    exact_target = next;
                }
            }

            if (next_distance > farthest_distance) {
                farthest_distance = next_distance;
                farthest_count = 1;
                farthest_target = next;
            } else if (next_distance == farthest_distance) {
                farthest_count += 1;
                if (affine_lock_random_bounded(env, farthest_count) == 0) {
                    farthest_target = next;
                }
            }
        }
    }

    if (exact_count > 0) {
        env->target = exact_target & shared->mask;
        env->target_distance = desired_distance;
    } else {
        env->target = farthest_target & shared->mask;
        env->target_distance = farthest_distance;
    }
    affine_lock_store_solution_path(
        env, scratch->parents, scratch->parent_actions, env->target);
}

static void affine_lock_generate_visible_target_table_target(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    int requested_depth = env->scramble_depth;

    if (affine_lock_prepare_visible_targets(shared) != 0) {
        fprintf(stderr, "affine_lock: failed to load visible target table\n");
        abort();
    }

    const AffineLockVisibleTargetDepth* depth =
        affine_lock_visible_target_depth(shared, requested_depth);
    if (depth == NULL || depth->stored_count == 0) {
        fprintf(stderr,
            "affine_lock: visible target table has no records for depth %d\n",
            requested_depth);
        abort();
    }

    int choice = affine_lock_random_bounded(env, (int)depth->stored_count);
    uint32_t record_index = depth->first_record + (uint32_t)choice;
    if (record_index >= shared->visible_target_table.record_count) {
        fprintf(stderr, "affine_lock: invalid visible target record index\n");
        abort();
    }
    const AffineLockVisibleTargetRecord* record =
        &shared->visible_target_table.records[record_index];
    if ((int)record->depth != requested_depth ||
            record->solution_length != record->depth) {
        fprintf(stderr, "affine_lock: invalid visible target record\n");
        abort();
    }

    env->state = (uint32_t)record->start & shared->mask;
    env->target = (uint32_t)record->target & shared->mask;
    affine_lock_clear_generated_path(env);
    env->target = (uint32_t)record->target & shared->mask;
    env->target_distance = (int)record->depth;
    affine_lock_store_visible_solution_path(env, record);
}

static void affine_lock_finalize_reset(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    env->start_mismatches =
        affine_lock_count_bits((env->state ^ env->target) & shared->mask);
    if (env->target_distance >= 0) {
        env->one_action_target = env->target_distance == 1;
        env->two_action_target =
            env->target_distance == 1 || env->target_distance == 2;
    } else {
        env->one_action_target =
            affine_lock_target_reachable_in_one(shared, env->state, env->target);
        env->two_action_target =
            affine_lock_target_reachable_in_two(shared, env->state, env->target);
    }
}

static void affine_lock_reset_state(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    env->scramble_depth = env->curriculum_depth;
    env->max_steps = env->scramble_depth + shared->step_grace;
    env->step_count = 0;
    env->episode_return = 0.0f;
    env->target_distance = -1;
    env->episode_id += 1;

    env->known_solution = 1;
    if (shared->initialization_mode == AFFINE_LOCK_INIT_EXACT_DISTANCE) {
        env->state = affine_lock_random_state_bits(env, shared);
        affine_lock_generate_exact_distance_target(env);
        env->max_steps = env->target_distance + shared->step_grace;
    } else {
        affine_lock_generate_visible_target_table_target(env);
        env->max_steps = env->target_distance + shared->step_grace;
    }
    affine_lock_finalize_reset(env);
}

static void affine_lock_init_env(
        AffineLock* env, AffineLockShared* shared, unsigned int seed, int env_id) {
    env->shared = shared;
    env->rng = seed;
    env->env_id = env_id;
    env->num_agents = 1;
    env->curriculum_depth = shared->start_depth;
    env->scramble_depth = shared->start_depth;
    env->known_solution = 1;
    env->target_distance = -1;
    env->max_steps = shared->start_depth + shared->step_grace;
    env->step_count = 0;
    env->episode_return = 0.0f;
    env->last_reward = 0.0f;
    env->last_terminal = 0;
    env->last_solved = 0;
    env->hint_visible = 0;
    env->hint_action = -1;
}

static void affine_lock_add_log(
        AffineLock* env,
        int solved,
        int invalid,
        int reward_state_mismatch) {
    AffineLockShared* shared = env->shared;
    int log_depth = affine_lock_log_depth(env);
    int at_max_depth = log_depth == shared->max_depth;
    uint32_t final_diff = (env->state ^ env->target) & shared->mask;
    float solve_credit = solved ?
        affine_lock_solve_credit(shared, log_depth) : 0.0f;
    env->log.perf += solve_credit;
    env->log.score += solve_credit;
    env->log.solve_rate += solved ? 1.0f : 0.0f;
    env->log.scramble_depth += (float)env->scramble_depth;
    env->log.at_max_depth += at_max_depth ? 1.0f : 0.0f;
    env->log.max_depth_solve +=
        (solved && at_max_depth) ? 1.0f : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->step_count;
    env->log.solve_steps += solved ? (float)env->step_count : 0.0f;
    env->log.timeout_rate += (!solved && !invalid) ? 1.0f : 0.0f;
    env->log.invalid_rate += invalid ? 1.0f : 0.0f;
    env->log.start_mismatches += (float)env->start_mismatches;
    env->log.final_mismatches += (float)affine_lock_count_bits(final_diff);
    env->log.one_action_target_rate += env->one_action_target ? 1.0f : 0.0f;
    env->log.two_action_target_rate += env->two_action_target ? 1.0f : 0.0f;
    env->log.short_solve_rate += (solved && env->step_count <= 2) ? 1.0f : 0.0f;
    env->log.solve_efficiency += solved && log_depth > 0 ?
        (float)env->step_count / (float)log_depth : 0.0f;
    env->log.reward_state_mismatch += reward_state_mismatch ? 1.0f : 0.0f;
    env->log.target_distance += (float)env->target_distance;
    env->log.solved_target_distance +=
        (solved && env->target_distance >= 0) ? (float)env->target_distance : 0.0f;
    env->log.depth_2_rate += log_depth == 2 ? 1.0f : 0.0f;
    env->log.depth_2_solve_rate +=
        (solved && log_depth == 2) ? 1.0f : 0.0f;
    env->log.depth_4_rate += log_depth == 4 ? 1.0f : 0.0f;
    env->log.depth_4_solve_rate +=
        (solved && log_depth == 4) ? 1.0f : 0.0f;
    env->log.depth_8_rate += log_depth == 8 ? 1.0f : 0.0f;
    env->log.depth_8_solve_rate +=
        (solved && log_depth == 8) ? 1.0f : 0.0f;
    env->log.depth_16_rate += log_depth == 16 ? 1.0f : 0.0f;
    env->log.depth_16_solve_rate +=
        (solved && log_depth == 16) ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void affine_lock_compute_observations(AffineLock* env) {
    const float (*patterns)[8] = env->shared->observation_bit_patterns;
    uint32_t state = env->state;
    uint32_t target = env->target;
    memcpy(&env->observations[0], patterns[state & 0xffu], 8 * sizeof(float));
    memcpy(&env->observations[8], patterns[(state >> 8) & 0xffu], 8 * sizeof(float));
    memcpy(&env->observations[16], patterns[target & 0xffu], 8 * sizeof(float));
    memcpy(&env->observations[24], patterns[(target >> 8) & 0xffu], 8 * sizeof(float));
    env->observations[AFFINE_LOCK_TIMER_INDEX] = env->max_steps > 0 ?
        (float)env->step_count / (float)env->max_steps : 0.0f;
}

static void compute_observations(AffineLock* env) {
    affine_lock_compute_observations(env);
}

static void c_reset(AffineLock* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->last_reward = 0.0f;
    env->last_terminal = 0;
    env->last_solved = 0;
    env->hint_visible = 0;
    env->hint_action = -1;
    affine_lock_reset_state(env);
    compute_observations(env);
}

static void affine_lock_advance_curriculum(AffineLock* env, int solved) {
    AffineLockShared* shared = env->shared;
    if (!solved) {
        env->curriculum_depth = shared->start_depth;
        return;
    }

    int next_depth = env->scramble_depth * shared->depth_multiplier;
    if (next_depth < env->scramble_depth) {
        next_depth = shared->max_depth;
    }
    if (next_depth > shared->max_depth) {
        next_depth = shared->max_depth;
    }
    env->curriculum_depth = next_depth;
}

static void affine_lock_finish_episode(
        AffineLock* env,
        int solved,
        int invalid,
        int reward_state_mismatch) {
    affine_lock_add_log(env, solved, invalid, reward_state_mismatch);
    affine_lock_advance_curriculum(env, solved);
    affine_lock_reset_state(env);
}

static void c_step(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    int action = -1;
    int valid_action = affine_lock_parse_action(env->actions[0], &action);
    float reward = AFFINE_LOCK_STEP_REWARD;
    int terminal = 0;
    int solved = 0;
    int invalid = 0;
    int reward_state_mismatch = 0;

    env->terminals[0] = 0.0f;
    env->hint_visible = 0;
    env->hint_action = -1;
    env->step_count += 1;

    if (!valid_action) {
        reward = -1.0f;
        terminal = 1;
        invalid = 1;
    } else {
        env->state = affine_lock_apply_action(shared, env->state, action);
        if (env->state == env->target) {
            reward = 1.0f;
            terminal = 1;
            solved = 1;
        } else if (env->step_count >= env->max_steps) {
            reward = -1.0f;
            terminal = 1;
        }
    }
    reward_state_mismatch = (reward == 1.0f && env->state != env->target);

    env->rewards[0] = reward;
    env->episode_return += reward;
    env->last_reward = reward;

    if (terminal) {
        env->terminals[0] = 1.0f;
        env->last_terminal = 1;
        env->last_solved = solved;
        affine_lock_finish_episode(env, solved, invalid, reward_state_mismatch);
    }

    compute_observations(env);
}

static void c_close(AffineLock* env) {
    if (env->client == NULL) {
        return;
    }
#ifndef AFFINE_LOCK_NO_RENDER
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
    free(env->client);
    env->client = NULL;
}

#ifndef AFFINE_LOCK_NO_RENDER
static Client* affine_lock_make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->screen_width = 780;
    client->screen_height = 360;
    InitWindow(client->screen_width, client->screen_height, "PufferLib AffineLock");
    SetTargetFPS(30);
    return client;
}

static Color affine_lock_bit_fill(int on) {
    return on ? (Color){80, 210, 140, 255} : (Color){38, 48, 58, 255};
}

static void affine_lock_draw_bit_row(
        AffineLock* env, const char* label, uint32_t value, int y) {
    DrawText(label, 30, y + 9, 20, RAYWHITE);
    for (int bit = 0; bit < AFFINE_LOCK_BITS; bit++) {
        int x = 145 + bit * 34;
        int on = (value >> bit) & 1u;
        int mismatch = ((env->state ^ env->target) >> bit) & 1u;
        Color fill = affine_lock_bit_fill(on);
        Color border = mismatch ?
            (Color){238, 88, 88, 255} : (Color){182, 196, 205, 255};
        DrawRectangle(x, y, 24, 34, fill);
        DrawRectangleLinesEx((Rectangle){(float)x, (float)y, 24.0f, 34.0f},
            mismatch ? 3.0f : 1.0f, border);
        DrawText(TextFormat("%d", bit), x + 5, y + 40, 10,
            (Color){128, 140, 150, 255});
    }
}

static void c_render(AffineLock* env) {
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        c_close(env);
        exit(0);
    }

    if (env->client == NULL) {
        env->client = affine_lock_make_client();
    }

    uint32_t rel = (env->state ^ env->target) & env->shared->mask;
    const char* status = "running";
    Color status_color = (Color){190, 198, 206, 255};
    if (env->last_terminal) {
        status = env->last_solved ? "solved" : "failed";
        status_color = env->last_solved ?
            (Color){80, 210, 140, 255} : (Color){238, 88, 88, 255};
    }

    BeginDrawing();
    ClearBackground((Color){12, 15, 18, 255});
    DrawText("Affine Lock", 30, 24, 28, RAYWHITE);
    DrawText(TextFormat("depth %d/%d  step %d/%d  last reward %.2f",
        env->scramble_depth, env->shared->max_depth,
        env->step_count, env->max_steps, env->last_reward),
        30, 62, 20, (Color){180, 190, 200, 255});
    DrawText(TextFormat("status %s  mismatches 0x%04x",
        status, rel), 30, 90, 20, status_color);

    affine_lock_draw_bit_row(env, "current", env->state, 138);
    affine_lock_draw_bit_row(env, "target", env->target, 220);

    if (env->hint_visible) {
        const char* hint = env->hint_action >= 0 ?
            TextFormat("Hint: press %d (%s)",
                env->hint_action + 1,
                affine_lock_action_name(env->hint_action)) :
            "Hint: already solved";
        int hint_width = MeasureText(hint, 18);
        DrawText(hint, env->client->screen_width - hint_width - 30,
            274, 18, (Color){245, 205, 92, 255});
    }

    DrawText("1 shiftL  2 shiftR  3 inv7  4 bit-swap  5 pair-swap",
        30, 300, 16, (Color){160, 170, 178, 255});
    DrawText("6 nib-swap  7 rev-nib  8 rev-byte  R reset  H = Hint",
        30, 322, 16, (Color){160, 170, 178, 255});
    EndDrawing();
}
#else
static void c_render(AffineLock* env) {
    (void)env;
}
#endif
