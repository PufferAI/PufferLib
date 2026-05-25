#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef AFFINE_LOCK_NO_RENDER
#include "raylib.h"
#endif

#include "affine_lock_visible_targets.h"
#ifdef AFFINE_LOCK_ENABLE_TRANSFORM_TABLE_HELPERS
#include "generated/affine_lock_transform_table.h"
#endif

#define AFFINE_LOCK_BITS 16
#define AFFINE_LOCK_TIMER_INDEX (2 * AFFINE_LOCK_BITS)
#define AFFINE_LOCK_OBS_SIZE (AFFINE_LOCK_TIMER_INDEX + 1)
// PufferLib uses one action slot for this single-discrete-action env.
#define AFFINE_LOCK_NUM_ATNS 1
#define AFFINE_LOCK_NUM_ACTIONS 8
#define AFFINE_LOCK_MAX_SCRAMBLE_DEPTH 16
#define AFFINE_LOCK_MAX_SOLUTION_DEPTH 64
#define AFFINE_LOCK_STEP_REWARD (-0.01f)
#define AFFINE_LOCK_DEFAULT_DEBUG_LOG_DIR "logs/affine_lock"
#ifndef AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH
#define AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH \
    "ocean/affine_lock/generated/affine_lock_odd7_visible_targets.bin"
#endif

typedef enum AffineLockInitializationMode {
    AFFINE_LOCK_INIT_SCRAMBLE = 0,
    AFFINE_LOCK_INIT_RANDOM = 1,
    AFFINE_LOCK_INIT_EXACT_DISTANCE = 2,
    AFFINE_LOCK_INIT_WCA_RANDOM_STATE = 3,
    AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE = 4,
    AFFINE_LOCK_INIT_PRECOMPUTED_TRANSFORM =
        AFFINE_LOCK_INIT_VISIBLE_TARGET_TABLE,
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
    int bits;
    int start_depth;
    int max_depth;
    int depth_multiplier;
    int step_grace;
    int initialization_mode;
    int num_states;
    uint32_t mask;
    int inverse_actions[AFFINE_LOCK_NUM_ACTIONS];
    int debug_log_level;
    int debug_log_env_id;
    int debug_log_max_episodes;
    int debug_log_min_depth;
    char debug_log_dir[256];
    uint32_t* next;
#ifdef AFFINE_LOCK_ENABLE_TRANSFORM_TABLE_HELPERS
    uint16_t* precomputed_permuted_states;
#endif
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
    int scramble_length;
    int scramble_actions[AFFINE_LOCK_MAX_SCRAMBLE_DEPTH];
    uint32_t scramble_states[AFFINE_LOCK_MAX_SCRAMBLE_DEPTH + 1];
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
    int trace_this_episode;
    int debug_traced_episodes;
    FILE* debug_log_file;
    char debug_log_path[256];
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

static const char* affine_lock_initialization_mode_name(int mode) {
    switch (mode) {
        case AFFINE_LOCK_INIT_SCRAMBLE: return "scramble";
        case AFFINE_LOCK_INIT_RANDOM: return "random";
        case AFFINE_LOCK_INIT_EXACT_DISTANCE: return "exact_distance";
        case AFFINE_LOCK_INIT_WCA_RANDOM_STATE: return "wca_random_state";
        case AFFINE_LOCK_INIT_PRECOMPUTED_TRANSFORM:
            return "visible_target_table";
        default: return "unknown";
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

static void affine_lock_write_bits(FILE* file, uint32_t value) {
    fputc('"', file);
    for (int bit = 0; bit < AFFINE_LOCK_BITS; bit++) {
        fputc((value & (1u << bit)) ? '1' : '0', file);
    }
    fputc('"', file);
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

#ifdef AFFINE_LOCK_ENABLE_TRANSFORM_TABLE_HELPERS
static uint32_t affine_lock_apply_precomputed_perm(int perm_id, uint32_t state) {
    uint32_t out = 0u;
    const uint8_t* perm = AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[perm_id];
    for (int out_bit = 0; out_bit < AFFINE_LOCK_BITS; out_bit++) {
        if ((state & (1u << perm[out_bit])) != 0u) {
            out |= 1u << out_bit;
        }
    }
    return out;
}

static int affine_lock_prepare_precomputed_transforms(AffineLockShared* shared) {
    if (shared->precomputed_permuted_states != NULL) {
        return 0;
    }
    if (AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT != 16384 ||
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT != 32 ||
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE < 1 ||
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSET_COUNT !=
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE + 2 ||
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[0] != 0 ||
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[
                AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE + 1] !=
                    AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT) {
        fprintf(stderr, "affine_lock: invalid precomputed transform table\n");
        return -1;
    }

    size_t table_count =
        (size_t)AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT *
        (size_t)shared->num_states;
    uint16_t* table = (uint16_t*)malloc(table_count * sizeof(uint16_t));
    if (table == NULL) {
        fprintf(stderr,
            "affine_lock: failed to allocate precomputed transform cache\n");
        return -1;
    }

    for (int perm_id = 0;
            perm_id < AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT;
            perm_id++) {
        uint16_t* perm_table =
            table + (size_t)perm_id * (size_t)shared->num_states;
        for (uint32_t state = 0; state < (uint32_t)shared->num_states; state++) {
            perm_table[state] =
                (uint16_t)affine_lock_apply_precomputed_perm(perm_id, state);
        }
    }

    shared->precomputed_permuted_states = table;
    return 0;
}

static uint32_t affine_lock_apply_precomputed_transform(
        const AffineLockShared* shared,
        uint32_t state,
        const AffineLockPrecomputedTransform* transform) {
    uint32_t permuted;
    uint32_t masked_state = state & shared->mask;
    if (shared->precomputed_permuted_states != NULL) {
        const uint16_t* perm_table =
            shared->precomputed_permuted_states +
            (size_t)transform->perm_id * (size_t)shared->num_states;
        permuted = perm_table[masked_state];
    } else {
        permuted =
            affine_lock_apply_precomputed_perm(transform->perm_id, masked_state);
    }
    return (permuted ^ transform->xor_mask) & shared->mask;
}
#endif

static int affine_lock_init_shared(
        AffineLockShared* shared,
        int bits,
        int start_depth,
        int max_depth,
        int depth_multiplier,
        int step_grace) {
    memset(shared, 0, sizeof(*shared));

    if (bits != AFFINE_LOCK_BITS) {
        fprintf(stderr,
            "affine_lock: bits must be %d; got %d\n",
            AFFINE_LOCK_BITS, bits);
        return -1;
    }
    if (start_depth < 1 || start_depth > AFFINE_LOCK_MAX_SCRAMBLE_DEPTH) {
        fprintf(stderr,
            "affine_lock: start_depth must be in [1, %d]; got %d\n",
            AFFINE_LOCK_MAX_SCRAMBLE_DEPTH, start_depth);
        return -1;
    }
    if (max_depth < start_depth || max_depth > AFFINE_LOCK_MAX_SCRAMBLE_DEPTH) {
        fprintf(stderr,
            "affine_lock: max_depth must be in [%d, %d]; got %d\n",
            start_depth, AFFINE_LOCK_MAX_SCRAMBLE_DEPTH, max_depth);
        return -1;
    }
    if (depth_multiplier < 1) {
        fprintf(stderr,
            "affine_lock: depth_multiplier must be >= 1; got %d\n",
            depth_multiplier);
        return -1;
    }
    if (step_grace < 0) {
        fprintf(stderr,
            "affine_lock: step_grace must be >= 0; got %d\n",
            step_grace);
        return -1;
    }

    shared->bits = bits;
    shared->start_depth = start_depth;
    shared->max_depth = max_depth;
    shared->depth_multiplier = depth_multiplier;
    shared->step_grace = step_grace;
    shared->initialization_mode = AFFINE_LOCK_INIT_SCRAMBLE;
    shared->num_states = 1 << AFFINE_LOCK_BITS;
    shared->mask = (1u << AFFINE_LOCK_BITS) - 1u;
    shared->debug_log_level = 0;
    shared->debug_log_env_id = 0;
    shared->debug_log_max_episodes = 0;
    shared->debug_log_min_depth = 0;
    snprintf(shared->debug_log_dir, sizeof(shared->debug_log_dir),
        "%s", AFFINE_LOCK_DEFAULT_DEBUG_LOG_DIR);
    affine_lock_init_observation_bit_patterns(shared);

    size_t transition_count =
        (size_t)shared->num_states * AFFINE_LOCK_NUM_ACTIONS;
    shared->next = (uint32_t*)calloc(transition_count, sizeof(uint32_t));
    if (shared->next == NULL) {
        fprintf(stderr, "affine_lock: failed to allocate action table\n");
        return -1;
    }

    shared->inverse_actions[AFFINE_LOCK_ACTION_SHIFT_LEFT] =
        AFFINE_LOCK_ACTION_SHIFT_RIGHT;
    shared->inverse_actions[AFFINE_LOCK_ACTION_SHIFT_RIGHT] =
        AFFINE_LOCK_ACTION_SHIFT_LEFT;
    shared->inverse_actions[AFFINE_LOCK_ACTION_INVERT_RIGHT_7] =
        AFFINE_LOCK_ACTION_INVERT_RIGHT_7;
    shared->inverse_actions[AFFINE_LOCK_ACTION_SWAP_ADJACENT_BITS] =
        AFFINE_LOCK_ACTION_SWAP_ADJACENT_BITS;
    shared->inverse_actions[AFFINE_LOCK_ACTION_SWAP_ADJACENT_PAIRS] =
        AFFINE_LOCK_ACTION_SWAP_ADJACENT_PAIRS;
    shared->inverse_actions[AFFINE_LOCK_ACTION_SWAP_NIBBLES_EACH_BYTE] =
        AFFINE_LOCK_ACTION_SWAP_NIBBLES_EACH_BYTE;
    shared->inverse_actions[AFFINE_LOCK_ACTION_REVERSE_EACH_NIBBLE] =
        AFFINE_LOCK_ACTION_REVERSE_EACH_NIBBLE;
    shared->inverse_actions[AFFINE_LOCK_ACTION_REVERSE_EACH_BYTE] =
        AFFINE_LOCK_ACTION_REVERSE_EACH_BYTE;

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
            AFFINE_LOCK_VISIBLE_TARGET_ODD7_ACTION_SET_HASH,
            &shared->visible_target_table,
            error,
            sizeof(error)) != 0) {
        fprintf(stderr, "affine_lock: %s\n", error);
        return -1;
    }

    if (shared->visible_target_table.bits != AFFINE_LOCK_BITS ||
            shared->visible_target_table.num_actions != AFFINE_LOCK_NUM_ACTIONS) {
        fprintf(stderr, "affine_lock: visible target table shape mismatch\n");
        affine_lock_visible_targets_free(&shared->visible_target_table);
        return -1;
    }

    shared->visible_target_table_loaded = 1;
    return 0;
}

static int affine_lock_configure_initialization(
        AffineLockShared* shared,
        int initialization_mode) {
    if (initialization_mode != AFFINE_LOCK_INIT_SCRAMBLE &&
            initialization_mode != AFFINE_LOCK_INIT_RANDOM &&
            initialization_mode != AFFINE_LOCK_INIT_EXACT_DISTANCE &&
            initialization_mode != AFFINE_LOCK_INIT_WCA_RANDOM_STATE &&
            initialization_mode != AFFINE_LOCK_INIT_PRECOMPUTED_TRANSFORM) {
        fprintf(stderr,
            "affine_lock: initialization_mode must be 0 (scramble), 1 (random), 2 (exact_distance), 3 (wca_random_state), or 4 (visible_target_table); got %d\n",
            initialization_mode);
        return -1;
    }
    if (initialization_mode == AFFINE_LOCK_INIT_PRECOMPUTED_TRANSFORM &&
            affine_lock_prepare_visible_targets(shared) != 0) {
        return -1;
    }
    shared->initialization_mode = initialization_mode;
    return 0;
}

static void affine_lock_configure_debug(
        AffineLockShared* shared,
        int debug_log_level,
        int debug_log_env_id,
        int debug_log_max_episodes,
        int debug_log_min_depth) {
    shared->debug_log_level = debug_log_level;
    shared->debug_log_env_id = debug_log_env_id;
    shared->debug_log_max_episodes = debug_log_max_episodes;
    shared->debug_log_min_depth = debug_log_min_depth;
}

static void affine_lock_cleanup_thread_scratch(void);

#ifdef AFFINE_LOCK_TEST_HOOKS
static void affine_lock_configure_debug_dir(
        AffineLockShared* shared,
        const char* debug_log_dir) {
    if (debug_log_dir == NULL || debug_log_dir[0] == '\0') {
        debug_log_dir = AFFINE_LOCK_DEFAULT_DEBUG_LOG_DIR;
    }
    snprintf(shared->debug_log_dir, sizeof(shared->debug_log_dir),
        "%s", debug_log_dir);
}
#endif

static void affine_lock_free_shared(AffineLockShared* shared) {
    if (shared == NULL) {
        return;
    }
    free(shared->next);
#ifdef AFFINE_LOCK_ENABLE_TRANSFORM_TABLE_HELPERS
    free(shared->precomputed_permuted_states);
#endif
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

static int affine_lock_shortest_distance(
        const AffineLockShared* shared, uint32_t start, uint32_t target) {
    start &= shared->mask;
    target &= shared->mask;
    if (start == target) {
        return 0;
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
            if (next == target) {
                return next_distance;
            }
            affine_lock_bfs_visit(scratch, next, next_distance, state, action);
            scratch->queue[tail++] = (uint16_t)next;
        }
    }

    return -1;
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

static int affine_lock_open_debug_log(AffineLock* env) {
    if (env->debug_log_file != NULL) {
        return 1;
    }

    if (strcmp(env->shared->debug_log_dir, AFFINE_LOCK_DEFAULT_DEBUG_LOG_DIR) == 0) {
        mkdir("logs", 0777);
    }
    mkdir(env->shared->debug_log_dir, 0777);
    snprintf(env->debug_log_path, sizeof(env->debug_log_path),
        "%s/debug_trace_%ld_env%d.jsonl",
        env->shared->debug_log_dir, (long)getpid(), env->env_id);
    env->debug_log_file = fopen(env->debug_log_path, "a");
    return env->debug_log_file != NULL;
}

static void affine_lock_write_action_id_array(
        FILE* file, const int* actions, int length) {
    fputc('[', file);
    for (int i = 0; i < length; i++) {
        fprintf(file, "%s%d", i == 0 ? "" : ",", actions[i]);
    }
    fputc(']', file);
}

static void affine_lock_write_action_name_array(
        FILE* file, const int* actions, int length) {
    fputc('[', file);
    for (int i = 0; i < length; i++) {
        fprintf(file, "%s\"%s\"", i == 0 ? "" : ",",
            affine_lock_action_name(actions[i]));
    }
    fputc(']', file);
}

static void affine_lock_trace_scramble(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    if (!env->trace_this_episode || shared->debug_log_level < 1) {
        return;
    }
    FILE* file = env->debug_log_file;
    int exact_distance =
        affine_lock_shortest_distance(shared, env->state, env->target);
    fprintf(file,
        "{\"type\":\"reset\",\"env_id\":%d,\"episode\":%d,"
        "\"initialization_mode\":\"%s\",\"known_solution\":%s,"
        "\"depth\":%d,\"max_steps\":%d,\"start\":",
        env->env_id, env->episode_id,
        affine_lock_initialization_mode_name(shared->initialization_mode),
        env->known_solution ? "true" : "false",
        env->scramble_depth, env->max_steps);
    affine_lock_write_bits(file, env->state);
    fprintf(file, ",\"target\":");
    affine_lock_write_bits(file, env->target);
    fprintf(file,
        ",\"start_mismatches\":%d,\"min_win_moves\":%d,"
        "\"one_action_target\":%s,"
        "\"two_action_target\":%s,\"reachable\":%s,\"exact_distance\":%d,"
        "\"exact_shortest_distance\":%d,\"solution_length\":%d,"
        "\"solution_action_ids\":",
        env->start_mismatches,
        env->target_distance,
        env->one_action_target ? "true" : "false",
        env->two_action_target ? "true" : "false",
        exact_distance >= 0 ? "true" : "false",
        exact_distance,
        exact_distance,
        env->solution_length);
    affine_lock_write_action_id_array(file, env->solution_actions, env->solution_length);
    fprintf(file, ",\"solution_actions\":");
    affine_lock_write_action_name_array(file, env->solution_actions, env->solution_length);
    fprintf(file, ",\"scramble_actions\":");
    affine_lock_write_action_name_array(file, env->scramble_actions, env->scramble_length);
    fprintf(file, "}\n");

    if (shared->debug_log_level >= 2) {
        for (int i = 0; i < env->scramble_length; i++) {
            fprintf(file,
                "{\"type\":\"scramble_step\",\"env_id\":%d,\"episode\":%d,"
                "\"step\":%d,\"action\":\"%s\",\"before\":",
                env->env_id, env->episode_id, i + 1,
                affine_lock_action_name(env->scramble_actions[i]));
            affine_lock_write_bits(file, env->scramble_states[i]);
            fprintf(file, ",\"after\":");
            affine_lock_write_bits(file, env->scramble_states[i + 1]);
            fprintf(file, ",\"target\":");
            affine_lock_write_bits(file, env->target);
            fprintf(file, "}\n");
        }
    }
    fflush(file);
}

static void affine_lock_trace_policy_step(
        AffineLock* env,
        int step_before,
        int action,
        uint32_t before,
        uint32_t after,
        float reward,
        int terminal,
        int solved,
        int invalid,
        int reward_state_mismatch) {
    AffineLockShared* shared = env->shared;
    if (!env->trace_this_episode || shared->debug_log_level < 2) {
        return;
    }
    FILE* file = env->debug_log_file;
    fprintf(file,
        "{\"type\":\"policy_step\",\"env_id\":%d,\"episode\":%d,"
        "\"step\":%d,\"action\":%d,\"action_name\":\"%s\","
        "\"before\":",
        env->env_id, env->episode_id, step_before + 1,
        action, affine_lock_action_name(action));
    affine_lock_write_bits(file, before);
    fprintf(file, ",\"after\":");
    affine_lock_write_bits(file, after);
    fprintf(file, ",\"target\":");
    affine_lock_write_bits(file, env->target);
    fprintf(file,
        ",\"mismatches\":%d,\"timer_before\":%.6f,\"timer_after\":%.6f,"
        "\"reward\":%.6f,\"terminal\":%s,\"solved\":%s,\"invalid\":%s,"
        "\"state_equals_target\":%s,\"reward_state_mismatch\":%s}\n",
        affine_lock_count_bits((after ^ env->target) & shared->mask),
        env->max_steps > 0 ? (float)step_before / (float)env->max_steps : 0.0f,
        env->max_steps > 0 ? (float)env->step_count / (float)env->max_steps : 0.0f,
        reward,
        terminal ? "true" : "false",
        solved ? "true" : "false",
        invalid ? "true" : "false",
        after == env->target ? "true" : "false",
        reward_state_mismatch ? "true" : "false");
    fflush(file);
}

static void affine_lock_trace_episode_end(
        AffineLock* env,
        int solved,
        int invalid,
        int reward_state_mismatch) {
    AffineLockShared* shared = env->shared;
    if (!env->trace_this_episode || shared->debug_log_level < 1) {
        return;
    }
    FILE* file = env->debug_log_file;
    fprintf(file,
        "{\"type\":\"episode_end\",\"env_id\":%d,\"episode\":%d,"
        "\"solved\":%s,\"invalid\":%s,\"steps\":%d,\"depth\":%d,"
        "\"episode_return\":%.6f,\"state\":",
        env->env_id, env->episode_id,
        solved ? "true" : "false",
        invalid ? "true" : "false",
        env->step_count, env->scramble_depth, env->episode_return);
    affine_lock_write_bits(file, env->state);
    fprintf(file, ",\"target\":");
    affine_lock_write_bits(file, env->target);
    fprintf(file,
        ",\"final_mismatches\":%d,\"state_equals_target\":%s,"
        "\"reward_state_mismatch\":%s}\n",
        affine_lock_count_bits((env->state ^ env->target) & shared->mask),
        env->state == env->target ? "true" : "false",
        reward_state_mismatch ? "true" : "false");
    fflush(file);
}

static int affine_lock_pick_scramble_action(
        AffineLock* env,
        uint32_t state,
        int prev_action) {
    AffineLockShared* shared = env->shared;
    for (int attempt = 0; attempt < 256; attempt++) {
        int action = affine_lock_random_bounded(env, AFFINE_LOCK_NUM_ACTIONS);
        if (prev_action >= 0 && action == shared->inverse_actions[prev_action]) {
            continue;
        }
        uint32_t next = affine_lock_apply_action(shared, state, action);
        if (next != state) {
            return action;
        }
    }

    for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
        if (prev_action >= 0 && action == shared->inverse_actions[prev_action]) {
            continue;
        }
        uint32_t next = affine_lock_apply_action(shared, state, action);
        if (next != state) {
            return action;
        }
    }

    return -1;
}

static void affine_lock_generate_scramble(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    uint32_t target = env->state;
    int prev_action = -1;
    env->scramble_states[0] = target;

    env->scramble_length = env->scramble_depth;
    env->solution_length = env->scramble_depth;
    for (int i = 0; i < AFFINE_LOCK_MAX_SCRAMBLE_DEPTH; i++) {
        env->scramble_actions[i] = -1;
    }
    for (int i = 0; i < AFFINE_LOCK_MAX_SOLUTION_DEPTH; i++) {
        env->solution_actions[i] = -1;
    }

    for (int i = 0; i < env->scramble_length; i++) {
        int action = affine_lock_pick_scramble_action(env, target, prev_action);
        if (action < 0) {
            fprintf(stderr, "affine_lock: failed to generate scramble action\n");
            abort();
        }

        uint32_t next = affine_lock_apply_action(shared, target, action);
        env->scramble_actions[i] = action;
        env->solution_actions[i] = action;
        target = next;
        env->scramble_states[i + 1] = target;
        prev_action = action;
    }

    env->target = target & shared->mask;
}

static void affine_lock_clear_generated_path(AffineLock* env) {
    env->scramble_length = 0;
    env->solution_length = 0;
    env->scramble_states[0] = env->state;
    for (int i = 0; i < AFFINE_LOCK_MAX_SCRAMBLE_DEPTH; i++) {
        env->scramble_actions[i] = -1;
        env->scramble_states[i + 1] = env->state;
    }
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

static void affine_lock_generate_random_target(AffineLock* env) {
    AffineLockShared* shared = env->shared;

    affine_lock_clear_generated_path(env);
    env->target_distance = -1;

    for (int attempt = 0; attempt < 64; attempt++) {
        env->target = affine_lock_random_state_bits(env, shared);
        if (env->target != env->state) {
            return;
        }
    }

    env->target = (env->state ^ 1u) & shared->mask;
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

static void affine_lock_generate_wca_random_state_target(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    int desired_distance = env->scramble_depth;
    affine_lock_clear_generated_path(env);

    AffineLockBfsScratch* scratch = affine_lock_begin_bfs_scratch(shared);
    if (scratch == NULL) {
        fprintf(stderr, "affine_lock: failed to allocate random-state BFS scratch\n");
        abort();
    }

    int head = 0;
    int tail = 0;
    affine_lock_bfs_visit(scratch, env->state, 0, env->state, -1);
    scratch->queue[tail++] = (uint16_t)env->state;

    int target_count = 0;
    uint32_t sampled_target = env->state;
    int farthest_distance = 0;
    int farthest_count = 1;
    uint32_t farthest_target = env->state;

    while (head < tail) {
        uint32_t state = scratch->queue[head++];
        int distance = (int)scratch->distances[state];
        if (target_count > 0 && distance >= desired_distance) {
            // All states at the requested shell were discovered by expanding
            // the previous BFS level. Deeper states are only needed for fallback.
            break;
        }

        int next_distance = distance + 1;
        for (int action = 0; action < AFFINE_LOCK_NUM_ACTIONS; action++) {
            uint32_t next = affine_lock_apply_action(shared, state, action);
            if (affine_lock_bfs_seen(scratch, next)) {
                continue;
            }

            affine_lock_bfs_visit(scratch, next, next_distance, state, action);
            scratch->queue[tail++] = (uint16_t)next;

            if (next_distance == desired_distance) {
                target_count += 1;
                if (affine_lock_random_bounded(env, target_count) == 0) {
                    sampled_target = next;
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

    if (target_count > 0) {
        env->target = sampled_target & shared->mask;
        env->target_distance = scratch->distances[sampled_target];
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
    env->trace_this_episode = 0;
    if (shared->debug_log_level > 0 &&
            env->env_id == shared->debug_log_env_id &&
            env->debug_traced_episodes < shared->debug_log_max_episodes &&
            env->scramble_depth >= shared->debug_log_min_depth &&
            affine_lock_open_debug_log(env)) {
        env->trace_this_episode = 1;
        env->debug_traced_episodes += 1;
        affine_lock_trace_scramble(env);
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

    if (shared->initialization_mode == AFFINE_LOCK_INIT_RANDOM) {
        env->state = affine_lock_random_state_bits(env, shared);
        env->known_solution = 0;
        affine_lock_generate_random_target(env);
    } else if (shared->initialization_mode == AFFINE_LOCK_INIT_EXACT_DISTANCE) {
        env->state = affine_lock_random_state_bits(env, shared);
        env->known_solution = 1;
        affine_lock_generate_exact_distance_target(env);
        env->max_steps = env->target_distance + shared->step_grace;
    } else if (shared->initialization_mode == AFFINE_LOCK_INIT_WCA_RANDOM_STATE) {
        env->state = affine_lock_random_state_bits(env, shared);
        env->known_solution = 1;
        affine_lock_generate_wca_random_state_target(env);
        env->max_steps = env->target_distance + shared->step_grace;
    } else if (shared->initialization_mode ==
            AFFINE_LOCK_INIT_PRECOMPUTED_TRANSFORM) {
        env->known_solution = 1;
        affine_lock_generate_visible_target_table_target(env);
        env->max_steps = env->target_distance + shared->step_grace;
    } else {
        env->known_solution = 1;
        for (int attempt = 0; attempt < 32; attempt++) {
            env->state = affine_lock_random_state_bits(env, shared);
            affine_lock_generate_scramble(env);
            if (env->state != env->target) {
                break;
            }
        }
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
    env->debug_log_file = NULL;
    env->debug_log_path[0] = '\0';
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
    affine_lock_trace_episode_end(env, solved, invalid, reward_state_mismatch);
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
    int step_before = env->step_count;
    uint32_t state_before = env->state;

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
    affine_lock_trace_policy_step(
        env, step_before, action, state_before, env->state,
        reward, terminal, solved, invalid, reward_state_mismatch);

    if (terminal) {
        env->terminals[0] = 1.0f;
        env->last_terminal = 1;
        env->last_solved = solved;
        affine_lock_finish_episode(env, solved, invalid, reward_state_mismatch);
    }

    compute_observations(env);
}

static void c_close(AffineLock* env) {
    if (env->debug_log_file != NULL) {
        fclose(env->debug_log_file);
        env->debug_log_file = NULL;
    }
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
