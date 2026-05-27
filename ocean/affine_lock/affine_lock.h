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
#define AFFINE_LOCK_MAX_SOLUTION_DEPTH 16
#define AFFINE_LOCK_CURRICULUM_DEPTH_COUNT 6
#define AFFINE_LOCK_STEP_REWARD (-0.01f)
#ifndef AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH
#define AFFINE_LOCK_VISIBLE_TARGET_TABLE_PATH \
    "ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin"
#endif

static const int AFFINE_LOCK_CURRICULUM_DEPTHS[
    AFFINE_LOCK_CURRICULUM_DEPTH_COUNT] = {2, 4, 5, 6, 8, 16};

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
    float max_depth_solve;
    float episode_return;
    float episode_length;
    float solve_steps;
    float timeout_rate;
    float invalid_rate;
    float solve_efficiency;
    float target_distance;
    float solved_target_distance;
    float depth_2_rate;
    float depth_2_solve_rate;
    float depth_4_rate;
    float depth_4_solve_rate;
    float depth_5_rate;
    float depth_5_solve_rate;
    float depth_6_rate;
    float depth_6_solve_rate;
    float depth_8_rate;
    float depth_8_solve_rate;
    float depth_16_rate;
    float depth_16_solve_rate;
    float n;
} Log;

typedef struct AffineLockShared {
    int start_depth;
    int max_depth;
    int step_grace;
    int num_states;
    uint32_t mask;
    uint32_t* next;
    int visible_target_table_loaded;
    AffineLockVisibleTargetTable visible_target_table;
    float observation_bit_patterns[256][8];
} AffineLockShared;

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
    int target_distance;
    float episode_return;
    unsigned int rng;
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
        int step_grace) {
    memset(shared, 0, sizeof(*shared));

    shared->start_depth = start_depth;
    shared->max_depth = max_depth;
    shared->step_grace = step_grace;
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

static void affine_lock_free_shared(AffineLockShared* shared) {
    if (shared == NULL) {
        return;
    }
    free(shared->next);
    affine_lock_visible_targets_free(&shared->visible_target_table);
    memset(shared, 0, sizeof(*shared));
}

static uint32_t affine_lock_apply_action(
        const AffineLockShared* shared, uint32_t rel, int action) {
    return shared->next[(rel & shared->mask) * AFFINE_LOCK_NUM_ACTIONS + action];
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
    affine_lock_clear_generated_path(env);
    env->target = record->target & shared->mask;
    env->target_distance = (int)record->depth;
    affine_lock_store_visible_solution_path(env, record);
}

static void affine_lock_reset_state(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    env->scramble_depth = env->curriculum_depth;
    env->max_steps = env->scramble_depth + shared->step_grace;
    env->step_count = 0;
    env->episode_return = 0.0f;
    env->target_distance = -1;

    affine_lock_generate_visible_target_table_target(env);
    env->max_steps = env->target_distance + shared->step_grace;
}

static void affine_lock_init_env(
        AffineLock* env, AffineLockShared* shared, unsigned int seed) {
    env->shared = shared;
    env->rng = seed;
    env->num_agents = 1;
    env->curriculum_depth = shared->start_depth;
    env->scramble_depth = shared->start_depth;
    env->target_distance = -1;
    env->max_steps = shared->start_depth + shared->step_grace;
    env->step_count = 0;
    env->episode_return = 0.0f;
}

static void affine_lock_add_log(
        AffineLock* env,
        int solved,
        int invalid) {
    AffineLockShared* shared = env->shared;
    int log_depth = affine_lock_log_depth(env);
    int at_max_depth = log_depth == shared->max_depth;
    float solve_credit = solved ?
        affine_lock_solve_credit(shared, log_depth) : 0.0f;
    env->log.perf += solve_credit;
    env->log.score += solve_credit;
    env->log.solve_rate += solved ? 1.0f : 0.0f;
    env->log.max_depth_solve +=
        (solved && at_max_depth) ? 1.0f : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->step_count;
    env->log.solve_steps += solved ? (float)env->step_count : 0.0f;
    env->log.timeout_rate += (!solved && !invalid) ? 1.0f : 0.0f;
    env->log.invalid_rate += invalid ? 1.0f : 0.0f;
    env->log.solve_efficiency += solved && log_depth > 0 ?
        (float)env->step_count / (float)log_depth : 0.0f;
    env->log.target_distance += (float)env->target_distance;
    env->log.solved_target_distance +=
        (solved && env->target_distance >= 0) ? (float)env->target_distance : 0.0f;
    env->log.depth_2_rate += log_depth == 2 ? 1.0f : 0.0f;
    env->log.depth_2_solve_rate +=
        (solved && log_depth == 2) ? 1.0f : 0.0f;
    env->log.depth_4_rate += log_depth == 4 ? 1.0f : 0.0f;
    env->log.depth_4_solve_rate +=
        (solved && log_depth == 4) ? 1.0f : 0.0f;
    env->log.depth_5_rate += log_depth == 5 ? 1.0f : 0.0f;
    env->log.depth_5_solve_rate +=
        (solved && log_depth == 5) ? 1.0f : 0.0f;
    env->log.depth_6_rate += log_depth == 6 ? 1.0f : 0.0f;
    env->log.depth_6_solve_rate +=
        (solved && log_depth == 6) ? 1.0f : 0.0f;
    env->log.depth_8_rate += log_depth == 8 ? 1.0f : 0.0f;
    env->log.depth_8_solve_rate +=
        (solved && log_depth == 8) ? 1.0f : 0.0f;
    env->log.depth_16_rate += log_depth == 16 ? 1.0f : 0.0f;
    env->log.depth_16_solve_rate +=
        (solved && log_depth == 16) ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void affine_lock_compute_observations(AffineLock* env) {
    float (*patterns)[8] = env->shared->observation_bit_patterns;
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
    affine_lock_reset_state(env);
    compute_observations(env);
}

static int affine_lock_next_curriculum_depth(
        const AffineLockShared* shared,
        int current_depth) {
    for (int i = 0; i < AFFINE_LOCK_CURRICULUM_DEPTH_COUNT; i++) {
        int depth = AFFINE_LOCK_CURRICULUM_DEPTHS[i];
        if (depth > current_depth) {
            return depth < shared->max_depth ? depth : shared->max_depth;
        }
    }
    return shared->max_depth;
}

static void affine_lock_advance_curriculum(AffineLock* env, int solved) {
    AffineLockShared* shared = env->shared;
    if (!solved) {
        env->curriculum_depth = shared->start_depth;
        return;
    }

    env->curriculum_depth = affine_lock_next_curriculum_depth(
        shared, env->scramble_depth);
}

static void affine_lock_finish_episode(
        AffineLock* env,
        int solved,
        int invalid) {
    affine_lock_add_log(env, solved, invalid);
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

    env->terminals[0] = 0.0f;
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
    env->rewards[0] = reward;
    env->episode_return += reward;

    if (terminal) {
        env->terminals[0] = 1.0f;
        affine_lock_finish_episode(env, solved, invalid);
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
    float display_reward = env->rewards[0];
    int display_terminal = env->terminals[0] != 0.0f;
    int display_solved = display_terminal && display_reward > 0.0f;
    const char* status = "running";
    Color status_color = (Color){190, 198, 206, 255};
    if (display_terminal) {
        status = display_solved ? "solved" : "failed";
        status_color = display_solved ?
            (Color){80, 210, 140, 255} : (Color){238, 88, 88, 255};
    }

    BeginDrawing();
    ClearBackground((Color){12, 15, 18, 255});
    DrawText("Affine Lock", 30, 24, 28, RAYWHITE);
    DrawText(TextFormat("depth %d/%d  step %d/%d  last reward %.2f",
        env->scramble_depth, env->shared->max_depth,
        env->step_count, env->max_steps, display_reward),
        30, 62, 20, (Color){180, 190, 200, 255});
    DrawText(TextFormat("status %s  mismatches 0x%04x",
        status, rel), 30, 90, 20, status_color);

    affine_lock_draw_bit_row(env, "current", env->state, 138);
    affine_lock_draw_bit_row(env, "target", env->target, 220);

    DrawText("1 shiftL  2 shiftR  3 inv7  4 bit-swap  5 pair-swap",
        30, 300, 16, (Color){160, 170, 178, 255});
    DrawText("6 nib-swap  7 rev-nib  8 rev-byte  R reset",
        30, 322, 16, (Color){160, 170, 178, 255});
    EndDrawing();
}
#else
static void c_render(AffineLock* env) {
    (void)env;
}
#endif
