#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define BB_SIZE 8
#define BB_CELLS (BB_SIZE * BB_SIZE)
#define BB_SLOTS 3
#define BB_PIECE_GRID 5
#define BB_PIECE_OBS (BB_SLOTS * BB_PIECE_GRID * BB_PIECE_GRID)
#define BB_OBS_SIZE (BB_CELLS + BB_PIECE_OBS + BB_SLOTS)
#define BB_ACTIONS (BB_SLOTS * BB_CELLS)
#define BB_MAX_PIECE_CELLS 9
#define BB_NUM_PIECES 30

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float lines_cleared;
    float clears;
    float placements;
    float invalid_actions;
    float board_fill;
    float available_moves;
    float avg_board_fill;
    float avg_available_moves;
    float dead_space;
    float empty_components;
    float largest_empty_region;
    float avg_dead_space;
    float avg_empty_components;
    float avg_largest_empty_region;
    float avg_min_slot_moves;
    float low_mobility_steps;
    float game_over;
    float max_combo;
    float survived_max_steps;
    float n;
} Log;

typedef struct {
    int dummy;
} State;

typedef struct {
    int cell_size;
    int gap;
    int last_slot;
    int last_row;
    int last_col;
} Client;

typedef struct {
    int count;
    int row[BB_MAX_PIECE_CELLS];
    int col[BB_MAX_PIECE_CELLS];
    int height;
    int width;
} BlockPiece;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    State state;

    int max_steps;
    float place_reward;
    float line_reward;
    float combo_reward;
    float free_space_reward;
    float mobility_reward;
    float fill_penalty;
    float no_clear_penalty;
    float dead_space_penalty;
    float fragmentation_penalty;
    float low_mobility_penalty;
    int low_mobility_threshold;
    float invalid_penalty;
    float terminal_penalty;

    unsigned char board[BB_CELLS];
    int pieces[BB_SLOTS];
    unsigned char active[BB_SLOTS];
    int steps;
    int score;
    int lines_cleared;
    int clears;
    int placements;
    int invalid_actions;
    int combo;
    int max_combo;
    float episode_return;
    float board_fill_sum;
    float available_moves_sum;
    float dead_space_sum;
    float empty_components_sum;
    float largest_empty_region_sum;
    float min_slot_moves_sum;
    int low_mobility_steps;
    int mobility_samples;
    Client* client;
} BlockBlast;

typedef struct {
    int empty_cells;
    int components;
    int largest_region;
    int small_region_cells;
} EmptyStats;

static const BlockPiece BB_PIECES[BB_NUM_PIECES] = {
    {1, {0}, {0}, 1, 1},
    {2, {0, 0}, {0, 1}, 1, 2},
    {2, {0, 1}, {0, 0}, 2, 1},
    {3, {0, 0, 0}, {0, 1, 2}, 1, 3},
    {3, {0, 1, 2}, {0, 0, 0}, 3, 1},
    {4, {0, 0, 0, 0}, {0, 1, 2, 3}, 1, 4},
    {4, {0, 1, 2, 3}, {0, 0, 0, 0}, 4, 1},
    {5, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}, 1, 5},
    {5, {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0}, 5, 1},
    {4, {0, 0, 1, 1}, {0, 1, 0, 1}, 2, 2},
    {9, {0, 0, 0, 1, 1, 1, 2, 2, 2}, {0, 1, 2, 0, 1, 2, 0, 1, 2}, 3, 3},
    {6, {0, 0, 0, 1, 1, 1}, {0, 1, 2, 0, 1, 2}, 2, 3},
    {6, {0, 0, 1, 1, 2, 2}, {0, 1, 0, 1, 0, 1}, 3, 2},
    {3, {0, 1, 1}, {0, 0, 1}, 2, 2},
    {3, {0, 0, 1}, {0, 1, 0}, 2, 2},
    {3, {0, 0, 1}, {0, 1, 1}, 2, 2},
    {3, {0, 1, 1}, {1, 0, 1}, 2, 2},
    {5, {0, 1, 2, 2, 2}, {0, 0, 0, 1, 2}, 3, 3},
    {5, {0, 0, 0, 1, 2}, {0, 1, 2, 0, 0}, 3, 3},
    {5, {0, 0, 0, 1, 2}, {0, 1, 2, 2, 2}, 3, 3},
    {5, {0, 1, 2, 2, 2}, {2, 2, 0, 1, 2}, 3, 3},
    {4, {0, 0, 0, 1}, {0, 1, 2, 1}, 2, 3},
    {4, {0, 1, 1, 2}, {1, 0, 1, 1}, 3, 2},
    {4, {0, 1, 1, 1}, {1, 0, 1, 2}, 2, 3},
    {4, {0, 1, 1, 2}, {0, 0, 1, 0}, 3, 2},
    {4, {0, 0, 1, 1}, {0, 1, 1, 2}, 2, 3},
    {4, {0, 0, 1, 1}, {1, 2, 0, 1}, 2, 3},
    {4, {0, 1, 1, 2}, {0, 0, 1, 1}, 3, 2},
    {4, {0, 1, 1, 2}, {1, 0, 1, 0}, 3, 2},
    {5, {0, 1, 1, 1, 2}, {1, 0, 1, 2, 1}, 3, 3},
};

static const Color BB_BG = {6, 24, 24, 255};
static const Color BB_GRID = {23, 57, 58, 255};
static const Color BB_CELL = {66, 180, 255, 255};
static const Color BB_CELL_2 = {86, 211, 146, 255};
static const Color BB_TEXT = {241, 241, 241, 255};
static const Color BB_GHOST = {255, 204, 64, 255};
static const Color BB_BAD = {231, 76, 60, 255};

static inline int bb_min(int a, int b) { return a < b ? a : b; }
static inline int bb_max(int a, int b) { return a > b ? a : b; }
static inline int bb_idx(int row, int col) { return row * BB_SIZE + col; }
static inline float bb_rand_unit(BlockBlast* env) {
    return (float)rand_r(&env->rng) / (float)RAND_MAX;
}

static inline int bb_piece_fits(BlockBlast* env, int piece_id, int row, int col) {
    if (piece_id < 0 || piece_id >= BB_NUM_PIECES) return 0;
    const BlockPiece* p = &BB_PIECES[piece_id];
    if (row < 0 || col < 0 || row + p->height > BB_SIZE || col + p->width > BB_SIZE) return 0;
    for (int i = 0; i < p->count; i++) {
        int r = row + p->row[i];
        int c = col + p->col[i];
        if (env->board[bb_idx(r, c)] != 0) return 0;
    }
    return 1;
}

static inline int bb_has_any_legal(BlockBlast* env) {
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        if (!env->active[slot]) continue;
        int piece = env->pieces[slot];
        for (int row = 0; row < BB_SIZE; row++) {
            for (int col = 0; col < BB_SIZE; col++) {
                if (bb_piece_fits(env, piece, row, col)) return 1;
            }
        }
    }
    return 0;
}

static inline int bb_write_action_mask(BlockBlast* env) {
    int legal = 0;
    if (env->action_mask == NULL) return bb_has_any_legal(env);
    memset(env->action_mask, 0, BB_ACTIONS * sizeof(unsigned char));
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        if (!env->active[slot]) continue;
        int piece = env->pieces[slot];
        for (int row = 0; row < BB_SIZE; row++) {
            for (int col = 0; col < BB_SIZE; col++) {
                int action = slot * BB_CELLS + bb_idx(row, col);
                if (bb_piece_fits(env, piece, row, col)) {
                    env->action_mask[action] = 255;
                    legal++;
                }
            }
        }
    }
    return legal > 0;
}

static inline void bb_draw_piece_tray(BlockBlast* env) {
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        env->pieces[slot] = rand_r(&env->rng) % BB_NUM_PIECES;
        env->active[slot] = 1;
    }
}

static inline int bb_all_slots_used(BlockBlast* env) {
    for (int i = 0; i < BB_SLOTS; i++) {
        if (env->active[i]) return 0;
    }
    return 1;
}

static inline int bb_count_occupied(BlockBlast* env) {
    int count = 0;
    for (int i = 0; i < BB_CELLS; i++) count += env->board[i] != 0;
    return count;
}

static inline int bb_slot_available_moves(BlockBlast* env, int slot) {
    if (slot < 0 || slot >= BB_SLOTS || !env->active[slot]) return 0;
    int count = 0;
    int piece = env->pieces[slot];
    for (int row = 0; row < BB_SIZE; row++) {
        for (int col = 0; col < BB_SIZE; col++) {
            count += bb_piece_fits(env, piece, row, col);
        }
    }
    return count;
}

static inline int bb_available_moves(BlockBlast* env) {
    int count = 0;
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        count += bb_slot_available_moves(env, slot);
    }
    return count;
}

static inline int bb_min_active_slot_moves(BlockBlast* env) {
    int min_moves = BB_ACTIONS;
    int active_count = 0;
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        if (!env->active[slot]) continue;
        int moves = bb_slot_available_moves(env, slot);
        min_moves = bb_min(min_moves, moves);
        active_count++;
    }
    return active_count > 0 ? min_moves : 0;
}

static inline EmptyStats bb_empty_stats(BlockBlast* env) {
    EmptyStats stats = {0};
    unsigned char visited[BB_CELLS] = {0};
    int queue[BB_CELLS];

    for (int start = 0; start < BB_CELLS; start++) {
        if (env->board[start] != 0 || visited[start]) continue;

        int head = 0;
        int tail = 0;
        int region = 0;
        visited[start] = 1;
        queue[tail++] = start;

        while (head < tail) {
            int idx = queue[head++];
            int row = idx / BB_SIZE;
            int col = idx % BB_SIZE;
            region++;

            const int dr[4] = {-1, 1, 0, 0};
            const int dc[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; k++) {
                int nr = row + dr[k];
                int nc = col + dc[k];
                if (nr < 0 || nr >= BB_SIZE || nc < 0 || nc >= BB_SIZE) continue;
                int ni = bb_idx(nr, nc);
                if (env->board[ni] != 0 || visited[ni]) continue;
                visited[ni] = 1;
                queue[tail++] = ni;
            }
        }

        stats.empty_cells += region;
        stats.components++;
        stats.largest_region = bb_max(stats.largest_region, region);
        if (region <= 3) {
            stats.small_region_cells += region;
        }
    }

    return stats;
}

static inline void bb_compute_observations(BlockBlast* env) {
    int offset = 0;
    for (int i = 0; i < BB_CELLS; i++) {
        env->observations[offset++] = env->board[i] ? 255 : 0;
    }
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        unsigned char active = env->active[slot] ? 255 : 0;
        unsigned char piece_obs[BB_PIECE_GRID * BB_PIECE_GRID] = {0};
        if (env->active[slot]) {
            const BlockPiece* p = &BB_PIECES[env->pieces[slot]];
            for (int i = 0; i < p->count; i++) {
                piece_obs[p->row[i] * BB_PIECE_GRID + p->col[i]] = 255;
            }
        }
        memcpy(env->observations + offset, piece_obs, sizeof(piece_obs));
        offset += BB_PIECE_GRID * BB_PIECE_GRID;
        env->observations[BB_CELLS + BB_PIECE_OBS + slot] = active;
    }
    bb_write_action_mask(env);
}

static inline void bb_reset_game(BlockBlast* env) {
    memset(env->board, 0, sizeof(env->board));
    env->steps = 0;
    env->score = 0;
    env->lines_cleared = 0;
    env->clears = 0;
    env->placements = 0;
    env->invalid_actions = 0;
    env->combo = 0;
    env->max_combo = 0;
    env->episode_return = 0.0f;
    env->board_fill_sum = 0.0f;
    env->available_moves_sum = 0.0f;
    env->dead_space_sum = 0.0f;
    env->empty_components_sum = 0.0f;
    env->largest_empty_region_sum = 0.0f;
    env->min_slot_moves_sum = 0.0f;
    env->low_mobility_steps = 0;
    env->mobility_samples = 0;
    if (env->client != NULL) {
        env->client->last_slot = -1;
        env->client->last_row = -1;
        env->client->last_col = -1;
    }
    bb_draw_piece_tray(env);
}

static inline int bb_place_piece(BlockBlast* env, int slot, int row, int col) {
    if (slot < 0 || slot >= BB_SLOTS || !env->active[slot]) return 0;
    int piece_id = env->pieces[slot];
    if (!bb_piece_fits(env, piece_id, row, col)) return 0;
    const BlockPiece* p = &BB_PIECES[piece_id];
    for (int i = 0; i < p->count; i++) {
        int r = row + p->row[i];
        int c = col + p->col[i];
        env->board[bb_idx(r, c)] = (unsigned char)(1 + (slot % 2));
    }
    env->active[slot] = 0;
    if (env->client != NULL) {
        env->client->last_slot = slot;
        env->client->last_row = row;
        env->client->last_col = col;
    }
    return p->count;
}

static inline int bb_clear_lines(BlockBlast* env, int* cleared_cells) {
    unsigned char full_rows[BB_SIZE] = {0};
    unsigned char full_cols[BB_SIZE] = {0};
    int lines = 0;
    *cleared_cells = 0;

    for (int row = 0; row < BB_SIZE; row++) {
        int full = 1;
        for (int col = 0; col < BB_SIZE; col++) {
            if (env->board[bb_idx(row, col)] == 0) {
                full = 0;
                break;
            }
        }
        if (full) {
            full_rows[row] = 1;
            lines++;
        }
    }

    for (int col = 0; col < BB_SIZE; col++) {
        int full = 1;
        for (int row = 0; row < BB_SIZE; row++) {
            if (env->board[bb_idx(row, col)] == 0) {
                full = 0;
                break;
            }
        }
        if (full) {
            full_cols[col] = 1;
            lines++;
        }
    }

    if (lines == 0) return 0;

    for (int row = 0; row < BB_SIZE; row++) {
        for (int col = 0; col < BB_SIZE; col++) {
            int idx = bb_idx(row, col);
            if ((full_rows[row] || full_cols[col]) && env->board[idx] != 0) {
                env->board[idx] = 0;
                (*cleared_cells)++;
            }
        }
    }

    env->lines_cleared += lines;
    env->clears++;
    env->combo++;
    env->max_combo = bb_max(env->max_combo, env->combo);
    return lines;
}

static inline void bb_write_log(BlockBlast* env, int game_over, int survived_max_steps) {
    env->log.score += env->score;
    env->log.perf += env->score / 1000.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->steps;
    env->log.lines_cleared += env->lines_cleared;
    env->log.clears += env->clears;
    env->log.placements += env->placements;
    env->log.invalid_actions += env->invalid_actions;
    env->log.board_fill += (float)bb_count_occupied(env) / (float)BB_CELLS;
    env->log.available_moves += bb_available_moves(env);
    EmptyStats final_empty = bb_empty_stats(env);
    env->log.dead_space +=
        (float)(final_empty.empty_cells - final_empty.largest_region) / (float)BB_CELLS;
    env->log.empty_components += final_empty.components;
    env->log.largest_empty_region += final_empty.largest_region;
    if (env->mobility_samples > 0) {
        env->log.avg_board_fill += env->board_fill_sum / (float)env->mobility_samples;
        env->log.avg_available_moves += env->available_moves_sum / (float)env->mobility_samples;
        env->log.avg_dead_space += env->dead_space_sum / (float)env->mobility_samples;
        env->log.avg_empty_components +=
            env->empty_components_sum / (float)env->mobility_samples;
        env->log.avg_largest_empty_region +=
            env->largest_empty_region_sum / (float)env->mobility_samples;
        env->log.avg_min_slot_moves += env->min_slot_moves_sum / (float)env->mobility_samples;
    }
    env->log.low_mobility_steps += env->low_mobility_steps;
    env->log.game_over += game_over;
    env->log.max_combo += env->max_combo;
    env->log.survived_max_steps += survived_max_steps;
    env->log.n += 1.0f;
}

static inline void c_reset(BlockBlast* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    bb_reset_game(env);
    bb_compute_observations(env);
}

static inline void c_step(BlockBlast* env) {
    int action = (int)env->actions[0];
    int slot = action / BB_CELLS;
    int cell = action % BB_CELLS;
    int row = cell / BB_SIZE;
    int col = cell % BB_SIZE;
    int cleared_cells = 0;
    int lines = 0;
    int placed = 0;
    int terminal = 0;
    int survived_max_steps = 0;
    float reward = 0.0f;

    env->terminals[0] = 0.0f;
    if (action < 0 || action >= BB_ACTIONS) {
        placed = 0;
    } else {
        placed = bb_place_piece(env, slot, row, col);
    }

    if (placed <= 0) {
        env->invalid_actions++;
        reward = env->invalid_penalty;
        env->combo = 0;
    } else {
        env->placements++;
        env->steps++;
        lines = bb_clear_lines(env, &cleared_cells);
        if (lines > 0) {
            int line_score = 10 * lines * lines + cleared_cells;
            env->score += placed + line_score + env->combo * 2;
            reward += env->place_reward * (float)placed;
            reward += env->line_reward * (float)(lines * lines);
            reward += env->combo_reward * (float)env->combo;
        } else {
            env->combo = 0;
            env->score += placed;
            reward += env->place_reward * (float)placed;
            reward -= env->no_clear_penalty;
        }

        if (bb_all_slots_used(env)) {
            bb_draw_piece_tray(env);
        }

        int legal_moves = bb_available_moves(env);
        int min_slot_moves = bb_min_active_slot_moves(env);
        EmptyStats empty = bb_empty_stats(env);
        float fill = (float)bb_count_occupied(env) / (float)BB_CELLS;
        float dead_space =
            (float)(empty.empty_cells - empty.largest_region) / (float)BB_CELLS;
        float fragmentation =
            empty.components > 1 ? (float)(empty.components - 1) / (float)BB_SIZE : 0.0f;
        float low_mobility = 0.0f;
        if (legal_moves < env->low_mobility_threshold) {
            low_mobility = (float)(env->low_mobility_threshold - legal_moves)
                / (float)bb_max(env->low_mobility_threshold, 1);
            env->low_mobility_steps++;
        }
        env->board_fill_sum += fill;
        env->available_moves_sum += (float)legal_moves;
        env->dead_space_sum += dead_space;
        env->empty_components_sum += (float)empty.components;
        env->largest_empty_region_sum += (float)empty.largest_region;
        env->min_slot_moves_sum += (float)min_slot_moves;
        env->mobility_samples++;
        reward += env->free_space_reward * (1.0f - fill);
        reward += env->mobility_reward * ((float)legal_moves / (float)BB_ACTIONS);
        reward -= env->fill_penalty * fill * fill;
        reward -= env->dead_space_penalty * dead_space;
        reward -= env->fragmentation_penalty * fragmentation;
        reward -= env->low_mobility_penalty * low_mobility;

        if (env->steps >= env->max_steps) {
            terminal = 1;
            survived_max_steps = 1;
            reward += 2.0f;
        } else if (!bb_has_any_legal(env)) {
            terminal = 1;
            reward -= env->terminal_penalty;
        }
    }

    env->rewards[0] = reward;
    env->episode_return += reward;

    if (terminal) {
        env->terminals[0] = 1.0f;
        bb_write_log(env, !survived_max_steps, survived_max_steps);
        bb_reset_game(env);
        env->rewards[0] = reward;
    }

    bb_compute_observations(env);
}

static inline void c_close(BlockBlast* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

static inline Client* bb_make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cell_size = 56;
    client->gap = 5;
    client->last_slot = -1;
    client->last_row = -1;
    client->last_col = -1;
    InitWindow(720, 540, "PufferLib Block Blast");
    SetTargetFPS(12);
    return client;
}

static inline void bb_draw_board(BlockBlast* env, int x0, int y0, int cell, int gap) {
    for (int row = 0; row < BB_SIZE; row++) {
        for (int col = 0; col < BB_SIZE; col++) {
            int x = x0 + col * cell;
            int y = y0 + row * cell;
            unsigned char value = env->board[bb_idx(row, col)];
            DrawRectangle(x, y, cell - gap, cell - gap, BB_GRID);
            if (value != 0) {
                DrawRectangle(x + 3, y + 3, cell - gap - 6, cell - gap - 6,
                    value == 1 ? BB_CELL : BB_CELL_2);
            }
        }
    }
}

static inline void bb_draw_piece_preview(BlockBlast* env, int slot, int x0, int y0) {
    const int small = 18;
    const int gap = 3;
    DrawText(TextFormat("%d", slot + 1), x0, y0 - 22, 18, BB_TEXT);
    if (!env->active[slot]) {
        DrawText("used", x0 + 26, y0 - 22, 16, BB_GRID);
        return;
    }
    const BlockPiece* p = &BB_PIECES[env->pieces[slot]];
    for (int i = 0; i < p->count; i++) {
        int x = x0 + p->col[i] * small;
        int y = y0 + p->row[i] * small;
        DrawRectangle(x, y, small - gap, small - gap, slot % 2 == 0 ? BB_CELL : BB_CELL_2);
    }
}

static inline void c_render(BlockBlast* env) {
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        c_close(env);
        exit(0);
    }
    if (env->client == NULL) env->client = bb_make_client();

    BeginDrawing();
    ClearBackground(BB_BG);
    bb_draw_board(env, 28, 46, env->client->cell_size, env->client->gap);
    DrawText("Block Blast", 28, 12, 24, BB_TEXT);
    DrawText(TextFormat("Score %d", env->score), 520, 44, 22, BB_TEXT);
    DrawText(TextFormat("Step %d/%d", env->steps, env->max_steps), 520, 74, 18, BB_TEXT);
    DrawText(TextFormat("Lines %d", env->lines_cleared), 520, 100, 18, BB_TEXT);
    DrawText(TextFormat("Moves %d", bb_available_moves(env)), 520, 126, 18, BB_TEXT);
    for (int slot = 0; slot < BB_SLOTS; slot++) {
        bb_draw_piece_preview(env, slot, 520, 190 + slot * 104);
    }
    if (env->client->last_slot >= 0) {
        DrawText(TextFormat("Last: piece %d at %d,%d",
            env->client->last_slot + 1, env->client->last_row, env->client->last_col),
            28, 502, 18, BB_GHOST);
    }
    if (!bb_has_any_legal(env)) {
        DrawText("No legal moves", 520, 480, 20, BB_BAD);
    }
    EndDrawing();
}
