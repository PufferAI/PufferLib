/* Nonogram: A logic puzzle environment
 * Players fill cells based on row and column clues (run-length encoding)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MAX_SIZE 8
#define MAX_CLUES (MAX_SIZE / 2)

const unsigned char EMPTY = 0;
const unsigned char FILLED = 1;

const float REWARD_WIN = 1.0;
const float REWARD_INVALID_MOVE = -0.01;
const float REWARD_TIMEOUT = -1.0;
const float REWARD_COMPLETE_LINE = 0.01;

// Required struct for logging
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

// Nonogram environment struct
typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    // Environment state
    int size;
    int max_steps;
    int steps_taken;
    int filled_total;
    int target_total;

    // Solution (for generating clues)
    unsigned char solution[MAX_SIZE * MAX_SIZE];

    // Clues
    unsigned char rows_clues[MAX_SIZE * MAX_CLUES];
    unsigned char cols_clues[MAX_SIZE * MAX_CLUES];
    unsigned char rows_num_runs[MAX_SIZE];
    unsigned char cols_num_runs[MAX_SIZE];
    unsigned char rows_target_sum[MAX_SIZE];
    unsigned char cols_target_sum[MAX_SIZE];
    unsigned char rows_max_clue[MAX_SIZE];
    unsigned char cols_max_clue[MAX_SIZE];

    // Current totals
    unsigned char rows_totals[MAX_SIZE];
    unsigned char cols_totals[MAX_SIZE];

    // Track completed lines
    unsigned char rows_completed[MAX_SIZE];
    unsigned char cols_completed[MAX_SIZE];

    // Episode reward accumulator
    float episode_reward;
} Nonogram;

// Helper function implementations
void add_log(Nonogram* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->steps_taken;
    env->log.episode_return += env->episode_reward;
    env->log.n++;
}

int get_row_run_length(Nonogram* env, int row, int col) {
    int row_start = row * env->size;
    int run_length = 1;

    // Count left
    for (int c = col - 1; c >= 0; c--) {
        if (env->observations[row_start + c] == FILLED) {
            run_length++;
        } else {
            break;
        }
    }

    // Count right
    for (int c = col + 1; c < env->size; c++) {
        if (env->observations[row_start + c] == FILLED) {
            run_length++;
        } else {
            break;
        }
    }

    return run_length;
}

int get_col_run_length(Nonogram* env, int row, int col) {
    int run_length = 1;

    // Count up
    for (int r = row - 1; r >= 0; r--) {
        if (env->observations[r * env->size + col] == FILLED) {
            run_length++;
        } else {
            break;
        }
    }

    // Count down
    for (int r = row + 1; r < env->size; r++) {
        if (env->observations[r * env->size + col] == FILLED) {
            run_length++;
        } else {
            break;
        }
    }

    return run_length;
}

int check_line_matches(unsigned char* line_data, unsigned char* clues, int num_runs, int size) {
    int run_idx = 0;
    int count = 0;

    for (int i = 0; i < size; i++) {
        count += line_data[i];
        if (line_data[i] == 0 && count > 0) {
            if (clues[run_idx] != count) {
                return 0;
            }
            run_idx++;
            count = 0;
        }
    }

    // Check final run
    if (count > 0) {
        if (clues[run_idx] != count) {
            return 0;
        }
        run_idx++;
    }

    return run_idx == num_runs;
}

// Required functions
void c_reset(Nonogram* env) {
    int grid_size = env->size * env->size;

    // Initialize player grid as all EMPTY
    memset(env->observations, EMPTY, grid_size * sizeof(unsigned char));

    // Generate random solution
    for (int i = 0; i < grid_size; i++) {
        env->solution[i] = (rand() % 2 == 0) ? EMPTY : FILLED;
    }

    // Reset clues arrays
    int max_clues = env->size / 2;
    memset(env->rows_clues, 0, MAX_SIZE * MAX_CLUES);
    memset(env->cols_clues, 0, MAX_SIZE * MAX_CLUES);

    // Calculate row clues
    for (int i = 0; i < env->size; i++) {
        int clue_idx = 0;
        int count = 0;
        for (int j = 0; j < env->size; j++) {
            if (env->solution[i * env->size + j] == FILLED) {
                count++;
            } else if (count > 0) {
                env->rows_clues[i * MAX_CLUES + clue_idx] = count;
                clue_idx++;
                count = 0;
            }
        }
        if (count > 0) {
            env->rows_clues[i * MAX_CLUES + clue_idx] = count;
            clue_idx++;
        }
        env->rows_num_runs[i] = clue_idx;
    }

    // Calculate column clues
    for (int j = 0; j < env->size; j++) {
        int clue_idx = 0;
        int count = 0;
        for (int i = 0; i < env->size; i++) {
            if (env->solution[i * env->size + j] == FILLED) {
                count++;
            } else if (count > 0) {
                env->cols_clues[j * MAX_CLUES + clue_idx] = count;
                clue_idx++;
                count = 0;
            }
        }
        if (count > 0) {
            env->cols_clues[j * MAX_CLUES + clue_idx] = count;
            clue_idx++;
        }
        env->cols_num_runs[j] = clue_idx;
    }

    // Store clues in observation
    int clue_size = env->size * max_clues;
    memcpy(env->observations + grid_size, env->rows_clues, clue_size);
    memcpy(env->observations + grid_size + clue_size, env->cols_clues, clue_size);

    // Calculate max clues and target sums
    memset(env->rows_totals, 0, MAX_SIZE);
    memset(env->cols_totals, 0, MAX_SIZE);
    memset(env->rows_completed, 0, MAX_SIZE);
    memset(env->cols_completed, 0, MAX_SIZE);
    env->filled_total = 0;

    for (int i = 0; i < env->size; i++) {
        // Find max clue for row
        int max_clue = 0;
        int sum = 0;
        for (int j = 0; j < max_clues; j++) {
            int clue = env->rows_clues[i * MAX_CLUES + j];
            if (clue > max_clue) {
                max_clue = clue;
            }
            sum += clue;
        }
        env->rows_max_clue[i] = max_clue;
        env->rows_target_sum[i] = sum;

        // Find max clue for col
        max_clue = 0;
        sum = 0;
        for (int j = 0; j < max_clues; j++) {
            int clue = env->cols_clues[i * MAX_CLUES + j];
            if (clue > max_clue) {
                max_clue = clue;
            }
            sum += clue;
        }
        env->cols_max_clue[i] = max_clue;
        env->cols_target_sum[i] = sum;
    }

    // Calculate target total
    env->target_total = 0;
    for (int i = 0; i < env->size; i++) {
        env->target_total += env->rows_target_sum[i];
    }

    env->steps_taken = 0;
    env->episode_reward = 0;
}

void c_step(Nonogram* env) {
    int pos = env->actions[0];
    int row = pos / env->size;
    int col = pos % env->size;

    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->steps_taken++;

    // Check timeout FIRST before any game logic
    if (env->steps_taken >= env->max_steps) {
        env->terminals[0] = 1;
        env->rewards[0] = REWARD_TIMEOUT;
        env->episode_reward += REWARD_TIMEOUT;
        add_log(env);
        c_reset(env);
        return;
    }

    unsigned char current = env->observations[pos];

    // If toggling on (EMPTY -> FILLED)
    if (current == EMPTY) {
        // First check: totals equal target - invalid move
        if (env->rows_totals[row] == env->rows_target_sum[row] ||
            env->cols_totals[col] == env->cols_target_sum[col]) {
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            return;
        }

        // Check if filling this cell would create a run longer than max allowed
        if (get_row_run_length(env, row, col) > env->rows_max_clue[row]) {
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            return;
        }

        if (get_col_run_length(env, row, col) > env->cols_max_clue[col]) {
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            return;
        }

        // Second check: if completing row/col, check runs match
        int row_completed = 0;
        int col_completed = 0;

        if (env->rows_totals[row] == env->rows_target_sum[row] - 1) {
            // Temporarily fill to check
            env->observations[pos] = FILLED;
            int row_start = row * env->size;
            if (!check_line_matches(env->observations + row_start,
                                   env->rows_clues + row * MAX_CLUES,
                                   env->rows_num_runs[row], env->size)) {
                // Runs don't match - invalid move
                env->observations[pos] = EMPTY;
                env->rewards[0] = REWARD_INVALID_MOVE;
                env->episode_reward += REWARD_INVALID_MOVE;
                return;
            }
            env->observations[pos] = EMPTY;
            row_completed = 1;
        }

        if (env->cols_totals[col] == env->cols_target_sum[col] - 1) {
            // Temporarily fill to check
            env->observations[pos] = FILLED;
            unsigned char col_data[MAX_SIZE];
            for (int i = 0; i < env->size; i++) {
                col_data[i] = env->observations[i * env->size + col];
            }
            if (!check_line_matches(col_data,
                                   env->cols_clues + col * MAX_CLUES,
                                   env->cols_num_runs[col], env->size)) {
                // Runs don't match - invalid move
                env->observations[pos] = EMPTY;
                env->rewards[0] = REWARD_INVALID_MOVE;
                env->episode_reward += REWARD_INVALID_MOVE;
                return;
            }
            env->observations[pos] = EMPTY;
            col_completed = 1;
        }

        // Apply toggle
        env->observations[pos] = FILLED;
        env->rows_totals[row]++;
        env->cols_totals[col]++;
        env->filled_total++;

        // Give reward for newly completed lines only
        int row_newly_completed = row_completed && !env->rows_completed[row];
        int col_newly_completed = col_completed && !env->cols_completed[col];

        if (row_newly_completed) env->rows_completed[row] = 1;
        if (col_newly_completed) env->cols_completed[col] = 1;

        float line_reward = (row_newly_completed + col_newly_completed) * REWARD_COMPLETE_LINE;
        env->rewards[0] += line_reward;
        env->episode_reward += line_reward;
    } else {
        // Toggling off (FILLED -> EMPTY) - always allowed
        env->observations[pos] = EMPTY;
        env->rows_totals[row]--;
        env->cols_totals[col]--;
        env->filled_total--;
    }

    // Check if solved
    if (env->filled_total == env->target_total) {
        env->terminals[0] = 1;
        env->rewards[0] = REWARD_WIN;
        env->episode_reward += REWARD_WIN;
        add_log(env);
        c_reset(env);
        return;
    }
}

void c_render(Nonogram* env) {
    if (!IsWindowReady()) {
        int board_width = 120 + env->size * 40;
        int board_height = 120 + env->size * 40;
        int screen_width = board_width * 2 + 60 + 40;
        int screen_height = board_height + 140;
        InitWindow(screen_width, screen_height, "Nonogram (C)");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){0, 0, 0, 255});

    int cell_size = 40;
    int clue_area = 120;
    int board_spacing = 60;
    int font_size = 20;
    int max_clues = env->size / 2;

    // Draw titles
    DrawText("CURRENT BOARD", 20, 20, 24, RAYWHITE);
    int solution_x = clue_area + env->size * cell_size + board_spacing + 20;
    DrawText("SOLUTION", solution_x, 20, 24, RAYWHITE);

    // Draw current board
    int offset_x = 20;
    int offset_y = 60;

    // Draw column clues for current board
    for (int clue_row = 0; clue_row < max_clues; clue_row++) {
        for (int c = 0; c < env->size; c++) {
            int clue = env->cols_clues[c * MAX_CLUES + clue_row];
            if (clue > 0) {
                char text[4];
                snprintf(text, sizeof(text), "%d", clue);
                int x = offset_x + clue_area + c * cell_size + cell_size / 2;
                int y = offset_y + clue_row * 20 + 10;
                int text_width = MeasureText(text, font_size);
                DrawText(text, x - text_width / 2, y, font_size, RAYWHITE);
            }
        }
    }

    // Draw row clues for current board
    for (int r = 0; r < env->size; r++) {
        int clue_x = offset_x + 10;
        for (int clue_idx = 0; clue_idx < max_clues; clue_idx++) {
            int clue = env->rows_clues[r * MAX_CLUES + clue_idx];
            if (clue > 0) {
                char text[4];
                snprintf(text, sizeof(text), "%d", clue);
                int y = offset_y + clue_area + r * cell_size + cell_size / 2 - font_size / 2;
                DrawText(text, clue_x, y, font_size, RAYWHITE);
                clue_x += MeasureText(text, font_size) + 5;
            }
        }
    }

    // Draw current grid
    for (int r = 0; r < env->size; r++) {
        for (int c = 0; c < env->size; c++) {
            int x = offset_x + clue_area + c * cell_size;
            int y = offset_y + clue_area + r * cell_size;
            int pos = r * env->size + c;

            if (env->observations[pos] == FILLED) {
                DrawRectangle(x, y, cell_size, cell_size, WHITE);
            } else {
                DrawRectangle(x, y, cell_size, cell_size, DARKGRAY);
            }
            DrawRectangleLines(x, y, cell_size, cell_size, LIGHTGRAY);
        }
    }

    // Draw solution board
    offset_x = solution_x;

    // Draw column clues for solution
    for (int clue_row = 0; clue_row < max_clues; clue_row++) {
        for (int c = 0; c < env->size; c++) {
            int clue = env->cols_clues[c * MAX_CLUES + clue_row];
            if (clue > 0) {
                char text[4];
                snprintf(text, sizeof(text), "%d", clue);
                int x = offset_x + clue_area + c * cell_size + cell_size / 2;
                int y = offset_y + clue_row * 20 + 10;
                int text_width = MeasureText(text, font_size);
                DrawText(text, x - text_width / 2, y, font_size, RAYWHITE);
            }
        }
    }

    // Draw row clues for solution
    for (int r = 0; r < env->size; r++) {
        int clue_x = offset_x + 10;
        for (int clue_idx = 0; clue_idx < max_clues; clue_idx++) {
            int clue = env->rows_clues[r * MAX_CLUES + clue_idx];
            if (clue > 0) {
                char text[4];
                snprintf(text, sizeof(text), "%d", clue);
                int y = offset_y + clue_area + r * cell_size + cell_size / 2 - font_size / 2;
                DrawText(text, clue_x, y, font_size, RAYWHITE);
                clue_x += MeasureText(text, font_size) + 5;
            }
        }
    }

    // Draw solution grid
    for (int r = 0; r < env->size; r++) {
        for (int c = 0; c < env->size; c++) {
            int x = offset_x + clue_area + c * cell_size;
            int y = offset_y + clue_area + r * cell_size;
            int pos = r * env->size + c;

            if (env->solution[pos] == FILLED) {
                DrawRectangle(x, y, cell_size, cell_size, GREEN);
            } else {
                DrawRectangle(x, y, cell_size, cell_size, DARKGRAY);
            }
            DrawRectangleLines(x, y, cell_size, cell_size, LIGHTGRAY);
        }
    }

    // Draw status
    int board_height = clue_area + env->size * cell_size;
    int status_y = board_height + 80;
    char status[128];
    snprintf(status, sizeof(status), "Steps: %d/%d | Filled: %d/%d",
             env->steps_taken, env->max_steps, env->filled_total, env->target_total);
    DrawText(status, 20, status_y, 20, RAYWHITE);

    // Draw instructions
    DrawText("Click cells to toggle | Press R to reset | ESC to quit", 20, status_y + 30, 16, LIGHTGRAY);

    EndDrawing();
}

void c_close(Nonogram* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
