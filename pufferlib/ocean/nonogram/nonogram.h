/* Nonogram: A logic puzzle environment
 * Players fill cells based on row and column clues (run-length encoding)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// Debug mode: set to 1 to enable debug output, 0 to disable
#define DEBUG 0

#if DEBUG
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...) ((void)0)
#endif

#define MAX_SIZE 8
#define MAX_CLUES (MAX_SIZE / 2)

const unsigned char CELL_EMPTY = 0;
const unsigned char CELL_WHITE = 1;
const unsigned char CELL_BLACK = 2;
const unsigned char CELL_PADDING = 3;

const float REWARD_WIN = 1.0;
const float REWARD_INVALID_MOVE = -0.2;
const float REWARD_OUT_OF_BOUNDS = -0.2;
const float REWARD_TIMEOUT = -0.1;
const float REWARD_COMPLETE_LINE = 0.02;
const float REWARD_EASY_LEARN_CORRECT = 0.01;
const float REWARD_EASY_LEARN_INCORRECT = -0.01;
const float REWARD_NO_MATCH = -0.05;

// Required struct for logging
typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float solved;
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
    int min_size;
    int max_size;
    int max_steps;
    int steps_taken;
    int filled_total;
    int target_total;
    int easy_learn;

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
    env->log.score += env->rewards[0];
    env->log.episode_length += env->steps_taken;
    env->log.episode_return += env->episode_reward;
    env->log.solved += (env->rewards[0] > 0) ? 1 : 0;
    env->log.n++;
}

int get_row_run_length(Nonogram* env, int row, int col) {
    int row_start = row * MAX_SIZE;
    int run_length = 1;

    debug_printf("  get_row_run_length: row=%d, col=%d, row_start=%d\n", row, col, row_start);
    debug_printf("  Row cells before marking: ");
    for (int c = 0; c < env->size; c++) {
        debug_printf("%d ", env->observations[row_start + c]);
    }
    debug_printf("\n");

    // Count left
    int left_count = 0;
    for (int c = col - 1; c >= 0; c--) {
        if (env->observations[row_start + c] == CELL_BLACK) {
            run_length++;
            left_count++;
        } else {
            break;
        }
    }
    debug_printf("  Left count: %d\n", left_count);

    // Count right
    int right_count = 0;
    for (int c = col + 1; c < env->size; c++) {
        if (env->observations[row_start + c] == CELL_BLACK) {
            run_length++;
            right_count++;
        } else {
            break;
        }
    }
    debug_printf("  Right count: %d\n", right_count);
    debug_printf("  Total run_length (1 + left + right): %d\n", run_length);

    return run_length;
}

int get_col_run_length(Nonogram* env, int row, int col) {
    int run_length = 1;

    debug_printf("  get_col_run_length: row=%d, col=%d\n", row, col);
    debug_printf("  Col cells before marking: ");
    for (int r = 0; r < env->size; r++) {
        debug_printf("%d ", env->observations[r * MAX_SIZE + col]);
    }
    debug_printf("\n");

    // Count up
    int up_count = 0;
    for (int r = row - 1; r >= 0; r--) {
        if (env->observations[r * MAX_SIZE + col] == CELL_BLACK) {
            run_length++;
            up_count++;
        } else {
            break;
        }
    }
    debug_printf("  Up count: %d\n", up_count);

    // Count down
    int down_count = 0;
    for (int r = row + 1; r < env->size; r++) {
        if (env->observations[r * MAX_SIZE + col] == CELL_BLACK) {
            run_length++;
            down_count++;
        } else {
            break;
        }
    }
    debug_printf("  Down count: %d\n", down_count);
    debug_printf("  Total run_length (1 + up + down): %d\n", run_length);

    return run_length;
}

int check_line_matches(unsigned char* line_data, unsigned char* clues, int num_runs, int size) {
    debug_printf("  check_line_matches: num_runs=%d, size=%d\n", num_runs, size);
    debug_printf("  Line data: ");
    for (int i = 0; i < size; i++) {
        debug_printf("%d ", line_data[i]);
    }
    debug_printf("\n");
    debug_printf("  Expected clues: ");
    for (int i = 0; i < num_runs; i++) {
        debug_printf("%d ", clues[i]);
    }
    debug_printf("\n");

    int run_idx = 0;
    int count = 0;

    for (int i = 0; i < size; i++) {
        if (line_data[i] == CELL_BLACK) {
            count++;
            debug_printf("  Position %d: BLACK, count=%d\n", i, count);
        } else if (line_data[i] == CELL_EMPTY || line_data[i] == CELL_WHITE) {
            if (count > 0) {
                debug_printf("  End of run at position %d: count=%d, expected=%d (run_idx=%d)\n",
                       i, count, clues[run_idx], run_idx);
                if (clues[run_idx] != count) {
                    debug_printf("  MISMATCH! Expected %d but got %d\n", clues[run_idx], count);
                    return 0;
                }
                run_idx++;
                count = 0;
            }
        }
    }

    // Check final run
    if (count > 0) {
        debug_printf("  Final run: count=%d, expected=%d (run_idx=%d)\n", count, clues[run_idx], run_idx);
        if (clues[run_idx] != count) {
            debug_printf("  FINAL MISMATCH! Expected %d but got %d\n", clues[run_idx], count);
            return 0;
        }
        run_idx++;
    }

    debug_printf("  Total runs found: %d, expected: %d\n", run_idx, num_runs);
    int matches = (run_idx == num_runs);
    debug_printf("  Pattern matches: %d\n", matches);
    return matches;
}

// Helper to generate random float in [0, 1]
float rand_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

// Required functions
void c_reset(Nonogram* env) {
    env->size = env->min_size + (rand() % (env->max_size - env->min_size + 1));
    env->max_steps = env->easy_learn ? env->size * env->size : 4 * env->size * env->size;

    int full_grid_size = MAX_SIZE * MAX_SIZE;
    int max_clues = MAX_SIZE / 2;

    // Initialize all grid as PADDING, then clear valid cells to EMPTY (using MAX_SIZE stride)
    memset(env->observations, CELL_PADDING, full_grid_size);
    for (int r = 0; r < env->size; r++) {
        for (int c = 0; c < env->size; c++) {
            env->observations[r * MAX_SIZE + c] = CELL_EMPTY;
        }
    }
    // Clear clue areas
    memset(env->observations + full_grid_size, 0, 2 * MAX_SIZE * max_clues);

    // Generate random solution using MAX_SIZE stride with uniform fill probability
    // Sample fill probability p uniformly from [0, 1] for difficulty variation
    float fill_prob = rand_uniform();
    memset(env->solution, CELL_WHITE, MAX_SIZE * MAX_SIZE);
    int has_filled = 0;
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            if (rand_uniform() < fill_prob) {
                env->solution[i * MAX_SIZE + j] = CELL_BLACK;
                has_filled = 1;
            }
        }
    }

    // Ensure at least one square is set
    if (!has_filled) {
        int rand_row = rand() % env->size;
        int rand_col = rand() % env->size;
        env->solution[rand_row * MAX_SIZE + rand_col] = CELL_BLACK;
    }

    // Reset clues arrays
    memset(env->rows_clues, 0, MAX_SIZE * MAX_CLUES);
    memset(env->cols_clues, 0, MAX_SIZE * MAX_CLUES);

    // Calculate row clues
    for (int i = 0; i < env->size; i++) {
        int clue_idx = 0;
        int count = 0;
        for (int j = 0; j < env->size; j++) {
            if (env->solution[i * MAX_SIZE + j] == CELL_BLACK) {
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
            if (env->solution[i * MAX_SIZE + j] == CELL_BLACK) {
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
    memcpy(env->observations + full_grid_size, env->rows_clues, MAX_SIZE * max_clues);
    memcpy(env->observations + full_grid_size + MAX_SIZE * max_clues, env->cols_clues, MAX_SIZE * max_clues);

    // Store board size as scalar at end of observation
    env->observations[full_grid_size + 2 * MAX_SIZE * max_clues] = env->size;

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

    // Debug: print solution and clues
    debug_printf("\n=== RESET: New puzzle generated (size=%d) ===\n", env->size);
    debug_printf("Solution grid:\n");
    for (int r = 0; r < env->size; r++) {
        debug_printf("  Row %d: ", r);
        for (int c = 0; c < env->size; c++) {
            debug_printf("%d ", env->solution[r * MAX_SIZE + c]);
        }
        debug_printf("\n");
    }

    debug_printf("\nRow clues:\n");
    for (int r = 0; r < env->size; r++) {
        debug_printf("  Row %d (num_runs=%d, target_sum=%d, max_clue=%d): ",
               r, env->rows_num_runs[r], env->rows_target_sum[r], env->rows_max_clue[r]);
        for (int i = 0; i < MAX_CLUES; i++) {
            int clue = env->rows_clues[r * MAX_CLUES + i];
            if (clue > 0) {
                debug_printf("%d ", clue);
            }
        }
        debug_printf("\n");
    }

    debug_printf("\nColumn clues:\n");
    for (int c = 0; c < env->size; c++) {
        debug_printf("  Col %d (num_runs=%d, target_sum=%d, max_clue=%d): ",
               c, env->cols_num_runs[c], env->cols_target_sum[c], env->cols_max_clue[c]);
        for (int i = 0; i < MAX_CLUES; i++) {
            int clue = env->cols_clues[c * MAX_CLUES + i];
            if (clue > 0) {
                debug_printf("%d ", clue);
            }
        }
        debug_printf("\n");
    }

    debug_printf("\nTarget total BLACK cells: %d\n", env->target_total);
    debug_printf("===================================\n\n");

    env->steps_taken = 0;
    env->episode_reward = 0;
}

void c_step(Nonogram* env) {
    int action = env->actions[0];

    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->steps_taken++;

    debug_printf("DEBUG c_step: action=%d, steps=%d\n", action, env->steps_taken);

    // Check timeout FIRST before any game logic
    if (env->steps_taken > env->max_steps) {
        debug_printf("DEBUG: TIMEOUT\n");
        env->terminals[0] = 1;
        env->rewards[0] = REWARD_TIMEOUT;
        env->episode_reward += REWARD_TIMEOUT;
        add_log(env);
        c_reset(env);
        return;
    }

    // Decode action: 0-63 = mark WHITE, 64-127 = mark BLACK
    int mark_black = action >= (MAX_SIZE * MAX_SIZE);
    int pos = action % (MAX_SIZE * MAX_SIZE);

    debug_printf("DEBUG: mark_black=%d, pos=%d\n", mark_black, pos);

    // Convert position to row/col using MAX_SIZE stride
    int row = pos / MAX_SIZE;
    int col = pos % MAX_SIZE;

    debug_printf("DEBUG: row=%d, col=%d, size=%d\n", row, col, env->size);

    // Check if action is out of bounds (hitting padding area)
    if (row >= env->size || col >= env->size) {
        debug_printf("DEBUG: OUT OF BOUNDS (row=%d, col=%d >= size=%d)\n", row, col, env->size);
        env->terminals[0] = 1;
        env->rewards[0] = REWARD_OUT_OF_BOUNDS;
        env->episode_reward += REWARD_OUT_OF_BOUNDS;
        add_log(env);
        c_reset(env);
        return;
    }

    unsigned char current = env->observations[pos];

    debug_printf("DEBUG: current cell value=%d (EMPTY=%d, WHITE=%d, BLACK=%d, PADDING=%d)\n",
           current, CELL_EMPTY, CELL_WHITE, CELL_BLACK, CELL_PADDING);

    // Can't mark a cell that's already been marked
    if (current != CELL_EMPTY) {
        debug_printf("DEBUG: INVALID - cell already marked (current=%d)\n", current);
        env->terminals[0] = 1;
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->episode_reward += REWARD_INVALID_MOVE;
        add_log(env);
        c_reset(env);
        return;
    }

    // Mark cell as BLACK or WHITE
    if (mark_black) {
        debug_printf("DEBUG: Marking BLACK\n");
        // Marking BLACK - check if valid
        // First check: totals equal target - invalid move (terminate episode)
        debug_printf("DEBUG: rows_totals[%d]=%d, rows_target_sum[%d]=%d\n",
               row, env->rows_totals[row], row, env->rows_target_sum[row]);
        debug_printf("DEBUG: cols_totals[%d]=%d, cols_target_sum[%d]=%d\n",
               col, env->cols_totals[col], col, env->cols_target_sum[col]);

        if (env->rows_totals[row] == env->rows_target_sum[row] ||
            env->cols_totals[col] == env->cols_target_sum[col]) {
            debug_printf("DEBUG: INVALID - row or col already full\n");
            env->terminals[0] = 1;
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            add_log(env);
            c_reset(env);
            return;
        }

        // Check if marking this cell BLACK would create a run longer than max allowed
        int row_run = get_row_run_length(env, row, col);
        debug_printf("DEBUG: row_run_length=%d, rows_max_clue[%d]=%d\n",
               row_run, row, env->rows_max_clue[row]);

        if (row_run > env->rows_max_clue[row]) {
            debug_printf("DEBUG: INVALID - row run too long\n");
            env->terminals[0] = 1;
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            add_log(env);
            c_reset(env);
            return;
        }

        int col_run = get_col_run_length(env, row, col);
        debug_printf("DEBUG: col_run_length=%d, cols_max_clue[%d]=%d\n",
               col_run, col, env->cols_max_clue[col]);

        if (col_run > env->cols_max_clue[col]) {
            debug_printf("DEBUG: INVALID - col run too long\n");
            env->terminals[0] = 1;
            env->rewards[0] = REWARD_INVALID_MOVE;
            env->episode_reward += REWARD_INVALID_MOVE;
            add_log(env);
            c_reset(env);
            return;
        }

        // Second check: if completing row/col, check runs match
        int row_completed = 0;
        int col_completed = 0;

        debug_printf("DEBUG: Checking line completion...\n");

        if (env->rows_totals[row] == env->rows_target_sum[row] - 1) {
            debug_printf("DEBUG: Would complete row %d, checking pattern...\n", row);
            // Temporarily mark BLACK to check
            env->observations[pos] = CELL_BLACK;
            int row_start = row * MAX_SIZE;
            int matches = check_line_matches(env->observations + row_start,
                                   env->rows_clues + row * MAX_CLUES,
                                   env->rows_num_runs[row], env->size);
            debug_printf("DEBUG: Row pattern matches: %d\n", matches);
            if (!matches) {
                // Runs don't match - invalid move (terminate episode)
                debug_printf("DEBUG: INVALID - row pattern doesn't match\n");
                env->observations[pos] = CELL_EMPTY;
                env->terminals[0] = 1;
                env->rewards[0] = REWARD_NO_MATCH;
                env->episode_reward += REWARD_NO_MATCH;
                add_log(env);
                c_reset(env);
                return;
            }
            env->observations[pos] = CELL_EMPTY;
            row_completed = 1;
        }

        if (env->cols_totals[col] == env->cols_target_sum[col] - 1) {
            debug_printf("DEBUG: Would complete col %d, checking pattern...\n", col);
            // Temporarily mark BLACK to check
            env->observations[pos] = CELL_BLACK;
            unsigned char col_data[MAX_SIZE];
            for (int i = 0; i < env->size; i++) {
                col_data[i] = env->observations[i * MAX_SIZE + col];
            }
            int matches = check_line_matches(col_data,
                                   env->cols_clues + col * MAX_CLUES,
                                   env->cols_num_runs[col], env->size);
            debug_printf("DEBUG: Col pattern matches: %d\n", matches);
            if (!matches) {
                // Runs don't match - invalid move (terminate episode)
                debug_printf("DEBUG: INVALID - col pattern doesn't match\n");
                env->observations[pos] = CELL_EMPTY;
                env->terminals[0] = 1;
                env->rewards[0] = REWARD_NO_MATCH;
                env->episode_reward += REWARD_NO_MATCH;
                add_log(env);
                c_reset(env);
                return;
            }
            env->observations[pos] = CELL_EMPTY;
            col_completed = 1;
        }

        // Apply mark BLACK
        env->observations[pos] = CELL_BLACK;
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
        // Marking WHITE - always valid (just marks empty as not-black)
        env->observations[pos] = CELL_WHITE;
    }

    // Easy learn mode: check if cell matches solution
    if (env->easy_learn) {
        unsigned char solution_cell = env->solution[pos];
        unsigned char actual = env->observations[pos];

        if (solution_cell == actual) {
            // Correct move: give positive reward and continue
            env->rewards[0] += REWARD_EASY_LEARN_CORRECT;
            env->episode_reward += REWARD_EASY_LEARN_CORRECT;
        } else {
            // Incorrect move: give negative reward, terminate and reset
            env->rewards[0] += REWARD_EASY_LEARN_INCORRECT;
            env->episode_reward += REWARD_EASY_LEARN_INCORRECT;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            return;
        }
    }

    // Check if solved (filled_total == target_total means all BLACK cells placed correctly)
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
        int board_width = 120 + MAX_SIZE * 40;
        int board_height = 120 + MAX_SIZE * 40;
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

    // Draw titles
    DrawText("CURRENT BOARD", 20, 20, 24, RAYWHITE);
    int solution_x = clue_area + env->size * cell_size + board_spacing + 20;
    DrawText("SOLUTION", solution_x, 20, 24, RAYWHITE);

    // Draw current board
    int offset_x = 20;
    int offset_y = 60;

    // Draw column clues for current board
    for (int clue_row = 0; clue_row < MAX_CLUES; clue_row++) {
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
        for (int clue_idx = 0; clue_idx < MAX_CLUES; clue_idx++) {
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
            int pos = r * MAX_SIZE + c;

            if (env->observations[pos] == CELL_BLACK) {
                DrawRectangle(x, y, cell_size, cell_size, (Color){50, 50, 50, 255});  // Dark gray for BLACK
            } else if (env->observations[pos] == CELL_WHITE) {
                DrawRectangle(x, y, cell_size, cell_size, (Color){240, 240, 240, 255});  // Light gray for WHITE
            } else {
                DrawRectangle(x, y, cell_size, cell_size, (Color){120, 120, 120, 255});  // Medium gray for EMPTY
            }
            DrawRectangleLines(x, y, cell_size, cell_size, LIGHTGRAY);
        }
    }

    // Draw solution board
    offset_x = solution_x;

    // Draw column clues for solution
    for (int clue_row = 0; clue_row < MAX_CLUES; clue_row++) {
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
        for (int clue_idx = 0; clue_idx < MAX_CLUES; clue_idx++) {
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
            int pos = r * MAX_SIZE + c;

            if (env->solution[pos] == CELL_BLACK) {
                DrawRectangle(x, y, cell_size, cell_size, GREEN);
            } else {
                DrawRectangle(x, y, cell_size, cell_size, (Color){200, 200, 200, 255});
            }
            DrawRectangleLines(x, y, cell_size, cell_size, LIGHTGRAY);
        }
    }

    // Draw status
    int board_height = clue_area + env->size * cell_size;
    int status_y = board_height + 80;
    char status[128];
    snprintf(status, sizeof(status), "Steps: %d/%d | Filled: %d/%d | Size: %dx%d",
             env->steps_taken, env->max_steps, env->filled_total, env->target_total, env->size, env->size);
    DrawText(status, 20, status_y, 20, RAYWHITE);

    // Draw reward info
    char reward_info[128];
    snprintf(reward_info, sizeof(reward_info), "Last Reward: %.3f | Episode Return: %.3f",
             env->rewards[0], env->episode_reward);
    DrawText(reward_info, 20, status_y + 25, 20, RAYWHITE);

    // Draw instructions
    DrawText("Click cells to toggle | Press R to reset | ESC to quit", 20, status_y + 60, 16, LIGHTGRAY);

    EndDrawing();
}

void c_close(Nonogram* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
