#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }

#define SIZE 4
#define EMPTY 0
#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4
#define BASE_MAX_TICKS 1000

// These work well
#define MERGE_REWARD_WEIGHT 0.0625f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f

// These may need experimenting, but work for now
#define STATE_REWARD_WEIGHT 0.01f // Fixed, small reward for maintaining "desirable" states
#define MONOTONICITY_REWARD_WEIGHT 0.00003f

// Features: 18 per cell
// 1. Normalized tile value (current_val / max_val)
// 2. One-hot for empty (1 if empty, 0 if occupied)
// 3-18. One-hot for tile values 2^1 to 2^16 (16 features)
#define NUM_FEATURES 18

static inline float calculate_perf(unsigned char max_tile) {
    // Reaching 65k -> 1.0, 32k -> 0.8, 16k -> 0.4, 8k -> 0.2, 4k -> 0.1, 2k -> 0.05
    float perf = 0.8f * (float)(1 << max_tile) / 32768.0f;
    if (perf > 1.0f) perf = 1.0f;
    return perf;
}

typedef struct {
    float perf;
    float score;
    float merge_score;
    float episode_return;
    float episode_length;
    float lifetime_max_tile;
    float reached_32768;
    float reached_65536;
    float snake_state;
    float monotonicity_reward;
    float snake_reward;
    float n;
} Log;

typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required

    bool can_go_over_65536;         // Set false for training, true for eval
    float reward_scaler;            // Pufferlib clips rew from -1 to 1, adjust the resulting rew accordingly

    float endgame_env_prob;         // The prob of env being initialized as an endgame-only env
    bool is_endgame_env;
    float scaffolding_ratio;        // The ratio for "scaffolding" runs, in which higher blocks are spawned
    bool is_scaffolding_episode;
    bool use_heuristic_rewards;
    float snake_reward_weight;
    bool use_sparse_reward;         // Ignore all rewards and provide 1 for reaching 16k, 32k, 65k

    int score;
    int tick;
    unsigned char grid[SIZE][SIZE];
    unsigned char lifetime_max_tile;
    unsigned char max_tile;         // Episode max tile
    float episode_reward;           // Accumulate episode reward
    float monotonicity_reward;
    float snake_reward;
    int moves_made;
    int max_episode_ticks;          // Dynamic max_ticks based on score
    bool is_snake_state;
    int snake_state_tick;
    bool stop_at_65536;

    // Cached values to avoid recomputation
    int empty_count;
    bool game_over_cached;
    bool grid_changed;
} Game;

// Precomputed color table for rendering optimization
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};

static Color tile_colors[17] = {
    {6, 24, 24, 255}, // Empty/background
    {187, 187, 187, 255}, // 2
    {170, 187, 187, 255}, // 4
    {150, 187, 187, 255}, // 8
    {130, 187, 187, 255},  // 16
    {110, 187, 187, 255},  // 32
    {90, 187, 187, 255},   // 64 (Getting more cyan)
    {70, 187, 187, 255},   // 128
    {50, 187, 187, 255},   // 256
    {30, 187, 187, 255},   // 512
    {0, 187, 187, 255},    // 1024 (PUFF_CYAN)
    {0, 150, 187, 255},    // 2048
    {0, 110, 187, 255},    // 4096
    {0, 70, 187, 255},     // 8192
    {187, 0, 0, 255},      // 16384 (PUFF_RED)
    {204, 173, 17, 255},   // 32768 (Gold)
    {6, 24, 24, 255},      // 65536+ (Invisible)
};

// Precomputed pow(x, 1.5) lookup table for x in [0, 19] to avoid expensive pow() calls.
static const unsigned char pow_1_5_lookup[20] = {
    0, 1, 2, 5, 8, 11, 14, 18, 22, 27, 31, 36, 41, 46, 52, 57, 64, 69, 75, 81
};

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* game);
void c_step(Game* game);
void c_render(Game* game);
void c_close(Game* game);

void init(Game* game) {
    game->lifetime_max_tile = 0;
    game->is_endgame_env = (rand() / (float)RAND_MAX) < game->endgame_env_prob;
}

void update_observations(Game* game) {
    // Observation: 4x4 grid, 18 features per cell
    // 1. Normalized tile value (current_val / max_val)
    // 2. One-hot for empty (1 if empty, 0 if occupied)
    // 3. One-hot for tile values 2^1 to 2^16 (16 features)
    // 4. Additional obs: is_snake_state (1)

    int num_cell = SIZE * SIZE;
    int num_additional_obs = 1;
    memset(game->observations, 0, (num_cell * NUM_FEATURES + num_additional_obs) * sizeof(unsigned char));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int feat1_idx = (i * SIZE + j);
            int feat2_idx = num_cell + feat1_idx;
            int feat3_idx = 2 * num_cell + 16 * feat1_idx;
            unsigned char grid_val = game->grid[i][j];

            // Feature 1: The original tile values ** 1.5, to make a bit superlinear within uint8
            game->observations[feat1_idx] = pow_1_5_lookup[grid_val];

            // Feature 2: One-hot for empty
            game->observations[feat2_idx] = (grid_val == EMPTY) ? 1 : 0;

            // Features 3-18: One-hot for tile values
            // NOTE: If this ever gets close to 131072, revisit this
            if (grid_val > 0) {
                grid_val = min(grid_val, 16);
                game->observations[feat3_idx + grid_val - 1] = 1;
            }
        }
    }
    // Additional obs
    int offset = num_cell * NUM_FEATURES;
    game->observations[offset] = game->is_snake_state;
}

void add_log(Game* game) {
    // Scaffolding runs will distort stats, so skip logging
    if (game->is_endgame_env || game->is_scaffolding_episode) return;

    // Update the lifetime best
    if (game->max_tile > game->lifetime_max_tile) {
        game->lifetime_max_tile = game->max_tile;
    }
    
    game->log.score += (float)(1 << game->max_tile);
    game->log.perf += calculate_perf(game->max_tile);
    game->log.merge_score += (float)game->score;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.lifetime_max_tile += (float)(1 << game->lifetime_max_tile);
    game->log.reached_32768 += (game->max_tile >= 15);
    game->log.reached_65536 += (game->max_tile >= 16);
    game->log.snake_state += (float)game->snake_state_tick / (float)game->tick;
    game->log.monotonicity_reward += game->monotonicity_reward * MONOTONICITY_REWARD_WEIGHT * game->reward_scaler;
    game->log.snake_reward += game->snake_reward * game->snake_reward_weight * game->reward_scaler;
    game->log.n += 1;
}

static inline unsigned char get_new_tile(void) {
    // 10% chance of 2, 90% chance of 1
    return (rand() % 10 == 0) ? 2 : 1;
}

static inline void place_tile_at_random_cell(Game* game, unsigned char tile) {
    if (game->empty_count == 0) return;

    int target = rand() % game->empty_count;
    int pos = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] == EMPTY) {
                if (pos == target) {
                    game->grid[i][j] = tile;
                    game->empty_count--;
                    return;
                }
                pos++;
            }
        }
    }
}

void set_scaffolding_curriculum(Game* game) {
    game->stop_at_65536 = true;

    if (game->lifetime_max_tile < 14) {
        int curriculum = rand() % 5;

        // Spawn one high tiles from 8192, 16384, 32768, 65536
        unsigned char high_tile = max(12 + curriculum, game->lifetime_max_tile);
        place_tile_at_random_cell(game, high_tile);
        if (high_tile >= 16) game->stop_at_65536 = false;

    } else {
        int curriculum = rand() % 8;

        if (curriculum < 2) { // curriculum 0, 1
            place_tile_at_random_cell(game, 14 + curriculum); // Spawn one of 16384 or 32768

        } else if (curriculum == 2) {
            // Place the tiles in the second row, so that they can be moved up in the first move
            unsigned char tiles[] = {14, 13};
            memcpy(game->grid[1], tiles, 2);
            game->empty_count -= 2;
        } else if (curriculum == 3) {  // harder
            game->grid[1][0] = 14; game->empty_count--;
            place_tile_at_random_cell(game, 13);

        } else if (curriculum == 4) {
            unsigned char tiles[] = {15, 14};
            memcpy(game->grid[1], tiles, 2);
            game->empty_count -= 2;
        } else if (curriculum == 5) {  // harder
            game->grid[1][0] = 15; game->empty_count--;
            place_tile_at_random_cell(game, 14);

        } else if (curriculum == 6) {
            unsigned char tiles[] = {15, 14, 13};
            memcpy(game->grid[1], tiles, 3);
            game->empty_count -= 3;
        } else if (curriculum == 7) {  // harder
            game->grid[1][0] = 15; game->empty_count--;
            place_tile_at_random_cell(game, 14);
            place_tile_at_random_cell(game, 13);
        }
    }
}

void set_endgame_curriculum(Game* game) {
    game->stop_at_65536 = true;
    int curriculum = rand() % 4;

    // Place the tiles in the second-third rows, so that they can be moved up in the first move
    unsigned char tiles[] = {15, 14, 13, 12};
    memcpy(game->grid[1], tiles, 4);
    game->empty_count -= 4;

    if (curriculum >= 1) { game->grid[2][3] = 11; game->empty_count--; }
    if (curriculum >= 2) { 
        game->grid[2][2] = 10;
        game->grid[2][1] = 9;
        game->grid[2][0] = 8;
        game->empty_count -= 3;
    }
}

void c_reset(Game* game) {
    memset(game->grid, EMPTY, SIZE * SIZE);
    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;
    game->empty_count = SIZE * SIZE;
    game->game_over_cached = false;
    game->grid_changed = true;
    game->moves_made = 0;
    game->max_episode_ticks = BASE_MAX_TICKS;
    game->max_tile = 0;
    game->snake_state_tick = 0;
    game->monotonicity_reward = 0;
    game->snake_reward = 0;
    game->is_snake_state = false;
    game->stop_at_65536 = game->can_go_over_65536;

    if (game->terminals) game->terminals[0] = 0;

    // End game envs only do endgame curriculum
    if (game->is_endgame_env) {
        set_endgame_curriculum(game);

    } else {
        // Higher tiles are spawned in scaffolding episodes
        // Having high tiles saves moves to get there, allowing agents to experience it faster
        game->is_scaffolding_episode = (rand() / (float)RAND_MAX) < game->scaffolding_ratio;
        if (game->is_scaffolding_episode) {
            set_scaffolding_curriculum(game);

        } else {
            // Add two random tiles at the start
            for (int i = 0; i < 2; i++) {
                place_tile_at_random_cell(game, get_new_tile());
            }
        }
    }

    update_observations(game);
}

// Optimized slide and merge with fewer memory operations
static inline bool slide_and_merge(Game* game, unsigned char* row, float* reward, float* score_increase) {
    bool moved = false;
    int write_pos = 0;
    
    // Single pass: slide and identify merge candidates
    for (int read_pos = 0; read_pos < SIZE; read_pos++) {
        if (row[read_pos] != EMPTY) {
            if (write_pos != read_pos) {
                row[write_pos] = row[read_pos];
                row[read_pos] = EMPTY;
                moved = true;
            }
            write_pos++;
        }
    }
    
    // Merge pass
    for (int i = 0; i < SIZE - 1; i++) {
        if (row[i] != EMPTY && row[i] == row[i + 1]) {
            row[i]++;
            *reward += ((float)row[i]) * MERGE_REWARD_WEIGHT;
            *score_increase += (float)(1 << (int)row[i]);
            // Shift remaining elements left
            for (int j = i + 1; j < SIZE - 1; j++) {
                row[j] = row[j + 1];
            }
            row[SIZE - 1] = EMPTY;
            moved = true;
        }
    }
    
    return moved;
}

bool move(Game* game, int direction, float* reward, float* score_increase) {
    bool moved = false;
    unsigned char temp[SIZE];
    
    if (direction == UP || direction == DOWN) {
        for (int col = 0; col < SIZE; col++) {
            // Extract column
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == UP) ? i : SIZE - 1 - i;
                temp[i] = game->grid[idx][col];
            }
            
            if (slide_and_merge(game, temp, reward, score_increase)) {
                moved = true;
                // Write back column
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == UP) ? i : SIZE - 1 - i;
                    game->grid[idx][col] = temp[i];
                }
            }
        }
    } else {
        for (int row = 0; row < SIZE; row++) {
            // Extract row
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                temp[i] = game->grid[row][idx];
            }
            
            if (slide_and_merge(game, temp, reward, score_increase)) {
                moved = true;
                // Write back row
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                    game->grid[row][idx] = temp[i];
                }
            }
        }
    }

    if (moved) {
        game->grid_changed = true;
        game->game_over_cached = false; // Invalidate cache
    }

    return moved;
}

bool is_game_over(Game* game) {
    // Use cached result if grid hasn't changed
    if (!game->grid_changed && game->game_over_cached) {
        return game->game_over_cached;
    }
    
    // Quick check: if there are empty cells, game is not over
    if (game->empty_count > 0) {
        game->game_over_cached = false;
        game->grid_changed = false;
        return false;
    }
    
    // Check for possible merges
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char current = game->grid[i][j];
            if (i < SIZE - 1 && current == game->grid[i + 1][j]) {
                game->game_over_cached = false;
                game->grid_changed = false;
                return false;
            }
            if (j < SIZE - 1 && current == game->grid[i][j + 1]) {
                game->game_over_cached = false;
                game->grid_changed = false;
                return false;
            }
        }
    }
    
    game->game_over_cached = true;
    game->grid_changed = false;
    return true;
}

// Combined grid stats and heuristic calculation for performance
float update_stats_and_get_heuristic_rewards(Game* game) {
    int empty_count = 0;
    int top_row_count = 0;
    unsigned char max_tile = 0;
    unsigned char second_max_tile = 0;
    unsigned char max_tile_in_row234 = 0;
    float heuristic_state_reward = 0.0f;
    float monotonicity_reward = 0.0f;
    float snake_reward = 0.0f;
    game->is_snake_state = false;
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char val = game->grid[i][j];
            
            // Update empty count and max tile
            if (val == EMPTY) empty_count++;

            // Count filled cells in the top row
            if (i == 0 && val != EMPTY) top_row_count++;
            
            // Allow max and the second max tile to be the same
            if (val >= max_tile) {
                second_max_tile = max_tile;
                max_tile = val;
            } else if (val > second_max_tile && val < max_tile) {
                second_max_tile = val;
            }

            // Get the max tile in the 2nd, 3rd, 4th row
            if (i > 0 && val > max_tile_in_row234) max_tile_in_row234 = val;
        }
    }

    game->empty_count = empty_count;
    game->max_tile = max_tile;

    /* Heuristic rewards */

    // Filled top row reward: A simple nudge to keep the top row filled
    if (top_row_count == SIZE) heuristic_state_reward += STATE_REWARD_WEIGHT;

    bool max_in_top_left = (game->grid[0][0] == max_tile);

    // Corner reward: A simple nudge to keep the max tiles horizontally in the top row, left corner.
    // When agents learn to put the max tile on the other corners, or put max tiles vertically
    // they miss out snake rew, and this does happen sometimes.
    if (max_in_top_left && game->grid[0][1] == second_max_tile && max_tile > 4) {
        heuristic_state_reward += STATE_REWARD_WEIGHT;
    }

    // Snake reward: look for the snake pattern, only when the max tile is at top left
    if (max_in_top_left) {
        monotonicity_reward += pow_1_5_lookup[max_tile];
        int evidence_for_snake = 0;

        for (int i = 0; i < 2; i++) {
            unsigned char row_min = 32;
            unsigned char next_row_max = 0;
            for (int j = 0; j < SIZE; j++) {
                unsigned char val = game->grid[i][j];

                // Check horizontal monotonicity (snake pattern) for top two rows only
                if (j < SIZE - 1) {
                    unsigned char next_col = game->grid[i][j+1];
                    if (val != EMPTY && next_col != EMPTY) {
                        // Row 0: Reward decreasing left to right, e.g., 12-11-10-9
                        if (i == 0 && val > next_col) {
                            monotonicity_reward += pow_1_5_lookup[next_col];
                            evidence_for_snake++;
                        }
                        // Row 1: Reward increasing left to right, e.g., 5-6-7-8
                        else if (i == 1 && val < next_col) {
                            monotonicity_reward += pow_1_5_lookup[val];
                        }
                    }
                }

                // Vertical monotonicity: give score after row scanning for min/max is done
                if (val != EMPTY && val < row_min) row_min = val;
                unsigned char next_row = game->grid[i+1][j];
                if (next_row != EMPTY && next_row > next_row_max) next_row_max = next_row;
                // // Small column-level vertical reward
                if (val != EMPTY && next_row != EMPTY && val > next_row) monotonicity_reward += next_row;
            }
            // Large row-level vertical reward
            if (i < 2 && row_min < 20 && next_row_max > 0 && row_min > next_row_max) {
                monotonicity_reward += 4 * pow_1_5_lookup[row_min];
                if (i == 0) evidence_for_snake++;
            }
        }

        // Snake bonus: sorted top row + the max_tile_in_row234 in the second row right
        // For example, top row: 14-13-12-11, second row: ()-()-()-10
        unsigned char snake_tail = game->grid[1][3];
        if (evidence_for_snake >= 4 && snake_tail == max_tile_in_row234) {
            game->is_snake_state = true;
            game->snake_state_tick++;
            snake_reward = snake_tail * snake_tail;
        }
    }

    // Trained models need game->is_snake_state as obs
    if (!game->use_heuristic_rewards) return 0.0f;

    game->monotonicity_reward += monotonicity_reward;
    game->snake_reward += snake_reward;
    
    return heuristic_state_reward + monotonicity_reward * MONOTONICITY_REWARD_WEIGHT + snake_reward * game->snake_reward_weight;
}

void c_step(Game* game) {
    float reward = 0.0f;
    float score_add = 0.0f;
    unsigned char prev_max_tile = game->max_tile;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        place_tile_at_random_cell(game, get_new_tile());
        game->score += score_add;

        // Add heuristic rewards/penalties and update grid stats
        reward += update_stats_and_get_heuristic_rewards(game);
        reward *= game->reward_scaler;

        update_observations(game); // Observations only change if the grid changes
        
        // This is to limit infinite invalid moves during eval (happens for noob agents)
        // Don't need to be tight. Don't need to show to human player.
        int tick_multiplier = max(1, game->lifetime_max_tile - 8); // practically no limit for competent agent
        game->max_episode_ticks = max(BASE_MAX_TICKS * tick_multiplier, game->score / 4);

    } else {
        reward = INVALID_MOVE_PENALTY;
        // No need to update observations if the grid hasn't changed
    }

    bool game_over = is_game_over(game);
    bool max_ticks_reached = game->tick >= game->max_episode_ticks;
    bool max_level_reached = game->stop_at_65536 && game->max_tile >= 16;
    game->terminals[0] = (game_over || max_ticks_reached || max_level_reached) ? 1 : 0;

    // Game over penalty overrides other rewards
    if (game_over) {
        reward = GAME_OVER_PENALTY;
    }

    if (game->use_sparse_reward) {
        reward = 0; // Ignore all previous reward
        if (game->max_tile >= 14 && game->max_tile > prev_max_tile) reward = 1;
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
}

// Stepping for client/eval: no reward, no reset
void step_without_reset(Game* game) {
    float score_add = 0.0f;
    float reward = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        place_tile_at_random_cell(game, get_new_tile());
        game->score += score_add;
        update_stats_and_get_heuristic_rewards(game); // The reward is ignored.
        update_observations(game); // Observations only change if the grid changes
    }

    bool game_over = is_game_over(game);
    game->terminals[0] = (game_over) ? 1 : 0;
}

// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;
    static char score_text[32];
    static const int px = 100;
    
    if (!window_initialized) {
        InitWindow(px * SIZE, px * SIZE + 50, "2048");
        SetTargetFPS(30); // Increased for smoother rendering
        window_initialized = true;
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int val = game->grid[i][j];
            
            // Use precomputed colors
            int color_idx = min(val, 16); // Cap at the max index of our color array
            Color color = tile_colors[color_idx];
            
            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            
            if (val > 0) {
                int display_val = 1 << val; // Power of 2
                // Pre-format text to avoid repeated formatting
                snprintf(score_text, sizeof(score_text), "%d", display_val);

                int font_size = 32;
                int x_offset = 20; // Default for 4-digit numbers
                if (display_val < 10) x_offset = 40; // 1-digit
                else if (display_val < 100) x_offset = 35; // 2-digit
                else if (display_val < 1000) x_offset = 25; // 3-digit
                else if (display_val < 10000) x_offset = 15; // 4-digit
                else if (display_val < 100000) x_offset = 2; // 5-digit
                else {
                    font_size = 24;
                    x_offset = 5;
                }

                DrawText(score_text, j * px + x_offset, i * px + 34, font_size, PUFF_WHITE);
            }
        }
    }
    
    // Draw score (format once per frame)
    snprintf(score_text, sizeof(score_text), "Score: %d", game->score);
    DrawText(score_text, 10, px * SIZE + 10, 24, PUFF_WHITE);

    snprintf(score_text, sizeof(score_text), "Moves: %d", game->moves_made);
    DrawText(score_text, 210, px * SIZE + 10, 24, PUFF_WHITE);
    
    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
