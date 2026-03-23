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

// Reward constants
#define MERGE_BASE_REWARD 0.05f
#define MERGE_REWARD_SCALE 0.03f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f

// Pow 1.5 lookup table for tiles 128+ (index = row[i] - 6)
// Index: 1=128, 2=256, 3=512, 4=1024, 5=2048, 6=4096, 7=8192, 8=16384, 9=32768, 10=65536, 11=131k
static const float pow15_table[12] = {
    0.0f, 1.0f, 2.83f, 5.20f, 8.0f, 11.18f, 14.70f, 18.52f, 22.63f, 27.0f, 31.62f, 36.48f,
};

static inline float calculate_perf(unsigned char max_tile) {
    // Reaching 131k -> 1.0, 65k -> 0.8, 32k -> 0.4, 16k -> 0.2, 8k -> 0.1
    float perf = 0.8f * (float)(1 << max_tile) / 65536.0f;
    if (perf > 1.0f) perf = 1.0f;
    return perf;
}

typedef struct Log {
    float perf;
    float score;
    float merge_score;
    float episode_return;
    float episode_length;
    float lifetime_max_tile;
    float reached_16384;
    float reached_32768;
    float reached_65536;
    float reached_131072;
    float n;
} Log;

typedef struct Game {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    double* actions;                // Required
    float* rewards;                 // Required
    float* terminals;               // Required
    int num_agents;                 // Required for env_binding

    float scaffolding_ratio;        // The ratio for "scaffolding" runs, in which higher blocks are spawned
    bool is_scaffolding_episode;

    int score;
    int tick;
    unsigned char grid[SIZE][SIZE];
    unsigned char lifetime_max_tile;
    unsigned char max_tile;         // Episode max tile
    float episode_reward;           // Accumulate episode reward
    int moves_made;
    int max_episode_ticks;          // Dynamic max_ticks based on score

    // Cached values to avoid recomputation
    int empty_count;
    bool game_over_cached;
    bool grid_changed;
    unsigned int rng;
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

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* game);
void c_step(Game* game);
void c_render(Game* game);
void c_close(Game* game);

void init(Game* game) {
    game->lifetime_max_tile = 0;
    memset(game->grid, EMPTY, SIZE * SIZE);    
}

void update_observations(Game* game) {
    memcpy(game->observations, game->grid, SIZE * SIZE);
}

void add_log(Game* game) {
    // Scaffolding runs will distort stats, so skip logging
    if (game->is_scaffolding_episode) return;

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
    game->log.reached_16384 += (game->max_tile >= 14);
    game->log.reached_32768 += (game->max_tile >= 15);
    game->log.reached_65536 += (game->max_tile >= 16);
    game->log.reached_131072 += (game->max_tile >= 17);
    game->log.n += 1;
}

static inline unsigned char get_new_tile(Game* game) {
    // 10% chance of 2, 90% chance of 1
    return (rand_r(&game->rng) % 10 == 0) ? 2 : 1;
}

static inline void place_tile_at_random_cell(Game* game, unsigned char tile) {
    if (game->empty_count == 0) return;

    int target = rand_r(&game->rng) % game->empty_count;
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
    if (game->lifetime_max_tile < 14) {
        // Spawn one high tile from 8192 to 65536
        int curriculum = rand_r(&game->rng) % 5;
        unsigned char high_tile = max(12 + curriculum, game->lifetime_max_tile);
        place_tile_at_random_cell(game, high_tile);

    } else {
        // base=14 until 65536 reached, then base=15 for 131072 practice
        // All random placement, 1-2 tiles max
        unsigned char base = (game->lifetime_max_tile >= 16) ? 15 : 14;
        int curriculum = rand_r(&game->rng) % 4;

        if (curriculum == 0) {
            place_tile_at_random_cell(game, base);
        } else if (curriculum == 1) {
            place_tile_at_random_cell(game, base + 1);
        } else if (curriculum == 2) {
            place_tile_at_random_cell(game, base);
            place_tile_at_random_cell(game, base - 1);
        } else {
            place_tile_at_random_cell(game, base + 1);
            place_tile_at_random_cell(game, base);
        }
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

    // Higher tiles are spawned in scaffolding episodes
    // Having high tiles saves moves to get there, allowing agents to experience it faster
    game->is_scaffolding_episode = (rand_r(&game->rng) / (float)RAND_MAX) < game->scaffolding_ratio;
    if (game->is_scaffolding_episode) {
        set_scaffolding_curriculum(game);

    } else {
        // Add two random tiles at the start
        for (int i = 0; i < 2; i++) {
            place_tile_at_random_cell(game, get_new_tile(game));
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
            // Tiles 2-64 (row[i] 1-6): base reward only
            // Tiles 128+ (row[i] 7+): base + pow1.5 scaled bonus
            if (row[i] <= 6) {
                *reward += MERGE_BASE_REWARD;
            } else {
                *reward += MERGE_BASE_REWARD + pow15_table[row[i] - 6] * MERGE_REWARD_SCALE;
            }
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
    if (!game->grid_changed) {
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

void update_stats(Game* game) {
    int empty_count = 0;
    unsigned char max_tile = 0;
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char val = game->grid[i][j];
            // Update empty count and max tile
            if (val == EMPTY) empty_count++;
            if (val > max_tile) {
                max_tile = val;
            }
        }
    }

    game->empty_count = empty_count;
    game->max_tile = max_tile;
}

void c_step(Game* game) {
    float reward = 0.0f;
    float score_add = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        // Refresh empty_count after merges so spawning uses the correct count.
        update_stats(game);
        place_tile_at_random_cell(game, get_new_tile(game));
        game->score += score_add;

        // Observations only change if the grid changes
        update_observations(game);
        
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
    game->terminals[0] = (game_over || max_ticks_reached) ? 1 : 0;

    // Game over penalty overrides other rewards
    if (game_over) {
        reward += GAME_OVER_PENALTY;
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

        // Refresh empty_count after merges so spawning uses the correct count.
        update_stats(game);
        place_tile_at_random_cell(game, get_new_tile(game));
        game->score += score_add;

        // Observations only change if the grid changes
        update_observations(game);
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
