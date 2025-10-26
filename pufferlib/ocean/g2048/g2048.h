#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define SIZE 4
#define EMPTY 0
#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4

// Precomputed constants
#define REWARD_MULTIPLIER 0.09090909f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required
    int score;
    int tick;
    unsigned char grid[SIZE][SIZE];
    float episode_reward;           // Accumulate episode reward
    
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

static Color tile_colors[12] = {
    {6, 24, 24, 255}, // Empty/background
    {187, 187, 187, 255}, // 2
    {170, 187, 187, 255}, // 4
    {150, 187, 187, 255}, // 8
    {130, 187, 187, 255},  // 16
    {110, 187, 187, 255},  // 32
    {90, 187, 187, 255},   // 64
    {70, 187, 187, 255}, // 128
    {50, 187, 187, 255},  // 256
    {30, 187, 187, 255},  // 512
    {10, 187, 187, 255},  // 1024
    {0, 187, 187, 255}   // 2048+
};

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* env);
void c_step(Game* env);
void c_render(Game* env);
void c_close(Game* env);

// Inline function for updating observations (avoid function call overhead)
static inline void update_observations(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->observations[i * SIZE + j] = game->grid[i][j];
        }
    }
}

// Cache empty cell count during grid operations
static inline void update_empty_count(Game* game) {
    int count = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] == EMPTY) count++;
        }
    }
    game->empty_count = count;
}

void add_log(Game* game) {
    game->log.score = (float)(1 << game->score);
    game->log.perf += ((float)game->score) * REWARD_MULTIPLIER;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->grid[i][j] = EMPTY;
        }
    }

    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;
    game->empty_count = SIZE * SIZE;
    game->game_over_cached = false;
    game->grid_changed = true;
    
    if (game->terminals) game->terminals[0] = 0;
    
    // Add two random tiles at the start - optimized version
    for (int added = 0; added < 2; ) {
        int pos = rand() % (SIZE * SIZE);
        int i = pos / SIZE;
        int j = pos % SIZE;
        if (game->grid[i][j] == EMPTY) {
            game->grid[i][j] = (rand() % 10 == 0) ? 2 : 1;
            added++;
            game->empty_count--;
        }
    }
    
    update_observations(game);
}

void add_random_tile(Game* game) {
    if (game->empty_count == 0) return;
    
    // Use reservoir sampling for better performance
    int chosen_pos = -1;
    int count = 0;
    
    for (int pos = 0; pos < SIZE * SIZE; pos++) {
        int i = pos / SIZE;
        int j = pos % SIZE;
        if (game->grid[i][j] == EMPTY) {
            count++;
            if (rand() % count == 0) {
                chosen_pos = pos;
            }
        }
    }
    
    if (chosen_pos >= 0) {
        int i = chosen_pos / SIZE;
        int j = chosen_pos % SIZE;
        game->grid[i][j] = (rand() % 10 == 0) ? 2 : 1;
        game->empty_count--;
        game->grid_changed = true;
    }
    
    update_observations(game);
}

// Optimized slide and merge with fewer memory operations
static inline bool slide_and_merge(unsigned char* row, float* reward) {
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
            *reward += ((float)row[i]) * REWARD_MULTIPLIER;
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

bool move(Game* game, int direction, float* reward) {
    bool moved = false;
    unsigned char temp[SIZE];
    
    if (direction == UP || direction == DOWN) {
        for (int col = 0; col < SIZE; col++) {
            // Extract column
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == UP) ? i : SIZE - 1 - i;
                temp[i] = game->grid[idx][col];
            }
            
            if (slide_and_merge(temp, reward)) {
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
            
            if (slide_and_merge(temp, reward)) {
                moved = true;
                // Write back row
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                    game->grid[row][idx] = temp[i];
                }
            }
        }
    }

    if (!moved) {
        *reward = INVALID_MOVE_PENALTY;
    } else {
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

// Optimized score calculation
static inline unsigned char calc_score(Game* game) {
    unsigned char max_tile = 0;
    // Unroll loop for better performance
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] > max_tile) {
                max_tile = game->grid[i][j];
            }
        }
    }
    return max_tile;
}

void c_step(Game* game) {
    float reward = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward);
    game->tick++;
    
    if (did_move) {
        add_random_tile(game);
        game->score = calc_score(game);
        update_empty_count(game); // Update after adding tile
    }
    
    bool game_over = is_game_over(game);
    game->terminals[0] = game_over ? 1 : 0;
    
    if (game_over) {
        reward = GAME_OVER_PENALTY;
    }
    
    game->rewards[0] = reward;
    game->episode_reward += reward;

    update_observations(game);

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
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
            Color color = (val == 0) ? tile_colors[0] : 
                         (val <= 11) ? tile_colors[val] : 
                         (Color){60, 60, 60, 255};
            
            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            
            if (val > 0) {
                int display_val = 1 << val; // Power of 2
                // Pre-format text to avoid repeated formatting
                snprintf(score_text, sizeof(score_text), "%d", display_val);
                if (display_val < 1000) {
                    DrawText(score_text, j * px + 30, i * px + 40, 32, PUFF_WHITE);
                } else {
                    DrawText(score_text, j * px + 20, i * px + 40, 32, PUFF_WHITE);
                }
            }
        }
    }
    
    // Draw score (format once per frame)
    snprintf(score_text, sizeof(score_text), "Score: %d", 1 << game->score);
    DrawText(score_text, 10, px * SIZE + 10, 24, PUFF_WHITE);
    
    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
