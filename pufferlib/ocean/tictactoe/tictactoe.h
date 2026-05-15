/* TicTacToe: Single-agent tic-tac-toe against a random opponent */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char ENEMY = 2;

// Required struct. Only use floats!
typedef struct {
    float perf;           // Win rate (1.0 for win, 0.0 for loss/draw)
    float score;          // Final reward of episode
    float episode_return; // Sum of rewards over episode (same as score for single-reward games)
    float episode_length; // Number of agent actions in episode
    float n;              // Required as the last field - number of completed episodes
} Log;

// Environment struct
typedef struct {
    Log log;                      // Required field
    unsigned char* observations;  // Required. 10 values per agent: 9 cells + 1 turn flag
    int* actions;                 // Required. Position 0-8 for each agent
    float* rewards;               // Required
    unsigned char* terminals;     // Required
    int tick;                     // Number of agent actions this episode
    int num_moves;                // Total moves on board
    int current_player;           // Which player's turn (0 or 1)
} TicTacToe;

void add_log(TicTacToe* env) {
    // Log from agent 0's perspective
    env->log.perf += (env->rewards[0] > 0) ? 1.0 : 0.0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Compute observations for both agents
void compute_observations(TicTacToe* env) {
    // Agent 0: board as-is (obs[0..8]) + turn flag (obs[9])
    env->observations[9] = (env->current_player == 0) ? 1 : 0;

    // Agent 1: flipped board (obs[10..18]) + turn flag (obs[19])
    for (int i = 0; i < 9; i++) {
        unsigned char cell = env->observations[i];
        env->observations[10 + i] = (cell == 1) ? 2 : (cell == 2) ? 1 : 0;
    }
    env->observations[19] = (env->current_player == 1) ? 1 : 0;
}

// Check if a player has won
int check_winner(TicTacToe* env, unsigned char player) {
    unsigned char* board = env->observations;

    // Check rows and columns
    for (int i = 0; i < 3; i++) {
        if ((board[i*3] == player && board[i*3+1] == player && board[i*3+2] == player) ||
            (board[i] == player && board[i+3] == player && board[i+6] == player)) {
            return 1;
        }
    }

    // Check diagonals
    if ((board[0] == player && board[4] == player && board[8] == player) ||
        (board[2] == player && board[4] == player && board[6] == player)) {
        return 1;
    }

    return 0;
}

// Required function
void c_reset(TicTacToe* env) {
    // Clear board (first 9 cells of observations)
    memset(env->observations, EMPTY, 9 * sizeof(unsigned char));
    env->tick = 0;
    env->num_moves = 0;
    env->current_player = rand() % 2; // Randomly choose who starts

    // Compute observations for both agents
    compute_observations(env);
}

// Required function
void c_step(TicTacToe* env) {
    env->tick++;

    // Get action from current player
    int action = env->actions[env->current_player];

    // Zero out rewards and terminals at the start
    env->terminals[0] = 0;
    env->terminals[1] = 0;
    env->rewards[0] = 0;
    env->rewards[1] = 0;

    // Check if move is valid
    if (env->observations[action] != EMPTY) {
        // Invalid move - current player loses
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        env->rewards[env->current_player] = -1.0;
        env->rewards[1 - env->current_player] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Make current player's move
    unsigned char player_piece = (env->current_player == 0) ? AGENT : ENEMY;
    env->observations[action] = player_piece;
    env->num_moves++;

    // Check if current player won
    if (check_winner(env, player_piece)) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        env->rewards[env->current_player] = 1.0;
        env->rewards[1 - env->current_player] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Check for draw (board full)
    if (env->num_moves == 9) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        env->rewards[0] = 0.0;
        env->rewards[1] = 0.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Switch to other player
    env->current_player = 1 - env->current_player;

    // Update observations for both agents
    compute_observations(env);
}

// Required function
void c_render(TicTacToe* env) {
    if (!IsWindowReady()) {
        InitWindow(600, 600, "PufferLib TicTacToe");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int cell_size = 200;

    // Draw grid lines
    for (int i = 1; i < 3; i++) {
        DrawLine(i * cell_size, 0, i * cell_size, 600, WHITE);
        DrawLine(0, i * cell_size, 600, i * cell_size, WHITE);
    }

    // Draw X's and O's
    for (int i = 0; i < 9; i++) {
        int row = i / 3;
        int col = i % 3;
        int x = col * cell_size;
        int y = row * cell_size;

        if (env->observations[i] == AGENT) {
            // Draw X (blue)
            int margin = 40;
            DrawLineEx((Vector2){x + margin, y + margin},
                      (Vector2){x + cell_size - margin, y + cell_size - margin},
                      8.0f, (Color){0, 187, 187, 255});
            DrawLineEx((Vector2){x + cell_size - margin, y + margin},
                      (Vector2){x + margin, y + cell_size - margin},
                      8.0f, (Color){0, 187, 187, 255});
        } else if (env->observations[i] == ENEMY) {
            // Draw O (red)
            DrawCircle(x + cell_size/2, y + cell_size/2, 60, (Color){187, 0, 0, 255});
            DrawCircle(x + cell_size/2, y + cell_size/2, 40, (Color){6, 24, 24, 255});
        }
    }

    EndDrawing();
}

// Required function
void c_close(TicTacToe* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
