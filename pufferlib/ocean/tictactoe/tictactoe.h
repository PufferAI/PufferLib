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
    unsigned char* observations;  // Required. 9 cells, values 0-2
    int* actions;                 // Required. Position 0-8
    float* rewards;               // Required
    unsigned char* terminals;     // Required
    int tick;                     // Number of agent actions this episode
    int num_moves;                // Total moves (agent + enemy) on board
} TicTacToe;

void add_log(TicTacToe* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0 : 0.0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
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
    memset(env->observations, EMPTY, 9 * sizeof(unsigned char));
    env->tick = 0;
    env->num_moves = 0;

    // Randomly decide if enemy goes first
    int enemy_first = rand() % 2;
    if (enemy_first) {
        int enemy_move = rand() % 9;
        env->observations[enemy_move] = ENEMY;
        env->num_moves = 1;
    }
}

// Required function
void c_step(TicTacToe* env) {
    env->tick++;

    int action = env->actions[0];

    // Zero out rewards and terminals at the start
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    // Check if agent's move is valid
    if (env->observations[action] != EMPTY) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Make agent's move
    env->observations[action] = AGENT;
    env->num_moves++;

    // Check if agent won
    if (check_winner(env, AGENT)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Check for draw (board full)
    if (env->num_moves == 9) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0;
        add_log(env);
        c_reset(env);
        return;
    }

    // Enemy makes a random valid move
    int enemy_move;
    int attempts = 0;
    do {
        enemy_move = rand() % 9;
        attempts++;
        if (attempts > 100) break; // Safety check
    } while (env->observations[enemy_move] != EMPTY);

    if (env->observations[enemy_move] == EMPTY) {
        env->observations[enemy_move] = ENEMY;
        env->num_moves++;

        // Check if enemy won
        if (check_winner(env, ENEMY)) {
            env->terminals[0] = 1;
            env->rewards[0] = -1.0;
            add_log(env);
            c_reset(env);
            return;
        }

        // Check for draw after enemy move
        if (env->num_moves == 9) {
            env->terminals[0] = 1;
            env->rewards[0] = 0.0;
            add_log(env);
            c_reset(env);
            return;
        }
    }
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
