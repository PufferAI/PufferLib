#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "raylib.h"

#define WIN_CONDITION 4

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

const int PLAYER_WIN = 1.0;
const int ENV_WIN = -1.0;
const unsigned char DONE = 1;
const unsigned char NOT_DONE = 0;
const int ROWS = 6;
const int COLUMNS = 7;

const float MAX_VALUE = 31;
const float WIN_VALUE = 30;
const float DRAW_VALUE = 0;

#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}
 
typedef struct CConnect4 CConnect4;
struct CConnect4 {
    // Pufferlib inputs / outputs
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;

    // Bit string representation from:
    //  https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
    //  & http://blog.gamesolver.org/solving-connect-four/01-introduction/
    uint64_t player_pieces;
    uint64_t env_pieces;

    // Rendering configuration
    int piece_width;
    int piece_height;
    int width;
    int height;
};

void allocate_cconnect4(CConnect4* env) {
    env->observations = (float*)calloc(42, sizeof(float));
    env->actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_cconnect4(CConnect4* env) {
    free_logbuffer(env->log_buffer);
}

void free_allocated_cconnect4(CConnect4* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_cconnect4(env);
}

uint64_t top_mask(uint64_t column) {
    // Get the bit at the top of 'column'. If the bit is 0 the column can be
    //  played.
    return (UINT64_C(1) << (ROWS - 1)) << column * (ROWS + 1);
}

uint64_t bottom_mask(uint64_t column) {
    // Get a bit mask for where a piece played at 'column' would end up.
    return UINT64_C(1) << column * (ROWS + 1);
}

uint64_t mask(uint64_t pieces, uint64_t other_pieces) {
    // Create a piece mask containing both 'pieces' and 'other_pieces'.
    return pieces | other_pieces;
}

uint64_t bottom() {
    // A bit mask used to create unique representation of the game state.
    return UINT64_C(1) << (COLUMNS - 1) * (ROWS + 1);
}

bool invalid_move(int column, uint64_t mask) {
    return (mask & top_mask(column)) == 1;
}

uint64_t play(int column, uint64_t mask,  uint64_t other_pieces) {
    mask |= mask + bottom_mask(column);
    return other_pieces ^ mask;
}

bool draw(uint64_t mask) {
    // A full board has this specifc value
    return mask == 4432406249472;
}

bool won(uint64_t pieces) {
    // Determine if 'pieces' contains at least one line of connected pieces.

    // Horizontal 
    uint64_t m = pieces & (pieces >> (ROWS + 1));
    if(m & (m >> (2 * (ROWS + 1)))) {
        return true;
    }

    // Diagonal 1
    m = pieces & (pieces >> ROWS);
    if(m & (m >> (2 * ROWS))) {
        return true;
    }

    // Diagonal 2 
    m = pieces & (pieces >> (ROWS + 2));
    if(m & (m >> (2 * (ROWS + 2)))) {
        return true;
    }

    // Vertical;
    m = pieces & (pieces >> 1);
    if(m & (m >> 2)) {
        return true;
    }

    return false;
}

float negamax(uint64_t pieces, uint64_t other_pieces, int depth) {
    // https://en.wikipedia.org/wiki/Negamax#Negamax_variant_with_no_color_parameter
    uint64_t piece_mask = mask(pieces, other_pieces);
    
    if (depth == 0 || won(other_pieces) || draw(piece_mask)) {
        if (won(other_pieces)) {
            return -WIN_VALUE;
        }
        // Break ties between non-terminal states
        return (float) (rand() % 10);
    }

    float value = -MAX_VALUE;
    for (uint64_t column = 0; column < 7; column ++) {
        if (invalid_move(column, piece_mask)) {
            continue;
        }
        uint64_t child_position = play(column, piece_mask, other_pieces);
        value = fmax(value, -negamax(other_pieces, child_position, depth - 1));
    }
    return value;
}

int compute_env_move(CConnect4* env) {
    uint64_t piece_mask = mask(env->player_pieces, env->env_pieces);
    
    // Compute a unique hash for this game state
    uint64_t hash = env->player_pieces + piece_mask + bottom();

    // Hard coded opening book to handle some early game traps
    // TODO: Add more opening book moves
    switch (hash) {
        case 4398050705408:
            // Respond to _ _ _ o _ _ _
            // with       _ _ x o _ _ _
            return 2;
        case 4398583382016:
            // Respond to _ _ _ _ o _ _
            // with       _ _ _ x o _ _
            return 3;
    }

    // https://en.wikipedia.org/wiki/Negamax#Negamax_variant_with_no_color_parameter
    uint64_t best_column = 0;
    float best_value = -MAX_VALUE;
    for (uint64_t column = 0; column < 7; column ++) {

        if (invalid_move(column, piece_mask)) {
            continue;
        }
        uint64_t child_env_pieces = play(column, piece_mask, env->player_pieces);
        float value = -negamax(env->player_pieces, child_env_pieces, 2);
        if (value > best_value) {
            best_value = value;
            best_column = column;
        }
    }
    return best_column;
}

void compute_observation(CConnect4* env) {
    // Use the bitstring representation of the game state to populate
    //  the observations vector.
    // Details: http://blog.gamesolver.org/solving-connect-four/06-bitboard/
    uint64_t player_pieces = env->player_pieces;
    uint64_t env_pieces = env->env_pieces;

    int obs_idx = 0;
    for (int i = 0; i < 49; i++) {
        // Skip the sentinel row
        if ((i + 1) % 7 == 0) {
            continue;
        }

        int p0_bit = (player_pieces >> i) & 1;
        if (p0_bit == 1) {
            env->observations[obs_idx] = PLAYER_WIN;
        }
        int p1_bit = (env_pieces >> i) & 1;
        if (p1_bit == 1) {
            env->observations[obs_idx] = ENV_WIN;
        }
        obs_idx += 1;
    }
}

void reset(CConnect4* env) {
    env->log = (Log){0};
    env->dones[0] = NOT_DONE;
    env->player_pieces = 0;
    env->env_pieces = 0;
    for (int i = 0; i < 42; i ++) {
        env->observations[i] = 0.0;
    }
}

void finish_game(CConnect4* env, float reward) {
    env->rewards[0] = reward;
    env->dones[0] = DONE;
    env->log.score = reward;
    env->log.episode_return = reward;
    compute_observation(env);
}

void step(CConnect4* env) {
    uint64_t column, piece_mask;

    env->log.episode_length += 1;
    env->rewards[0] = 0.0;

    if (env->dones[0] == DONE) {
        add_log(env->log_buffer, &env->log);
        reset(env);
        return;
    }

    // Player action (PLAYER_WIN)
    column = env->actions[0];

    piece_mask = mask(env->player_pieces, env->env_pieces);
    if (invalid_move(column, piece_mask)) {
        finish_game(env, ENV_WIN);
        return;
    }

    env->player_pieces = play(column, piece_mask, env->env_pieces);

    if (won(env->player_pieces)) {
        finish_game(env, PLAYER_WIN);
        return;
    }

    // Environment action (ENV_WIN)
    column = compute_env_move(env);

    piece_mask = mask(env->player_pieces, env->env_pieces);
    if (invalid_move(column, piece_mask)) {
        finish_game(env, PLAYER_WIN);
        return;
    }

    env->env_pieces = play(column, piece_mask, env->player_pieces);

    if (won(env->env_pieces)) {
        finish_game(env, ENV_WIN);
        return;
    }

    compute_observation(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;

    InitWindow(width, height, "PufferLib Ray Connect4");
    SetTargetFPS(15);

    client->puffers = LoadTexture("resources/puffers_128.png");
    return client;
}

void render(Client* client, CConnect4* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    int y_offset = client->height - env->piece_height;
    int obs_idx = 0;
    for (int i = 0; i < 49; i++) {
        // TODO: Simplify this by iterating over the observation more directly
        if ((i + 1) % 7 == 0) {
            continue;
        }

        int row = i % (ROWS + 1);
        int column = i / (ROWS + 1);
        int y = y_offset - row * env->piece_height;
        int x = column * env->piece_width;

        Color piece_color=PURPLE;
        int color_idx = 0;
        if (env->observations[obs_idx] == 0.0) {
            piece_color = BLACK;
        } else if (env->observations[obs_idx]  == PLAYER_WIN) {
            piece_color = PUFF_CYAN;
            color_idx = 1;
        } else if (env->observations[obs_idx]  == ENV_WIN) {
            piece_color = PUFF_RED;
            color_idx = 2;
        }
        obs_idx += 1;

        Color board_color = (Color){0, 80, 80, 255};
        DrawRectangle(x , y , env->piece_width, env->piece_width, board_color);
        DrawCircle(x + env->piece_width/2, y + env->piece_width/2, env->piece_width/2, piece_color);
        if (color_idx == 0) {
            continue;
        }
        DrawTexturePro(
            client->puffers,
            (Rectangle){
                (color_idx == 1) ? 0 : 128,
                0, 128, 128,
            },
            (Rectangle){x+16, y+16, env->piece_width-32, env->piece_width-32},
            (Vector2){0, 0},
            0,
            WHITE
        );
    }
    EndDrawing();
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
