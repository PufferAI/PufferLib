#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>

#define WIN_CONDITION 4

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

const int P0 = 1.0;
const int P1 = -1.0;
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
    uint64_t position;
    uint64_t mask;

    // Rendering configuration
    int piece_width;
    int piece_height;
    int width;
    int height;
};

void init_cconnect4(CConnect4* env) {}

void allocate_cconnect4(CConnect4* env) {
    env->observations = (float*)calloc(42, sizeof(float));
    env->actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    init_cconnect4(env);
}

void free_cconnect4(CConnect4* env) {
    free(env->log_buffer);
    free(env);
}

void free_allocated_cconnect4(CConnect4* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_cconnect4(env);
}

uint64_t top_mask(int column) {
  return (UINT64_C(1) << (ROWS - 1)) << column * (ROWS + 1);
}

uint64_t bottom_mask(int column) {
  return UINT64_C(1) << column * (ROWS + 1);
}

uint64_t bottom() {
  return UINT64_C(1) << (COLUMNS - 1) * (ROWS + 1);
}

bool valid_move(int column, u_int64_t mask) {
    return (mask & top_mask(column)) == 0;
}

uint64_t play(int column, u_int64_t mask) {
    mask |= mask + bottom_mask(column);
    return mask;
}

bool draw(uint64_t mask) {
    return mask == 4432406249472;
}

bool won(uint64_t position) {
  // Horizontal 
  uint64_t m = position & (position >> (ROWS + 1));
  if(m & (m >> (2 * (ROWS + 1)))) return true;

  // Diagonal 1
  m = position & (position >> ROWS);
  if(m & (m >> (2 * ROWS))) return true;

  // Diagonal 2 
  m = position & (position >> (ROWS + 2));
  if(m & (m >> (2 * (ROWS + 2)))) return true;

  // Vertical;
  m = position & (position >> 1);
  if(m & (m >> 2)) return true;

  return false;
}

float minmax(u_int64_t position, u_int64_t mask, int depth, bool maximising_player) {
    // https://en.wikipedia.org/wiki/Minimax
    bool has_won = won(position);
    if (depth == 0 || has_won || draw(mask)) {
        if (has_won) {
            float discount = (4 - depth) * 0.1; // Discount short games
            if (maximising_player) {
                return WIN_VALUE + discount;
            } else {
                return -WIN_VALUE - discount;
            }
        }
        return DRAW_VALUE;
    }

    float value;
    if (maximising_player) {
        value = -MAX_VALUE;
    } else {
        value = MAX_VALUE;
    }
    for (uint64_t column = 0; column < 7; column ++) {
        if (!valid_move(column, mask)) {
            continue;
        }
        u_int64_t child_position = position ^ mask;
        u_int64_t child_mask = play(column, mask);
        float child_value = minmax(child_position, child_mask, depth - 1, !maximising_player);
        if (maximising_player && child_value > value) {
            value = child_value;
        }
        if (!maximising_player && child_value < value) {
            value = child_value;
        }
    }
    return value;
}

int compute_env_move(CConnect4* env) {
    // Hard coded opening book
    uint64_t hash = env->position + env->mask + bottom();
    // TODO: Add more opening book moves
    if (hash == 4398048608256) {
        return 2;
    }

    uint64_t best_column = 0;
    float best_value = -MAX_VALUE;
    for (uint64_t column = 0; column < 7; column ++) {
        if (!valid_move(column, env->mask)) { continue; }
        u_int64_t child_position = env->position ^ env->mask;
        u_int64_t child_mask = play(column, env->mask);
        
        float value = minmax(child_position, child_mask, 2, false);
        if (value == DRAW_VALUE) {
            // Break ties between equal valued positions randomly
            // TODO: Implement a heuristic for non-terminal states
            value = (float) (rand() % 10);
        }

        if (value > best_value) {
            best_value = value;
            best_column = column;
        }
    }
    return best_column;
}

void compute_observation(CConnect4* env) {
    // Use the bitstring representation of the game state to populate
    //  the observations vector
    uint64_t p0 = env->position;
    uint64_t p1 = env->position ^ env->mask;

    int obs_idx = 0;
    for (int i = 0; i < 49; i++) {
        // Skip the sentinel row
        if ((i + 1) % 7 == 0) { continue; }
        obs_idx += 1;

        int p0_bit = (p0 >> i) & 1;
        if (p0_bit == 1) {
            env->observations[obs_idx] = P0;
        }
        int p1_bit = (p1 >> i) & 1;
        if (p1_bit == 1) {
            env->observations[obs_idx] = P1;
        }
    }
}

void reset_observation(CConnect4* env) {
    for (int i = 0; i < 42; i ++) {
        env->observations[i] = 0.0;
    }
}

void reset(CConnect4* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->position = 0;
    env->mask = 0;

    reset_observation(env);
}

void step(CConnect4* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;

    if (env->dones[0] == 1) {
        add_log(env->log_buffer, &env->log);
        reset(env);
        return;
    }

    // Input player action
    if (true) {
        uint64_t column = env->actions[0];;
        if (valid_move(column, env->mask)) {
            env->position ^= env->mask; // Swap player
            env->mask = play(column, env->mask);

            if (won(env->position)) {
                int reward = P1;
                env->rewards[0] = reward;
                env->dones[0] = 1;
                env->position ^= env->mask;
                env->log.score = reward;
                env->log.episode_return = reward;
                add_log(env->log_buffer, &env->log);
            }
        } else {
            env->rewards[0] = P1;
            env->dones[0] = 1;
        }
    }

    // Scripted opponent action
    if (env->dones[0] == 0) {
        uint64_t column = compute_env_move(env);

        if (valid_move(column, env->mask)) {
            env->position ^= env->mask; // Swap player
            env->mask = play(column, env->mask);

            if (won(env->position)) {
                int reward = P0;
                env->rewards[0] = reward;
                env->dones[0] = 1;

                env->log.score = reward;
                env->log.episode_return = reward;
                add_log(env->log_buffer, &env->log);
            }
        } else {
            env->rewards[0] = P0;
            env->dones[0] = 1;
        }
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
        if ((i + 1) % 7 == 0) { continue; }
        obs_idx += 1;

        int row = i % (ROWS + 1);
        int column = i / (ROWS + 1);
        int y = y_offset - row * env->piece_height;
        int x = column * env->piece_width;

        Color piece_color=PURPLE;
        int color_idx = 0;
        if (env->observations[obs_idx] == 0.0) {
            piece_color = BLACK;
        } else if (env->observations[obs_idx]  == P0) {
            piece_color = PUFF_CYAN;
            color_idx = 1;
        } else if (env->observations[obs_idx]  == P1) {
            piece_color = PUFF_RED;
            color_idx = 2;
        }

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
