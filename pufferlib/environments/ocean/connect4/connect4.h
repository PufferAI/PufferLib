#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>

#define NOOP 0
#define PLACE_PIECE_1 1
#define PLACE_PIECE_2 2
#define PLACE_PIECE_3 3
#define PLACE_PIECE_4 4
#define PLACE_PIECE_5 5
#define PLACE_PIECE_6 6
#define PLACE_PIECE_7 7
#define WIN_CONDITION 4
#define Y_OFFSET 10
#define TICK_RATE 1.0f/60.0f

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// how to start game compile - LD_LIBRARY_PATH=raylib-5.0_linux_amd64/lib ./connect4game 

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
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    int piece_width;
    int piece_height;
    float* board_x;
    float* board_y;
    float board_states[6][7];
    int width;
    int height;
    int game_over;
    int pieces_placed;
};

struct ActionValue {
    int column;
    float reward;
};

void generate_board_positions(CConnect4* env) {
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 7; col++) {
            int idx = row * 7 + col;
            env->board_x[idx] = col* env->piece_width;
            env->board_y[idx] = row*env->piece_height;
        }
    }
}

void init_cconnect4(CConnect4* env) {
    // Allocate memory for board_x, board_y, and board_states
    env->board_x = (float*)calloc(42, sizeof(float));
    env->board_y = (float*)calloc(42, sizeof(float));
    for(int i=0; i< 6; i++) {
        for(int j=0; j< 7; j++) {
            env->board_states[i][j] = 0.0;
        }
    }
    generate_board_positions(env);
}

void allocate_cconnect4(CConnect4* env) {
    env->observations = (float*)calloc(42, sizeof(float));
    env->actions = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    init_cconnect4(env);
}

void free_cconnect4(CConnect4* env) {
    free(env->board_x);
    free(env->board_y);
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

void compute_observations(CConnect4* env) {
    int idx=0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++) {
            env->observations[idx] = env->board_states[i][j];
            idx++;
        }
    }
}

void reset(CConnect4* env) {
    env->log = (Log){0};
    for(int i=0; i< 6; i++) {
        for(int j=0; j< 7; j++) {
            env->board_states[i][j] = 0.0;
        }
    }
    env->pieces_placed = 0;
    env->dones[0] = 0;
    compute_observations(env);
}

bool check_draw_condition(CConnect4* env) {
    // Check whether a draw has been reached
    return env->pieces_placed >= 42;
}

int compute_reward(CConnect4* env, int row, int col) {
    // Compute the reward, restrict the computation to lines going through 
    // 'row', 'col'.
    float player = env->board_states[row][col];
    
    if (player == 0.0) {
        // There is no piece at this position
        return 0.0;
    }
    // Horizontal, Vertical, Diagonal down, Diagonal up
    int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}}; 

    for (int d = 0; d < 4; d++) {
        int count = 1;
        int r = row;
        int c = col;
        
        // Check in positive direction
        while (true) {
            r += directions[d][0];
            c += directions[d][1];
            if (r < 0 || r >= 6 || c < 0 || c >= 7 || env->board_states[r][c] != player) break;
            count++;
        }
        
        // Check in negative direction
        r = row;
        c = col;
        while (true) {
            r -= directions[d][0];
            c -= directions[d][1];
            if (r < 0 || r >= 6 || c < 0 || c >= 7 || env->board_states[r][c] != player) break;
            count++;
        }
        
        if (count >= WIN_CONDITION) {
            return player;
        }
    }
    return 0.0;
}

int stage_move(CConnect4* env, int column, int player) {
    for (int row = 5; row >= 0; row--) {
        if (env->board_states[row][column] == 0.0) {
            env->board_states[row][column] = player;
            return row;
        }
    }
    return -1;
}

void commit_move(CConnect4* env, int row, int column) {
    // Commit a move previously staged with 'stage_move'

    if (row == -1) {
        // Move is invalid, don't do anything
        return;
    }
    env->pieces_placed++;
    float reward = compute_reward(env, row, column);
    bool win = (reward != 0.0);
    if (win) {
        env->dones[0] = 1;
        env->rewards[0] = reward;
        env->log.score = reward;
        env->log.episode_return = reward;
        add_log(env->log_buffer, &env->log);
    } else {
        bool draw = check_draw_condition(env);
        if (draw) {
            env->dones[0] = 1;
        }
    }
}

void unstage_move(CConnect4* env, int row, int column) {
    // Undo a move previously staged with 'stage_move'
    env->board_states[row][column] = 0.0;
}

struct ActionValue minmax(CConnect4* env, struct ActionValue action_value, float player, int depth) {
    // Find the best action to play using the minmax algorithm 
    // https://en.wikipedia.org/wiki/Minimax

    if (depth == 0 || check_draw_condition(env) || action_value.reward != 0.0) {
        return action_value;
    }

    bool maximising_player = player == 1.0;

    struct ActionValue best = { -1, maximising_player ? -100.0 : 100};
    for (int column = 0; column < 7; column++) {
        int row = stage_move(env, column, player);

        bool invalid = (row == -1);
        if (invalid) {
            continue;
        }

        int tmpcol = action_value.column == -1 ? column : action_value.column;
        float tmpreward = compute_reward(env, row, column);

        struct ActionValue child_action_value = minmax(
            env,
            (struct ActionValue) { tmpcol, tmpreward},
            -player,
            depth - 1
        );

        unstage_move(env, row, column);

        if (maximising_player) {
            if (child_action_value.reward > best.reward) {
                best = child_action_value;
            }
        } else {
            if (child_action_value.reward < best.reward) {
                best = child_action_value;
            }
        }
    }

    return best;
}

void step(CConnect4* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = env->actions[0];

    if (env->game_over == 1) {
        add_log(env->log_buffer, &env->log);
        reset(env);
        env->game_over = 0;
        return;
    }

    // Input player action
    if (action >= PLACE_PIECE_1 && action <= PLACE_PIECE_7) {
        int column = action - PLACE_PIECE_1;
        int row = stage_move(env, column, 1.0);
        commit_move(env, row, column);
    }

    // Scripted opponent action
    if (action != NOOP && env->dones[0] == 0)  {
        struct ActionValue action_value = minmax(env, (struct ActionValue) { -1, 0.0 }, -1.0, 4);
        // printf("column=%d, reward=%f\n", action_value.column, action_value.reward);

        int row = stage_move(env, action_value.column, -1.0);
        commit_move(env, row, action_value.column);
    }

    if (env->dones[0] == 1) {
        env->game_over=1;
    }
    compute_observations(env);
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

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    client->puffers = LoadTexture("resources/puffers_128.png");
    return client;
}

void render(Client* client, CConnect4* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // create a connect 4 board. there should be a hollow outline of tthe board where its a grid of 7x6 and then a circle in each of the 42 slots.
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 7; col++) {
            int board_idx = row * 7 + col;
            Color piece_color=PURPLE;
            int color_idx = 0;
            if (env->board_states[row][col] == 0.0) {
                piece_color = BLACK;
            } else if (env->board_states[row][col] == 1.0) {
                piece_color = PUFF_CYAN;
                color_idx = 1;
            } else if (env->board_states[row][col] == -1.0) {
                piece_color = PUFF_RED;
                color_idx = 2;
            }
            int x = env->board_x[board_idx];
            int y = env->board_y[board_idx];
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
    }
    EndDrawing();
    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
