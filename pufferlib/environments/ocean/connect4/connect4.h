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
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
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
    int* longest_connected;
    int width;
    int height;
    int game_over;
    int pieces_placed;
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
    env->longest_connected = (int*)calloc(2,sizeof(int));
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
    free(env->longest_connected);
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
    env->longest_connected[0] = 0;
    env->longest_connected[1] = 0;
    env->pieces_placed = 0;
    env->dones[0] = 0;
    compute_observations(env);
}

// if place piece at bottom of column 0 if no pieces is there idx should be 35
// if there is a piece, it should be location 28
int place_piece(CConnect4* env, int action_location, int player) {
    for (int row = 5; row >= 0; row--) {
        if (env->board_states[row][action_location] == 0.0) {
            env->board_states[row][action_location] = player;
            env->pieces_placed++;
            return row;
        }
    }
    return -1;
}

void check_draw_condition(CConnect4* env) {
    if (env->pieces_placed >= 42) {
        env->dones[0] = 1;
    }
}

void check_win_condition(CConnect4* env, int player, int selected_row, int selected_col) {
    int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}}; // Horizontal, Vertical, Diagonal down, Diagonal up
    int player_idx = player == 1 ? 0 : 1;
    env->longest_connected[player_idx] = 1; // Initialize to 1 (single piece)

    for (int d = 0; d < 4; d++) {
        int count = 1;
        int r = selected_row;
        int c = selected_col;
        
        // Check in positive direction
        while (true) {
            r += directions[d][0];
            c += directions[d][1];
            if (r < 0 || r >= 6 || c < 0 || c >= 7 || env->board_states[r][c] != player) break;
            count++;
        }
        
        // Check in negative direction
        r = selected_row;
        c = selected_col;
        while (true) {
            r -= directions[d][0];
            c -= directions[d][1];
            if (r < 0 || r >= 6 || c < 0 || c >= 7 || env->board_states[r][c] != player) break;
            count++;
        }
        
        if (count > env->longest_connected[player_idx]) {
            env->longest_connected[player_idx] = count;
        }
        if (count >= WIN_CONDITION) {
            env->dones[0] = 1;
            env->rewards[0] = player; // 1 for player win, -1 for opponent win
            env->log.score = player;
            env->log.episode_return = player;
            add_log(env->log_buffer, &env->log);
            return;
        }
    }
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

    if (action >= PLACE_PIECE_1 && action <= PLACE_PIECE_7) {
        int selected_row = place_piece(env, action - PLACE_PIECE_1, 1);
        check_win_condition(env, 1, selected_row, action - PLACE_PIECE_1);
        check_draw_condition(env);
    }

    // generate random action from 1- 6 must be int and set board state to -1
    if (action != NOOP && env->dones[0] == 0)  {
        int random_action = rand() % 6 + 1;
        int selected_row = place_piece(env, random_action, -1);
        check_win_condition(env, -1, selected_row, random_action);
        check_draw_condition(env);
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
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;

    InitWindow(width, height, "PufferLib Ray Connect4");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    return client;
}

void render(Client* client, CConnect4* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});


    // create a connect 4 board. there should be a hollow outline of tthe board where its a grid of 7x6 and then a circle in each of the 42 slots.
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 7; col++) {
            int board_idx = row * 7 + col;
            Color piece_color=PURPLE;

            if (env->board_states[row][col] == 0.0) {
                piece_color = BLACK;
            } else if (env->board_states[row][col] == 1.0) {
                piece_color = RED;
            } else if (env->board_states[row][col] == -1.0) {
                piece_color = BLUE;
            }
            int x = env->board_x[board_idx];
            int y = env->board_y[board_idx];
            Color board_color = DARKBLUE;
            DrawRectangle(x , y , env->piece_width, env->piece_width, board_color);
            DrawCircle(x + env->piece_width/2, y + env->piece_width/2, env->piece_width/2, piece_color);
        }
    }

    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
