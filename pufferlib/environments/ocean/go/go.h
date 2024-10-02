#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define NOOP 0
#define MOVE_MIN 1
#define MOVE_MAX 1000
#define HALF_MAX_SCORE 432
#define MAX_SCORE 864
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f

//  LD_LIBRARY_PATH=raylib-5.0_linux_amd64/lib ./gogame

typedef struct CGo CGo;
struct CGo {
    float* observations;
    unsigned short* actions;
    float* rewards;
    unsigned char* dones;
    int score;
    float episode_return;
    int width;
    int height;
    int* board_x;
    int* board_y;
    int board_width;
    int board_height;
    int grid_square_size;
    int grid_size;
    int* board_states;
};

void generate_board_positions(CGo* env) {
    for (int row = 0; row < env->grid_size ; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int idx = row * env->grid_size + col;
            env->board_x[idx] = col* env->grid_square_size;
            env->board_y[idx] = row*env->grid_square_size;
        }
    }
}

static inline int board_state_offset(CGo* env, int pos, int state) {
    return state*(env->grid_size+1)*(env->grid_size+1) + pos;
}

void init(CGo* env) {
    env->board_x = (int*)calloc((env->grid_size)*(env->grid_size), sizeof(int));
    env->board_y = (int*)calloc((env->grid_size)*(env->grid_size), sizeof(int));
    env->board_states = (int*)calloc((env->grid_size+1)*(env->grid_size+1)*3, sizeof(int));
    generate_board_positions(env);
}

void allocate(CGo* env) {
    init(env);
    env->observations = (float*)calloc(11, sizeof(float));
    env->actions = (unsigned short*)calloc(1, sizeof(unsigned short));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_initialized(CGo* env) {
    free(env->board_x);
    free(env->board_y);
    free(env->board_states);
}

void free_allocated(CGo* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_initialized(env);
}

void compute_observations(CGo* env) {

}

bool check_legal_placement(CGo* env, int tile_placement) {
    if (env->board_states[board_state_offset(env,tile_placement,0)] != 0) {
        return 0;
    } else {
        return 1;
    }
}


  
void reset(CGo* env) {
    env->dones[0] = 0;
    for (int i = 0; i < (env->grid_size+1)*(env->grid_size+1)*3; i++) {
        env->board_states[i] = 0;
    }
}

void step(CGo* env) {
    env->rewards[0] = 0.0;
    int action = (int)env->actions[0];
    if (action >= MOVE_MIN) {
        env->board_states[board_state_offset(env, action-1, 1)] = 1;
        env->board_states[board_state_offset(env,action-1,0)]=1;

        // opponent move
        // Opponent move (player 2)
        int legal_moves[361];  // Maximum possible moves on a 19x19 board
        int num_legal_moves = 0;
        
        // Find all legal moves
        for (int i = 0; i < (env->grid_size+1)*(env->grid_size+1); i++) {
            if (check_legal_placement(env, i)) {
                legal_moves[num_legal_moves++] = i;
            }
        }
        
        // Randomly select a legal move
        if (num_legal_moves > 0) {
            int random_index = rand() % num_legal_moves;
            int opponent_move = legal_moves[random_index];
            env->board_states[board_state_offset(env, opponent_move, 2)] = 1;
            env->board_states[board_state_offset(env,opponent_move,0)]=1;

        }
    }

    if (env->dones[0] == 1) {
        env->episode_return = env->score;
        reset(env);
    }
    compute_observations(env);
}


typedef struct Client Client;
struct Client {
    float width;
    float height;
};

Client* make_client(CGo* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Ray Breakout");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    return client;
}

void render(Client* client, CGo* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int row = 0; row < env->grid_size; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int idx = row * env->grid_size + col;
            int x = env->board_x[idx];
            int y = env->board_y[idx];
            Color tile_color = (Color){ 253, 208, 124, 255 };
            

            DrawRectangle(x+100, y+100, env->grid_square_size, env->grid_square_size, tile_color);
            DrawRectangleLines(x+100, y+100, env->grid_square_size, env->grid_square_size, BLACK);
            

        }
    }

    for (int i = 0; i < (env->grid_size + 1) * (env->grid_size + 1); i++) {

        int player_state = env->board_states[board_state_offset(env, i, 1)];
        int enemy_state = env->board_states[board_state_offset(env, i, 2)];

        int row = i / (env->grid_size + 1);
        int col = i % (env->grid_size + 1);
        int x = col * env->grid_square_size;
        int y = row * env->grid_square_size;



        // Calculate the circle position based on the grid
        int circle_x = x + 100;
        int circle_y = y + 100;
        // if player draw circle tile for black 
        if (player_state == 1) {
            DrawCircle(circle_x, circle_y, env->grid_square_size / 2, PINK);
            DrawCircleLines(circle_x, circle_y, env->grid_square_size / 2, PINK);
        }
        // if enemy draw circle tile for white
        if (enemy_state == 1) {
            DrawCircle(circle_x, circle_y, env->grid_square_size / 2, WHITE);
            DrawCircleLines(circle_x, circle_y, env->grid_square_size / 2, WHITE);
        }
    }

    EndDrawing();
    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
