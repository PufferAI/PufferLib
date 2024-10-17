#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"
#include <string.h>
#define NOOP 0
#define MOVE_MIN 1
#define MOVE_MAX 1000
#define HALF_MAX_SCORE 432
#define MAX_SCORE 864
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f
#define NUM_DIRECTIONS 4
static const int DIRECTIONS[NUM_DIRECTIONS][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
//  LD_LIBRARY_PATH=raylib-5.0_linux_amd64/lib ./go
#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    int games_played;
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
        log.games_played += logs->logs[i].games_played;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct DisjointSetNode DisjointSetNode;
struct DisjointSetNode{
    int parent;
    int rank;
    int liberties;
};



// Find the root of the set containing element x
int find(DisjointSetNode* disjoint_set, int x) {
    if (disjoint_set[x].parent != x) {
        disjoint_set[x].parent = find(disjoint_set, disjoint_set[x].parent);
    }
    return disjoint_set[x].parent;
}

// Union the sets containing elements x and y
void union_sets(DisjointSetNode* disjoint_set, int x, int y) {
    int rootX = find(disjoint_set,x);
    int rootY = find(disjoint_set,y);

    if (rootX != rootY) {
        if (disjoint_set[rootX].rank < disjoint_set[rootY].rank) {
            disjoint_set[rootX].parent = rootY;
        } else if (disjoint_set[rootX].rank > disjoint_set[rootY].rank) {
            disjoint_set[rootY].parent = rootX;
        } else {
            disjoint_set[rootY].parent = rootX;
            disjoint_set[rootX].rank++;
        }
    }
}

typedef struct CGo CGo;
struct CGo {
    float* observations;
    unsigned short* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    DisjointSetNode* disjoint_set;
    float score;
    int width;
    int height;
    int* board_x;
    int* board_y;
    int board_width;
    int board_height;
    int grid_square_size;
    int grid_size;
    int* board_states;
    int* previous_board_state;
    int last_capture_position;
    int* temp_board_states;
    DisjointSetNode* temp_disjoint_set;
    int moves_made;
    int* capture_count;
    float komi;
    int* visited;
};

// Initialize the disjoint set
void init_disjoint_sets(CGo* env) {
    int size = (env->grid_size + 1) * (env->grid_size + 1);
    for (int i = 0; i < size; i++) {
        env->disjoint_set[i].parent = i;
        env->disjoint_set[i].rank = 0;
        env->disjoint_set[i].liberties = 0;
        env->temp_disjoint_set[i].parent = i;
        env->temp_disjoint_set[i].rank = 0;
        env->temp_disjoint_set[i].liberties = 0;
    }
}

// Update the disjoint set to track groups and their liberties
void update_groups_and_liberties(CGo* env, DisjointSetNode* disjoint_set, int* board_states) {
    int grid_size = env->grid_size + 1;
    int total_positions = grid_size * grid_size;

    // Initialize disjoint set
    for (int i = 0; i < total_positions; i++) {
        disjoint_set[i].parent = i;
        disjoint_set[i].rank = 0;
        disjoint_set[i].liberties = 0;
    }

    // First pass: Union adjacent stones of the same player
    for (int pos = 0; pos < total_positions; pos++) {
        if (board_states[pos] == 0) {
            continue;
        }
        int x = pos % grid_size;
        int y = pos / grid_size;

        for (int i = 0; i < NUM_DIRECTIONS; i++) {
            int nx = x + DIRECTIONS[i][0];
            int ny = y + DIRECTIONS[i][1];
            int npos = ny * grid_size + nx;

            if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                if (board_states[npos] == board_states[pos]) {
                    union_sets(disjoint_set, pos, npos);
                }
            }
        }
    }

    // Second pass: Calculate liberties for each group
    for (int pos = 0; pos < total_positions; pos++) {
        if (board_states[pos] != 0) {
            continue; // Skip occupied positions
        }
        int x = pos % grid_size;
        int y = pos / grid_size;

        for (int i = 0; i < NUM_DIRECTIONS; i++) {
            int nx = x + DIRECTIONS[i][0];
            int ny = y + DIRECTIONS[i][1];
            int npos = ny * grid_size + nx;

            if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                if (board_states[npos] != 0) {
                    int root = find(disjoint_set, npos);
                    // Only increment liberties once per empty position per group
                    disjoint_set[root].liberties++;
                }
            }
        }
    }
}

void generate_board_positions(CGo* env) {
    for (int row = 0; row < env->grid_size ; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int idx = row * env->grid_size + col;
            env->board_x[idx] = col* env->grid_square_size;
            env->board_y[idx] = row*env->grid_square_size;
        }
    }
}

void init(CGo* env) {
    int size = (env->grid_size + 1) * (env->grid_size + 1);
    env->board_x = (int*)calloc((env->grid_size)*(env->grid_size), sizeof(int));
    env->board_y = (int*)calloc((env->grid_size)*(env->grid_size), sizeof(int));
    env->board_states = (int*)calloc(size, sizeof(int));
    env->temp_board_states = (int*)calloc(size, sizeof(int));
    env->visited = (int*)calloc(size, sizeof(int));
    env->previous_board_state = (int*)calloc(size, sizeof(int));
    env->capture_count = (int*)calloc(2, sizeof(int));
    env->disjoint_set = (DisjointSetNode*)calloc(size, sizeof(DisjointSetNode));
    env->temp_disjoint_set = (DisjointSetNode*)calloc(size, sizeof(DisjointSetNode));
    init_disjoint_sets(env);
    generate_board_positions(env);
}

void allocate(CGo* env) {
    init(env);
    env->observations = (float*)calloc((env->grid_size+1)*(env->grid_size+1)*2, sizeof(float));
    env->actions = (unsigned short*)calloc(1, sizeof(unsigned short));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(CGo* env) {
    free(env->board_x);
    free(env->board_y);
    free(env->board_states);
    free(env->visited);
    free(env->previous_board_state);
    free(env->temp_board_states);
    free(env->capture_count);
    free(env->disjoint_set);
}

void free_allocated(CGo* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_initialized(env);
}

void compute_observations(CGo* env) {
    for (int i = 0; i < (env->grid_size+1)*(env->grid_size+1); i++) {
        env->observations[i] = (float)env->board_states[i];
    }
    for (int i = 0; i < (env->grid_size+1)*(env->grid_size+1); i++) {
        env->observations[i + (env->grid_size+1)*(env->grid_size+1)] = (float)env->previous_board_state[i];
    }
    // env->observations[2*(env->grid_size+1)*(env->grid_size+1)] = (float)env->capture_count[0];
    // env->observations[2*(env->grid_size+1)*(env->grid_size+1)+1] = (float)env->capture_count[1];

}

void reset_visited(CGo* env) {
    for (int i = 0; i < (env->grid_size + 1) * (env->grid_size + 1); i++) {
        env->visited[i] = 0;
    }
}

void check_capture_pieces(CGo* env, int* board, int tile_placement, int simulate, DisjointSetNode* disjoint_set) {
    // Update groups and liberties using the provided disjoint set and board state
    update_groups_and_liberties(env, disjoint_set, board);

    int player = board[tile_placement];
    int opponent = (player == 1) ? 2 : 1;
    bool captured = false;
    int grid_size = env->grid_size + 1;

    int x = tile_placement % grid_size;
    int y = tile_placement / grid_size;

    // Check adjacent positions for capture opportunities
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
        int nx = x + DIRECTIONS[i][0];
        int ny = y + DIRECTIONS[i][1];
        int npos = ny * grid_size + nx;

        if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) {
            continue;
        }
        if (board[npos] != opponent) {
            continue;
        }

        int root = find(disjoint_set, npos);

        // If the opponent's group has no liberties, capture it
        if (disjoint_set[root].liberties == 0) {
            captured = true;
            for (int pos = 0; pos < grid_size * grid_size; pos++) {
                if (find(disjoint_set, pos) == root) {
                    board[pos] = 0;  // Remove captured stones
                    if (simulate != 0) {
                        continue;
                    }
                    // Update capture count and rewards
                    env->capture_count[player - 1]++;
                    if (player == 1) {
                        env->rewards[0] += 0.1f;
                        env->log.episode_return += 0.1f;
                    } else {
                        env->rewards[0] -= 0.1f;
                        env->log.episode_return -= 0.1f;
                    }
                    env->last_capture_position = pos;
                }
            }
        }
    }

    if (!captured) {
        env->last_capture_position = -1;
    }
}
bool check_legal_placement(CGo* env, int tile_placement, int player) {
    if (env->board_states[tile_placement] != 0) {
        return false;
    } 

    int size = (env->grid_size + 1) * (env->grid_size + 1);

    // Create a temporary board to simulate the move
    memcpy(env->temp_board_states, env->board_states, sizeof(int) * size);
    env->temp_board_states[tile_placement] = player;

    // Check if this move would capture any opponent stones
    check_capture_pieces(env, env->temp_board_states, tile_placement, 1, env->temp_disjoint_set);

    // Check if the placed stone has liberties after potential captures
    int root = find(env->temp_disjoint_set, tile_placement);
    if (env->temp_disjoint_set[root].liberties == 0) {
        return false;  // The move results in self-capture, so it's illegal
    }

    // Check for Ko rule violation
    if (env->last_capture_position != -1) {
        bool is_ko = true;
        for (int i = 0; i < size; i++) {
            if (env->temp_board_states[i] != env->previous_board_state[i]) {
                is_ko = false;
                break;
            }
        }
        if (is_ko) {
            return false;  // Ko rule violation
        }
    }
    return true;
}


void flood_fill(CGo* env, int x, int y, int* territory, int player) {
    if (x < 0 || x >= env->grid_size + 1 || y < 0 || y >= env->grid_size + 1) {
        return;
    }

    int pos = y * (env->grid_size + 1) + x;
    if (env->visited[pos] || env->board_states[pos] != 0) {
        return;
    }
    env->visited[pos] = 1;
    territory[player]++;
    // Check adjacent positions
    for (int i = 0; i < 4; i++) {
        flood_fill(env, x + DIRECTIONS[i][0], y + DIRECTIONS[i][1], territory, player);
    }
}

void compute_score_tromp_taylor(CGo* env) {
    int player_score = 0;
    int opponent_score = 0;
    int territory[3] = {0, 0, 0}; // [neutral, player, opponent]
    reset_visited(env);
    // Count stones and mark them as visited
    for (int i = 0; i < (env->grid_size + 1) * (env->grid_size + 1); i++) {
        env->visited[i] = 0;
        if (env->board_states[i] == 1) {
            player_score++;
            env->visited[i] = 1;
        } else if (env->board_states[i] == 2) {
            opponent_score++;
            env->visited[i] = 1;
        }
    }
    for (int y = 0; y < env->grid_size + 1; y++) {
        for (int x = 0; x < env->grid_size + 1; x++) {
            int pos = y * (env->grid_size + 1) + x;
            if (env->visited[pos]) {
                continue;
            }
            int player = 0; // Start as neutral
            // Check adjacent positions to determine territory owner
            for (int i = 0; i < 4; i++) {
                int nx = x + DIRECTIONS[i][0];
                int ny = y + DIRECTIONS[i][1];
                if (nx < 0 || nx >= env->grid_size + 1 || ny < 0 || ny >= env->grid_size + 1) {
                    continue;
                }
                int npos = ny * (env->grid_size + 1) + nx;
                if (env->board_states[npos] == 0) {
                    continue;
                }
                if (player == 0) {
                    player = env->board_states[npos];
                } else if (player != env->board_states[npos]) {
                    player = 0; // Neutral if bordered by both players
                    break;
                }
            }
            flood_fill(env, x, y, territory, player);
        }
    }
    // Calculate final scores
    player_score += territory[1];
    opponent_score += territory[2];
    env->score = (float)player_score - (float)opponent_score - env->komi;
}

void enemy_move(CGo* env){
// in place shuffle
        int shuffle_positions[(env->grid_size+1)*(env->grid_size+1)];
        int num_positions = (env->grid_size+1)*(env->grid_size+1);
        // Initialize the array with all possible positions
        for (int i = 0; i < num_positions; i++) {
            shuffle_positions[i] = i;
        }
        // Fisher-Yates shuffle
        for (int i = num_positions - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = shuffle_positions[i];
            shuffle_positions[i] = shuffle_positions[j];
            shuffle_positions[j] = temp;
        }
        // Find the first legal move in the shuffled array
        int opponent_move = -1;
        for (int i = 0; i < num_positions; i++) {
            if (check_legal_placement(env, shuffle_positions[i], 2)) {
                opponent_move = shuffle_positions[i];
                break;
            }
        }
        if (opponent_move != -1) {
            env->board_states[opponent_move] = 2;
            env->moves_made++;
            check_capture_pieces(env, env->board_states, opponent_move, 0, env->disjoint_set);
        } else {
            env->dones[0] = 1;
        }
}

void greedy_move(CGo *env){

}
  
void reset(CGo* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->score = 0;
    for (int i = 0; i < (env->grid_size+1)*(env->grid_size+1); i++) {
        env->board_states[i] = 0;
        env->temp_board_states[i] = 0;
        env->visited[i] = 0;
        env->previous_board_state[i] = 0;
    }
    env->capture_count[0] = 0;
    env->capture_count[1] = 0;
    env->last_capture_position = -1;
    env->moves_made = 0;
    compute_observations(env);
}

void end_game(CGo* env){
    compute_score_tromp_taylor(env);
    if (env->score > 0) {
        env->rewards[0] = 1.0 ;
    }
    else if (env->score < 0) {
        env->rewards[0] = -1.0 ;
    }
    else {
        env->rewards[0] = 0.0;
    }
    env->log.score = env->score;
    env->log.games_played++;
    env->log.episode_return += env->rewards[0];
    add_log(env->log_buffer, &env->log);
    reset(env);
}

void step(CGo* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = (int)env->actions[0];

    if (env->log.episode_length > 100) {
        env->dones[0] = 1;
        end_game(env);
        compute_observations(env);
        return;
    }

    if(action == NOOP){
        env->rewards[0] -= 0.25;
        env->log.episode_return -= 0.25;
        enemy_move(env);
        if (env->dones[0] == 1) {
            end_game(env);
            return;
        }
        compute_observations(env);
        return;
    }
    if (action >= MOVE_MIN && action <= (env->grid_size+1)*(env->grid_size+1)) {
        memcpy(env->previous_board_state, env->board_states, sizeof(int) * (env->grid_size+1) * (env->grid_size+1));
        if (check_legal_placement(env, action-1, 1)) {
            env->board_states[action-1] = 1;
            // Update groups and liberties using the permanent disjoint set
            check_capture_pieces(env, env->board_states, action-1,0, env->disjoint_set);
            env->moves_made++;
            env->rewards[0] += 0.1;
            env->log.episode_return += 0.1;
        }
        else {
            env->rewards[0] -= 0.1;
            env->log.episode_return -= 0.1;

            compute_observations(env);
            return;
        }
        enemy_move(env);
        
    }

    if (env->moves_made >= (env->grid_size+1)*(env->grid_size+1)*2) {        
        env->dones[0] = 1;
    }

    if (env->dones[0] == 1) {
        end_game(env);
        return;
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

    InitWindow(width, height, "PufferLib Ray Go");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    client->puffers = LoadTexture("resources/puffers_128.png");
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
        int position_state = env->board_states[i];
        int row = i / (env->grid_size + 1);
        int col = i % (env->grid_size + 1);
        int x = col * env->grid_square_size;
        int y = row * env->grid_square_size;
        // Calculate the circle position based on the grid
        int circle_x = x + 100;
        int circle_y = y + 100;
        // if player draw circle tile for black 
        if (position_state == 1) {
            DrawCircle(circle_x, circle_y, env->grid_square_size / 2, BLACK);
            DrawCircleLines(circle_x, circle_y, env->grid_square_size / 2, BLACK);
        }
        // if enemy draw circle tile for white
        if (position_state == 2) {
            DrawCircle(circle_x, circle_y, env->grid_square_size / 2, WHITE);
            DrawCircleLines(circle_x, circle_y, env->grid_square_size / 2, WHITE);
        }
    }
    // show capture count for both players
    DrawText(TextFormat("Player 1 Capture Count: %d", env->capture_count[0]), env->width - 300, 110, 20, WHITE);
    DrawText(TextFormat("Player 2 Capture Count: %d", env->capture_count[1]), env->width - 300, 130, 20, WHITE);
    EndDrawing();
}
void close_client(Client* client) {
    CloseWindow();
    free(client);
}
