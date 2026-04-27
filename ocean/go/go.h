#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"

#define NOOP 0
#define MOVE_MIN 1
#define TICK_RATE 1.0f/60.0f
#define NUM_DIRECTIONS 4
#define ENV_WIN -1
#define PLAYER_WIN 1
#define MAX_CHANGED_PER_MOVE 362
static const int DIRECTIONS[NUM_DIRECTIONS][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
//  LD_LIBRARY_PATH=raylib/lib ./go

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float illegal_move_count;
    float legal_move_count;
    float pass_move_count;
    float white_wins;
    float black_wins;
};

typedef struct Group Group;
struct Group {
    int parent;
    int rank;
    int size;
    int liberties;
};

int find(Group* groups, int x) {
    if (groups[x].parent != x)
        groups[x].parent = find(groups, groups[x].parent);
    return groups[x].parent;
}

void union_groups(Group* groups, int pos1, int pos2) {
    pos1 = find(groups, pos1);
    pos2 = find(groups, pos2);
    
    if (pos1 == pos2) return;
    
    if (groups[pos1].rank < groups[pos2].rank) {
        groups[pos1].parent = pos2;
        groups[pos2].size += groups[pos1].size;
        groups[pos2].liberties += groups[pos1].liberties;
    } else if (groups[pos1].rank > groups[pos2].rank) {
        groups[pos2].parent = pos1;
        groups[pos1].size += groups[pos2].size;
        groups[pos1].liberties += groups[pos2].liberties;
    } else {
        groups[pos2].parent = pos1;
        groups[pos1].rank++;
        groups[pos1].size += groups[pos2].size;
        groups[pos1].liberties += groups[pos2].liberties;
    }
}

typedef struct Client Client;
typedef struct CGo CGo;
struct CGo {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Log log;
    float score;
    int num_agents;
    int width;
    int height;
    int* board_x;
    int* board_y;
    int board_width;
    int board_height;
    int grid_square_size;
    int grid_size;
    uint8_t* board_states;
    uint8_t* previous_board_state;
    int last_capture_position;
    int moves_made;
    int* capture_count;
    float komi;
    uint8_t* visited;
    uint8_t current_version;
    Group* groups;
    Group* temp_groups;
    float reward_move_pass;
    float reward_move_invalid;
    float reward_move_valid;
    float reward_player_capture;
    float reward_opponent_capture;
    float tick;
    int selfplay;
    int turn;
    int side;
    int legal_move_count;
    int illegal_move_count;
    int pass_move_count;
    int previous_move;
    int human_play;
    // undo stack
    int changed_pos[MAX_CHANGED_PER_MOVE];
    uint8_t old_board_values[MAX_CHANGED_PER_MOVE];
    int changed_count;
    int old_capture_count[2];
    float old_reward;
    float old_episode_return;
    unsigned int rng;
};

void add_log(CGo* env) {
    env->log.episode_length += env->tick;
    
    // Calculate perf as a win rate (1.0 if win, 0.0 if loss)
    float win_value = (env->score > 0) ? 1.0f : (env->score < 0) ? 0.0f : 0.5f;
    float black_win = 0.0;
    float white_win = 0.0;
    if(env->score > 0){
        if(env->side == 1){
            black_win = 1.0;
        }
        else{
            white_win = 1.0;
        }
    }
    else if (env->score < 0){
        if(env->side == 1){
            white_win = 1.0;
        }
        else{
            black_win = 1.0; 
        }
    } else {
        black_win = 0.5;
        white_win = 0.5;
    }
    env->log.illegal_move_count += env->illegal_move_count;
    env->log.legal_move_count += env->legal_move_count;
    env->log.pass_move_count += env->pass_move_count;
    env->log.perf += win_value;
    env->log.black_wins += black_win;
    env->log.white_wins += white_win;
    env->log.score += env->score;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0;
}

void generate_board_positions(CGo* env) {
    for (int i = 0; i < (env->grid_size-1) * (env->grid_size-1); i++) {
        int row = i / (env->grid_size-1);
        int col = i % (env->grid_size-1);
        env->board_x[i] = col * (env->grid_square_size-1);
        env->board_y[i] = row * (env->grid_square_size-1);
    }
}

void init_groups(CGo* env) {
    for (int i = 0; i < (env->grid_size)*(env->grid_size); i++) {
        env->groups[i].parent = i;
        env->groups[i].rank = 0;
        env->groups[i].size = 1;
        env->groups[i].liberties = 0;
    }
}

void init(CGo* env) {
    int board_render_size = (env->grid_size-1)*(env->grid_size-1);
    int grid_size = env->grid_size*env->grid_size;
    env->board_x = (int*)calloc(board_render_size, sizeof(int));
    env->board_y = (int*)calloc(board_render_size, sizeof(int));
    env->board_states = (uint8_t*)calloc(grid_size, sizeof(uint8_t));
    env->visited = (uint8_t*)calloc(grid_size, sizeof(uint8_t)); 
    env->current_version = 1; 
    env->previous_board_state = (uint8_t*)calloc(grid_size, sizeof(uint8_t));
    env->capture_count = (int*)calloc(2, sizeof(int));
    env->groups = (Group*)calloc(grid_size, sizeof(Group));
    env->temp_groups = (Group*)calloc(grid_size, sizeof(Group));
    generate_board_positions(env);
    init_groups(env);
}

void allocate(CGo* env) {
    init(env);
    if(env->selfplay){
        env->observations = (float*)calloc(2*((env->grid_size)*(env->grid_size)*4 +2), sizeof(float));
        env->actions = (float*)calloc(2, sizeof(float));
    } else{
	// +2 correct?
        env->observations = (float*)calloc((env->grid_size)*(env->grid_size)*4 +2, sizeof(float));
        env->actions = (float*)calloc(1, sizeof(float));
    }
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_close(CGo* env) {
    free(env->board_x);
    free(env->board_y);
    free(env->board_states);
    free(env->visited);
    free(env->previous_board_state);
    free(env->capture_count);
    free(env->temp_groups);
    free(env->groups);
}

void free_allocated(CGo* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

static inline void increment_version(CGo* env) {
    env->current_version++;
    if (env->current_version == 0) { 
        memset(env->visited, 0, (env->grid_size) * (env->grid_size));
        env->current_version = 1;
    }
}

void compute_observations(CGo* env) {
    int obs_len = env->grid_size * env->grid_size * 4 + 2;
    int N = env->grid_size * env->grid_size;
    int iterations = env->selfplay ? 2 : 1;

    for(int i = 0; i < iterations; i++){
        float* current_obs = env->observations + (i * obs_len);
        
        int self, opp;
        if (i == 0) {
            self = env->side;
            opp = 3 - self;
        } else {
            // Flip perspective for selfplay
            self = 3 - env->side;
            opp = env->side;
        }
        int turn = env->turn + 1 == self ? 1 : 0;

        // Memory Layout: [Current Self][Current Opp][Prev Self][Prev Opp]
        float* plane_self      = current_obs;
        float* plane_opp       = current_obs + N;
        float* plane_prev_self = current_obs + (2 * N);
        float* plane_prev_opp  = current_obs + (3 * N);

        for (int idx = 0; idx < N; idx++) {
            int val = env->board_states[idx];
            int prev_val = env->previous_board_state[idx];

            plane_self[idx]      = (float)(val == self);
            plane_opp[idx]       = (float)(val == opp);
            
            plane_prev_self[idx] = (float)(prev_val == self);
            plane_prev_opp[idx]  = (float)(prev_val == opp);
        }

        // Set the color bit at the very end
        current_obs[4 * N] = (float)(self - 1);
        current_obs[4 * N + 1] = (float)(turn);
    } 
}

int is_valid_position(CGo* env, int x, int y) {
    return (x >= 0 && x < env->grid_size && y >= 0 && y < env->grid_size);
}


void flood_fill(CGo* env, int x, int y, int* territory, int player) {
    if (!is_valid_position(env, x, y)) {
        return;
    }

    int pos = y * (env->grid_size) + x;
    if (env->visited[pos] == env->current_version || env->board_states[pos] != 0) {
        return;
    }
    env->visited[pos] = env->current_version;
    territory[player]++;
    // Check adjacent positions
    for (int i = 0; i < 4; i++) {
        flood_fill(env, x + DIRECTIONS[i][0], y + DIRECTIONS[i][1], territory, player);
    }
}

void compute_score_tromp_taylor(CGo* env) {
    int player_score = 0;
    int opponent_score = 0;
    int player = env->side;
    int opponent = 3 - player;
    increment_version(env); 
    // Queue for BFS
    int queue_size = (env->grid_size) * (env->grid_size);
    int queue[queue_size];
    
    // First count stones
    for (int i = 0; i < queue_size; i++) {
        if (env->board_states[i] == player) {
            player_score++;
        } else if (env->board_states[i] == opponent) {
            opponent_score++;
        }
    }
    
    // Then process empty territories
    for (int start_pos = 0; start_pos < queue_size; start_pos++) {
        // Skip if not empty or already visited
        if (env->board_states[start_pos] != 0 || env->visited[start_pos] == env->current_version) {
            continue;
        }
        
        // Initialize BFS
        int front = 0, rear = 0;
        int territory_size = 0;
        int bordering_player = 0;  // 0=neutral, 1=player1, 2=player2, 3=mixed
        
        queue[rear++] = start_pos;
        env->visited[start_pos] = env->current_version;
        
        // Process connected empty points
        while (front < rear) {
            int pos = queue[front++];
            territory_size++;
            int x = pos % env->grid_size;
            int y = pos / env->grid_size;
            
            // Check all adjacent positions
            for (int i = 0; i < 4; i++) {
                int nx = x + DIRECTIONS[i][0];
                int ny = y + DIRECTIONS[i][1];
                
                if (!is_valid_position(env, nx, ny)) {
                    continue;
                }
                
                int npos = ny * env->grid_size + nx;
                int neighbor_color = env->board_states[npos];
                if (neighbor_color ==0) {
                    // Add unvisited empty points to queue
                    if(env->visited[npos] != env->current_version) {
                        queue[rear++] = npos;
                        env->visited[npos] = env->current_version;
                    }
                } else if (bordering_player == 0) {
                    bordering_player = neighbor_color;
                } else if (bordering_player != neighbor_color) {
                    bordering_player = 3;  // Mixed territory
                }
            }
        }
        
        // Assign territory points
        if (bordering_player == player) {
            player_score += territory_size;
        } else if (bordering_player == opponent) {
            opponent_score += territory_size;
        }
        // Mixed territories (bordering_player == 3) are neutral and not counted
    }
    float komi = (env->side == 2) ? env->komi : -env->komi;
    env->score = (float)player_score - (float)opponent_score + komi;
    //printf("Score: %f\n", env->score);
}

int find_in_group(int* group, int group_size, int value) {
    for (int i = 0; i < group_size; i++) {
        if (group[i] == value) {
            return 1;  // Found
        }
    }
    return 0;  // Not found
}


void capture_group(CGo* env, uint8_t* board, int root, int* affected_groups, int* affected_count) {
    increment_version(env);
    // Use a queue for BFS
    int queue_size = (env->grid_size) * (env->grid_size);
    int queue[queue_size];
    int front = 0, rear = 0;

    int captured_player = board[root];       // Player whose stones are being captured
    if (captured_player != 1 && captured_player !=2) return; 
    int capturing_player = 3 - captured_player;          // Player who captures

    queue[rear++] = root;
    env->visited[root] = env->current_version;

    while (front != rear) {
        int pos = queue[front++];
        env->old_board_values[env->changed_count] = board[pos];  // captured_player
        env->changed_pos[env->changed_count] = pos;
        env->changed_count++;
        board[pos] = 0;  // Remove stone
        env->capture_count[capturing_player - 1]++;  // Update capturing player's count
	if(capturing_player == env->side){
		env->rewards[0] += env->reward_player_capture;
		env->log.episode_return += env->reward_player_capture;
	} else{
		env->rewards[0] += env->reward_opponent_capture;
		env->log.episode_return += env->reward_opponent_capture;
	}
        int x = pos % (env->grid_size);
        int y = pos / (env->grid_size);

        for (int i = 0; i < 4; i++) {
            int nx = x + DIRECTIONS[i][0];
            int ny = y + DIRECTIONS[i][1];
            int npos = ny * (env->grid_size) + nx;

            if (!is_valid_position(env, nx, ny)) {
                continue;
            }

            if (board[npos] == captured_player && env->visited[npos]!=env->current_version) {
                env->visited[npos] = env->current_version;
                queue[rear++] = npos;
            }
            else if (board[npos] == capturing_player) {
                int adj_root = find(env->temp_groups, npos);
                if (find_in_group(affected_groups, *affected_count, adj_root)) {
                    continue;
                }
                affected_groups[(*affected_count)] = adj_root;
                (*affected_count)++;
            }
        }
    }
}


int count_liberties(CGo* env, int root, int* queue, uint8_t* board) {
    increment_version(env);
    int liberties = 0;
    int front = 0;
    int rear = 0;
    
    queue[rear++] = root;
    env->visited[root] = env->current_version;
    while (front < rear) {
        int pos = queue[front++];
        int x = pos % (env->grid_size);
        int y = pos / (env->grid_size);
        
        for (int i = 0; i < 4; i++) {
            int nx = x + DIRECTIONS[i][0];
            int ny = y + DIRECTIONS[i][1];
            if (!is_valid_position(env, nx, ny)) {
                continue;
            }
            
            int npos = ny * (env->grid_size) + nx;
            if (env->visited[npos]== env->current_version) {
                continue;
            }

            int temp_npos = board[npos];
            if (temp_npos == 0) {
                liberties++;
                env->visited[npos] = env->current_version;
            } else if (temp_npos == board[root]) {
                queue[rear++] = npos;
                env->visited[npos] = env->current_version;
            }
        }
    }
    return liberties;
}

int make_move(CGo* env, int pos, int player){
    int x = pos % (env->grid_size);
    int y = pos / (env->grid_size);
    // cannot place stone on occupied tile
    if (env->board_states[pos] != 0) {
        if(player == env->side){
            env->illegal_move_count+=1;
        }
        return 0 ;
    }
    env->old_capture_count[0] = env->capture_count[0];
    env->old_capture_count[1] = env->capture_count[1];
    env->old_reward = env->rewards[0];
    env->old_episode_return = env->log.episode_return;

    env->changed_count = 0;

    env->old_board_values[env->changed_count] = env->board_states[pos];
    env->changed_pos[env->changed_count++] = pos;
    env->board_states[pos] = player;
    // temp structures
    memcpy(env->temp_groups, env->groups, sizeof(Group) * (env->grid_size) * (env->grid_size));
    // create new group
    env->temp_groups[pos].parent = pos;
    env->temp_groups[pos].rank = 0;
    env->temp_groups[pos].size = 1;
    env->temp_groups[pos].liberties = 0;
    
    int max_affected_groups = (env->grid_size) * (env->grid_size);
    int affected_groups[max_affected_groups];
    int affected_count = 0;
    affected_groups[affected_count++] = pos;

    int queue[(env->grid_size) * (env->grid_size)];

    // Perform unions and track affected groups
    for (int i = 0; i < 4; i++) {
        int nx = x + DIRECTIONS[i][0];
        int ny = y + DIRECTIONS[i][1];
        int npos = ny * (env->grid_size) + nx;
        if (!is_valid_position(env, nx, ny)) {
            continue;
        }
        if (env->board_states[npos] == player) {
            union_groups(env->temp_groups, pos, npos);
            affected_groups[affected_count++] = npos;
        } else if (env->board_states[npos] == 3 - player) {
            affected_groups[affected_count++] = npos;
        }
    }

    // Recalculate liberties only for affected groups
    for (int i = 0; i < affected_count; i++) {
        int root = find(env->temp_groups, affected_groups[i]);
        env->temp_groups[root].liberties = count_liberties(env, root, queue, env->board_states);
    }

    // Check for captures
    bool captured = false;
    for (int i = 0; i < affected_count; i++) {
        int root = find(env->temp_groups, affected_groups[i]);
        if (env->board_states[root] == 3 - player && env->temp_groups[root].liberties == 0) {
            capture_group(env, env->board_states, root, affected_groups, &affected_count);
            captured = true;
        }
    }
    // If captures occurred, recalculate liberties again
    if (captured) {
        for (int i = 0; i < affected_count; i++) {
            int root = find(env->temp_groups, affected_groups[i]);
            env->temp_groups[root].liberties = count_liberties(env, root, queue, env->board_states);
        }
        // Check for ko rule violation
    }

    // self capture
    int root = find(env->temp_groups, pos);
    if (env->temp_groups[root].liberties == 0) {
        goto rollback;
    }

    if(captured && memcmp(env->board_states, env->previous_board_state, env->grid_size*env->grid_size*sizeof(uint8_t)) == 0){
        goto rollback;
    }
    memcpy(env->previous_board_state, env->board_states, sizeof(uint8_t) * (env->grid_size) * (env->grid_size));
    memcpy(env->groups, env->temp_groups, sizeof(Group) * (env->grid_size) * (env->grid_size));
    for(int i = 0; i < env->changed_count; i++){
        env->previous_board_state[env->changed_pos[i]] = env->old_board_values[i];
    }
    return 1;

rollback:
    for (int i = 0; i < env->changed_count; i++) {
        env->board_states[env->changed_pos[i]] = env->old_board_values[i];
    }
    env->capture_count[0] = env->old_capture_count[0];
    env->capture_count[1] = env->old_capture_count[1];
    env->rewards[0] = env->old_reward;
    env->log.episode_return = env->old_episode_return;

    if (player == env->side) env->illegal_move_count++;

    return 0;
}


void enemy_random_move(CGo* env, int side){
    int num_positions = (env->grid_size)*(env->grid_size);
    int positions[num_positions];
    int count = 0;

    // Collect all empty positions
    for(int i = 0; i < num_positions; i++){
        if(env->board_states[i] == 0){
            positions[count++] = i;
        }
    }
    // Shuffle the positions
    for(int i = count - 1; i > 0; i--){
        int j = rand_r(&env->rng) % (i + 1);
        int temp = positions[i];
        positions[i] = positions[j];
        positions[j] = temp;
    }
    // Try to make a move in a random empty position
    for(int i = 0; i < count; i++){
        if(make_move(env, positions[i], side)){
            env->previous_move = positions[i] + 1;
            return;
        }
    }
    // If no move is possible, pass or end the game
    env->previous_move = 0;
    env->terminals[0] = 1;
}

int find_group_liberty(CGo* env, int root){
    increment_version(env);
    int queue[(env->grid_size)*(env->grid_size)];
    int front = 0, rear = 0;
    queue[rear++] = root;
    env->visited[root] = env->current_version;

    while(front < rear){
        int pos = queue[front++];
        int x = pos % (env->grid_size);
        int y = pos / (env->grid_size);

        for(int i = 0; i < 4; i++){
            int nx = x + DIRECTIONS[i][0];
            int ny = y + DIRECTIONS[i][1];
            int npos = ny * (env->grid_size) + nx;
            if(!is_valid_position(env, nx, ny)){
                continue;
            }
            if(env->board_states[npos] == 0){
                return npos; // Found a liberty
            } else if(env->board_states[npos] == env->board_states[root] && env->visited[npos] != env->current_version){
                env->visited[npos] = env->current_version;
                queue[rear++] = npos;
            }
        }
    }
    return -1; // Should not happen if liberties > 0
}

void enemy_greedy_hard(CGo* env, int side){

    int opp = 3 - side;
	// Attempt to capture opponent stones in atari
    int liberties[4][(env->grid_size) * (env->grid_size)];
    int liberty_counts[4] = {0};
    for(int i = 0; i < (env->grid_size)*(env->grid_size); i++){
        if(env->board_states[i]==0){
                continue;
        }
        if (env->board_states[i]==opp){
            int root = find(env->groups, i);
            int group_liberties = env->groups[root].liberties;
            if (group_liberties >= 1 && group_liberties <= 4) {
                int liberty = find_group_liberty(env, root);
                liberties[group_liberties - 1][liberty_counts[group_liberties - 1]++] = liberty;
            }
        } else if (env->board_states[i]==side){
            int root = find(env->groups, i);
            int group_liberties = env->groups[root].liberties;
            if (group_liberties==1) {
                int liberty = find_group_liberty(env, root);
                liberties[group_liberties - 1][liberty_counts[group_liberties - 1]++] = liberty;
            }
        }
    }
    // make move to attack or defend
    for (int priority = 0; priority < 4; priority++) {
        for (int i = 0; i < liberty_counts[priority]; i++) {
            if (make_move(env, liberties[priority][i], side)) {
                env->previous_move = liberties[priority][i]+1;
                return;
            }
        }
    }

    // random move
    enemy_random_move(env, side);
}

void enemy_greedy_easy(CGo* env, int side){
    // Attempt to capture opponent stones in atari
    for(int i = 0; i < (env->grid_size)*(env->grid_size); i++){
        if(env->board_states[i] != 1){
            continue;
        }
        int root = find(env->groups, i);
        if(env->groups[root].liberties == 1){
            int liberty = find_group_liberty(env, root);
            if(make_move(env, liberty, 2)){
                return; // Successful capture
            }
        }
    }
    // Protect own stones in atari
    for(int i = 0; i < (env->grid_size)*(env->grid_size); i++){
        if(env->board_states[i] != 2){
            continue;
        }
        // Enemy's own stones
        int root = find(env->groups, i);
        if(env->groups[root].liberties == 1){
            int liberty = find_group_liberty(env, root);
            if(make_move(env, liberty, 2)){
                return; // Successful defense
            }
        }
    }
    // Play a random legal move
    enemy_random_move(env, side);
}

void c_reset(CGo* env) {
    env->tick = 0;
    env->illegal_move_count = 0;
    env->legal_move_count = 0;
    env->pass_move_count = 0;
    env->turn = 0;
    env->previous_move = -1;
    // We don't reset the log struct - leave it accumulating like in Pong
    env->score = 0;
    for (int i = 0; i < (env->grid_size)*(env->grid_size); i++) {
        env->board_states[i] = 0;
        env->visited[i] = 0;
        env->previous_board_state[i] = 0;
        env->groups[i].parent = i;
        env->groups[i].rank = 0;
        env->groups[i].size = 0;
        env->groups[i].liberties = 0;
    }
    env->capture_count[0] = 0;
    env->capture_count[1] = 0;
    env->last_capture_position = -1;
    env->moves_made = 0;
    compute_observations(env);
}

void clip_rewards(CGo* env){
    if(env->rewards[0] > 1){
	    env->rewards[0] = 1;
    } 
    if(env->rewards[0] < -1){
	    env->rewards[0] = -1;
    }
}

void end_game(CGo* env){
    compute_score_tromp_taylor(env);
    if (env->score > 0) {
        env->rewards[0] = 1.0;
    }
    else if (env->score < 0) {
        env->rewards[0] = -1.0;
    }
    else {
        env->rewards[0] = 0.0;
    }
    //env->rewards[0] = env->score / 10.0f;
    clip_rewards(env);
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
}

void human_play(CGo* env){
    int indx=1;
    if(!env->selfplay || !env->human_play){
        return;
    }
    if(env->selfplay && env->turn + 1 != env->side){
        env->actions[indx] = -1;
    }
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        Vector2 mousePos = GetMousePosition();

        // Calculate the offset for the board
        int boardOffsetX = env->grid_square_size;
        int boardOffsetY = env->grid_square_size;
        
        // Adjust mouse position relative to the board
        int relativeX = mousePos.x - boardOffsetX;
        int relativeY = mousePos.y - boardOffsetY;
        
        // Calculate cell indices for the corners
        int cellX = (relativeX + env->grid_square_size / 2) / env->grid_square_size;
        int cellY = (relativeY + env->grid_square_size / 2) / env->grid_square_size;
        
        // Ensure the click is within the game board
        if (cellX >= 0 && cellX <= env->grid_size && cellY >= 0 && cellY <= env->grid_size) {
            // Calculate the point index (1-19) based on the click position
            int pointIndex = cellY * (env->grid_size) + cellX + 1; 
            env->actions[indx] = (unsigned short)pointIndex;
        }
        // Check if pass button is clicked
        int left = (env->grid_size + 1)*env->grid_square_size;
        int top = env->grid_square_size;
        int passButtonX = left;
        int passButtonY = top + 90;
        int passButtonWidth = 100;
        int passButtonHeight = 50;

        if (mousePos.x >= passButtonX && mousePos.x <= passButtonX + passButtonWidth &&
            mousePos.y >= passButtonY && mousePos.y <= passButtonY + passButtonHeight) {
            env->actions[indx] = 0; // Send action 0 for pass
        }
    }

}

void c_step(CGo* env) {
    env->tick += 1;
    env->rewards[0] = 0.0;
    env->terminals[0] = 0;
    int action = 0;
    int bot_side = 3 - env->side; 
    int is_legal = 0;
    if(env->human_play){
        human_play(env);
    }
    if(env->selfplay){
        action = (env->turn +1 == env->side) ? (int)env->actions[0] : (int)env->actions[1];
    } else {
        action = (int)env->actions[0];
    }
    if(action == -1){
        compute_observations(env);
        return;
    }
    // useful for training , can prob be a hyper param. Recommend to increase with larger board size
    float max_moves = 3 * env->grid_size * env->grid_size;
    if (env->tick > max_moves && !env->human_play) {
         env->terminals[0] = 1;
         end_game(env);
         compute_observations(env);
         return;
    }
    // play against bots 
    if(!env->selfplay && env->turn == (bot_side - 1)){
        enemy_greedy_hard(env, bot_side);
        if (env->terminals[0] == 1) {
            end_game(env);
        }
        compute_observations(env);
        clip_rewards(env);
        env->turn = (env->turn + 1) % 2;
        return;
    }
    // process action
    if(action == NOOP){
        if(env->turn + 1 == env->side){
            //printf("Pass\n");
            env->legal_move_count +=1;
            env->rewards[0] = env->reward_move_pass;
            env->log.episode_return += env->reward_move_pass;
            env->pass_move_count += 1;
        }
        if (env->terminals[0] == 1 || env->previous_move == NOOP) {
            end_game(env);
            return;
        }
        env->previous_move = NOOP;
        env->turn = (env->turn+1)%2;
        compute_observations(env);
        return;
    }
    if (action >= MOVE_MIN && action <= (env->grid_size)*(env->grid_size)) {
        is_legal = make_move(env, action - 1, env->turn + 1); 
        if(is_legal) {
            env->moves_made++;
            if(env->turn + 1 == env->side){
                env->legal_move_count +=1;
                env->rewards[0] += env->reward_move_valid;
                env->log.episode_return += env->reward_move_valid;
            }
        } else {
            if(env->turn + 1 == env->side){
                env->rewards[0] = env->reward_move_invalid;
                env->log.episode_return += env->reward_move_invalid;
            }
        }
    }
    env->previous_move = action;

    if (env->terminals[0] == 1) {
        end_game(env);
        return;
    }
    if(is_legal){
        env->turn = (env->turn + 1) % 2;
    }
    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

struct Client {
    float width;
    float height;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ray Go");
    SetTargetFPS(10);
    return client;
}


void c_render(CGo* env) {
    if (env->client == NULL) {
        env->client = make_client(env->width, env->height);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int board_size = (env->grid_size + 1) * env->grid_square_size;
    DrawRectangle(0, 0, board_size, board_size, PUFF_BACKGROUND);
    DrawRectangle(
        env->grid_square_size,
        env->grid_square_size,
        board_size - 2*env->grid_square_size,
        board_size - 2*env->grid_square_size,
        PUFF_BACKGROUND2
    );
    int start = env->grid_square_size;
    int end = board_size - env->grid_square_size;
    for (int i = 1; i <= env->grid_size; i++) {
        DrawLineEx(
            (Vector2){start, i*start},
            (Vector2){end, i*start},
            4, PUFF_BACKGROUND
        );
        DrawLineEx(
            (Vector2){i*start, start},
            (Vector2){i*start, end},
            4, PUFF_BACKGROUND
        );
    }

    for (int i = 0; i < (env->grid_size) * (env->grid_size); i++) {
        int position_state = env->board_states[i];
        int row = i / (env->grid_size);
        int col = i % (env->grid_size);
        int x = col * env->grid_square_size;
        int y = row * env->grid_square_size;
        // Calculate the circle position based on the grid
        int circle_x = x + env->grid_square_size;
        int circle_y = y + env->grid_square_size;
        // if player draw circle tile for black 
        int inner = (env->grid_square_size / 2) - 4;
        int outer = (env->grid_square_size / 2) - 2;
        if (position_state == 1) {
            DrawCircleGradient(circle_x, circle_y, outer, STONE_GRAY, BLACK);
        }
        // if enemy draw circle tile for white
        if (position_state == 2) {
            DrawCircleGradient(circle_x, circle_y, inner, WHITE, GRAY);
        }
    }
    // design a pass button
    int left = (env->grid_size + 1)*env->grid_square_size;
    int top = env->grid_square_size;
    DrawRectangle(left, top + 90, 100, 50, GRAY);
    DrawText("Pass", left + 25, top + 105, 20, PUFF_WHITE);
    DrawText(
        TextFormat("Tick: %d", (int)env->tick),
        left, top + 150, 20, PUFF_WHITE
    );
    if(env->side == 1){
        DrawText(
            TextFormat("Agent: black"),
            left, top + 170, 20, PUFF_WHITE
        );
    }
    else {
        DrawText(
            TextFormat("Agent: white"),
            left, top + 170, 20, PUFF_WHITE
        );
    }
    DrawText(
        TextFormat("Black Capture Count: %d", env->capture_count[0]),
        left, top, 20, PUFF_WHITE
    );
    DrawText(
        TextFormat("White Capture Count: %d", env->capture_count[1]),
        left, top + 40, 20, PUFF_WHITE
    );
    EndDrawing();
}
void close_client(Client* client) {
    CloseWindow();
    free(client);
}
