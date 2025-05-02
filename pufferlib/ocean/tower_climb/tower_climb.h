#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION            330
#else   // PLATFORM_ANDROID, PLATFORM_WEB
    #define GLSL_VERSION            100

#endif
#define RLIGHTS_IMPLEMENTATION

#include "rlights.h"

#define NOOP -1
#define UP 3
#define LEFT 2
#define RIGHT 0
#define DOWN 1
#define GRAB 4
#define DROP 5
// robot state
#define DEFAULT 0
#define HANGING 1
// observation space
#define PLAYER_OBS 3
#define OBS_VISION 225
// PLG VS ENV
#define PLG_MODE 0
#define RL_MODE 1
//logs
#define LOG_BUFFER_SIZE 1024
// level size
#define row_max 10
#define col_max 10
#define depth_max 10
// block bytes
#define BLOCK_BYTES 125
// FNV-1a hash function
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL
// moves
#define MOVE_ILLEGAL 0
#define MOVE_SUCCESS 1
#define MOVE_DEATH 2
// bitmask operations
#define SET_BIT(mask, i)    ( (mask)[(i)/8] |=  (1 << ((i)%8)) )
#define CLEAR_BIT(mask, i)  ( (mask)[(i)/8] &= ~(1 << ((i)%8)) )
#define TEST_BIT(mask, i)   ( ((mask)[(i)/8] & (1 << ((i)%8))) != 0 )

// BFS
#define MAX_BFS_SIZE 70000000
#define MAX_NEIGHBORS 6 // based on action space

// hash table 
#define TABLE_SIZE 70000003

// direction vectors
#define NUM_DIRECTIONS 4
static const int BFS_DIRECTION_VECTORS_X[NUM_DIRECTIONS] = {1, 0, -1, 0};
static const int BFS_DIRECTION_VECTORS_Z[NUM_DIRECTIONS] = {0, 1, 0, -1};
// shimmy wrap constants 
static const int wrap_x[4][2] = {
    {1,1},
    {1,-1},
    {-1,-1},
    {-1, 1}
};
static const int wrap_z[4][2] = {
    {-1, 1},
    {1, 1},
    {1, -1},
    {-1, -1}
};
static const int wrap_orientation[4][2] = {
    {DOWN,UP},
    {LEFT, RIGHT},
    {UP, DOWN},
    {RIGHT, LEFT}
};

typedef struct Level Level;
struct Level {
    int* map;
    int rows;
    int cols;
    int size;
    int total_length;
    int goal_location;
    int spawn_location;
};

void init_level(Level* lvl){
	lvl->map = calloc(1000,sizeof(unsigned int));
    lvl->rows = 10;
    lvl->cols = 10;
    lvl->size = 100;
    lvl->total_length = 1000;
    lvl->goal_location = 999;
    lvl->spawn_location = 0;
}

void reset_level(Level* lvl){
    lvl->goal_location = 999;
    lvl->spawn_location = 0;
    memset(lvl->map, 0, 1000 * sizeof(unsigned int));
}

void free_level(Level* lvl){
	free(lvl->map);
	free(lvl);
}

typedef struct PuzzleState PuzzleState;
struct PuzzleState {
    unsigned char* blocks;
    int robot_position;
    int robot_orientation;
    int robot_state;
    int block_grabbed;
};

void init_puzzle_state(PuzzleState* ps){
	ps->blocks = calloc(BLOCK_BYTES, sizeof(unsigned char));
}

void free_puzzle_state(PuzzleState* ps){
	free(ps->blocks);
	free(ps);
}

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
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
    if (logs->idx == 0) return log;  // Avoid division by zero

    for (int i = 0; i < logs->idx; i++) {
        log.episode_return  += logs->logs[i].episode_return  / logs->idx;
        log.episode_length  += logs->logs[i].episode_length  / logs->idx;
        log.score += logs->logs[i].score / logs->idx;
	    log.perf += logs->logs[i].perf / logs->idx;
    }

    logs->idx = 0;
    return log;
}

typedef struct CTowerClimb CTowerClimb;
struct CTowerClimb {
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    float score;
    Level* level;
    PuzzleState* state;  // Contains blocks bitmask, position, orientation, etc.
    int rows_cleared;
    float reward_climb_row;
    float reward_fall_row;
    float reward_illegal_move;
    float reward_move_block;
};

void levelToPuzzleState(Level* level, PuzzleState* state) {
    memset(state->blocks, 0, BLOCK_BYTES);
    for (int i = 0; i < level->total_length; i++) {
        if (level->map[i] == 1) {
            SET_BIT(state->blocks, i);
        }
    }
    state->robot_position = level->spawn_location;
    state->robot_orientation = UP;  
    state->robot_state = 0;        
    state->block_grabbed = -1;    
}

void init(CTowerClimb* env) {
	env->level = calloc(1, sizeof(Level));
    env->state = calloc(1, sizeof(PuzzleState));	
    init_level(env->level);
    init_puzzle_state(env->state);
    env->rows_cleared = 0;
}

void setPuzzle(CTowerClimb* env, PuzzleState* src, Level* lvl){
	memcpy(env->state->blocks, src->blocks, BLOCK_BYTES * sizeof(unsigned char));
	env->state->robot_position = src->robot_position;
	env->state->robot_orientation = src->robot_orientation;
	env->state->robot_state = src->robot_state;
	env->state->block_grabbed = src->block_grabbed; 
    memcpy(env->level->map, lvl->map, lvl->total_length * sizeof(int));
    env->level->rows = lvl->rows;
    env->level->cols = lvl->cols;
    env->level->size = lvl->size;
    env->level->total_length = lvl->total_length;
    env->level->goal_location = lvl->goal_location;
    env->level->spawn_location = lvl->spawn_location;
}

CTowerClimb* allocate() {
    CTowerClimb* env = (CTowerClimb*)calloc(1, sizeof(CTowerClimb));
    init(env);
    env->observations = (unsigned char*)calloc(OBS_VISION+PLAYER_OBS, sizeof(unsigned char));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    return env;
}

void free_initialized(CTowerClimb* env) {
	free_level(env->level);
	free_puzzle_state(env->state);
}

void free_allocated(CTowerClimb* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
    free(env);
}

void calculate_window_bounds(int* bounds, int center_pos, int window_size, int max_size) {
    int half_size = window_size / 2;
    // Try to center on position
    bounds[0] = center_pos - half_size;  // start
    bounds[1] = bounds[0] + window_size; // end
    // Adjust if window is larger than max size
    if (window_size > max_size) {
        bounds[0] = 0;
        bounds[1] = max_size;
    }
    // Adjust if too close to start
    else if (bounds[0] < 0) {
        bounds[0] = 0;
        bounds[1] = window_size;
    }
    // Adjust if too close to end
    else if (bounds[1] > max_size) {
        bounds[1] = max_size;
        bounds[0] = bounds[1] - window_size;
        if (bounds[0] < 0) bounds[0] = 0;
    }
}

void compute_observations(CTowerClimb* env) {
    int sz = env->level->size;
    int cols = env->level->cols;
    int rows = env->level->rows;
    int max_floors = env->level->total_length / sz;
    // Get player position
    int current_floor = env->state->robot_position / sz;
    int grid_pos = env->state->robot_position % sz;
    int player_x = grid_pos % cols;
    int player_z = grid_pos / cols;
    // Calculate window bounds using the new function
    int y_bounds[2], x_bounds[2], z_bounds[2];
    calculate_window_bounds(y_bounds, current_floor + 1, 5, max_floors);
    calculate_window_bounds(x_bounds, player_x, 9, cols);
    calculate_window_bounds(z_bounds, player_z, 5, rows);
    // Fill in observations
    for (int y = 0; y < 5; y++) {
        int world_y = y + y_bounds[0];
        for (int z = 0; z < 5; z++) {
            int world_z = z + z_bounds[0];
            for (int x = 0; x < 9; x++) {
                int world_x = x + x_bounds[0];
                int obs_idx = x + z * 9 + y * (9 * 5);
                // Check if position is out of bounds
                int board_idx = world_y * sz + world_z * cols + world_x;
                // Position is in bounds, set observation
                if (board_idx == env->state->robot_position) {
                    env->observations[obs_idx] = 3;
                    continue;
                }
                else if (board_idx == env->level->goal_location){
                    env->observations[obs_idx] = 2;
                    continue;
                }
                // Use bitmask directly instead of board_state array
                env->observations[obs_idx] = TEST_BIT(env->state->blocks, board_idx);
            }
        }
    }
    // Add player state information at the end
    int state_start = 9 * 5 * 5;
    env->observations[state_start] = env->state->robot_orientation;
    env->observations[state_start + 1] = env->state->robot_state;
    env->observations[state_start + 2] = (env->state->block_grabbed != -1);
}

void c_reset(CTowerClimb* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    env->rows_cleared = 0;
    memset(env->state->blocks, 0, BLOCK_BYTES * sizeof(unsigned char));
    compute_observations(env);
}

void illegal_move(CTowerClimb* env){
    env->rewards[0] = env->reward_illegal_move;
    env->log.episode_return += env->reward_illegal_move;
}

void death(CTowerClimb* env){
	env->rewards[0] = -1;
	env->log.episode_return -= 1;
	env->log.perf = 0;
	add_log(env->log_buffer, &env->log);
}

int isGoal(  PuzzleState* s,  Level* lvl) {
    if (s->robot_position - lvl->size != lvl->goal_location) return 0;
    return 1;
}

int move(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int new_position = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*lvl->cols;
    outState->robot_position = new_position;
    return 1;
}

int climb(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int cell_direct_above = outState->robot_position + lvl->size;
    int cell_next_above = cell_direct_above + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*lvl->cols;
    int goal = lvl->goal_location;

    int in_bounds = cell_direct_above < lvl->total_length && cell_next_above < lvl->total_length;
    int cells_blocking_climb = TEST_BIT(outState->blocks, cell_direct_above) || TEST_BIT(outState->blocks, cell_next_above);
    int goal_blocking_climb = cell_direct_above == goal || cell_next_above == goal;
    int can_climb = in_bounds && !cells_blocking_climb && !goal_blocking_climb;
    
    if (!can_climb) return 0;
    int floor_cleared = (cell_direct_above / lvl->size) - 2;
    if(mode == RL_MODE && floor_cleared > env->rows_cleared){
        env->rows_cleared = floor_cleared;
        env->rewards[0] = env->reward_climb_row;
        env->log.episode_return += env->reward_climb_row;
        env->log.score = floor_cleared;
    }
    outState->robot_position = cell_next_above;
    outState->robot_state = 0;
    return 1;
}

int drop(PuzzleState* outState, int action, int mode, CTowerClimb* env, const Level* lvl){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action]*lvl->cols;
    int next_below_cell = next_cell - lvl->size;
    int next_double_below_cell = next_cell - 2*lvl->size;
    if (next_below_cell < 0) return 0;
    int step_down = next_double_below_cell >= 0 && TEST_BIT(outState->blocks, next_double_below_cell);
    if(mode == RL_MODE){
        env->rewards[0] = env->reward_fall_row;
        env->log.episode_return += env->reward_fall_row;
    }
    if (step_down){
        outState->robot_position = next_below_cell;
        return 1;
    } else {
        outState->robot_position = next_below_cell;
        outState->robot_orientation = (outState->robot_orientation + 2) % 4;
        outState->robot_state = 1;
        return 1;
    }
}

int drop_from_hang(PuzzleState* outState, int action, int mode, const Level* lvl){
    int below_cell = outState->robot_position - lvl->size;
    while(below_cell > lvl->size && !TEST_BIT(outState->blocks, below_cell)){
        below_cell -= lvl->size;
    }
    if (below_cell >= lvl->size) {
        outState->robot_position = below_cell+lvl->size;
        outState->robot_state = 0;
        return 1;
    }
    if (mode == PLG_MODE) return MOVE_ILLEGAL;
    return MOVE_DEATH;
}
static inline int bfs_is_valid_position(int pos, const Level* level) {
    return (pos >= 0 && pos < level->total_length);
}

// Helper function to check block stability
static int bfs_is_block_stable(const PuzzleState* state, int position, const Level* level) {
    const int fs = level->size;
    const int positions[] = {
        position - fs,              // Bottom
        position - fs - 1,          // Left
        position - fs + 1,          // Right
        position - fs - level->cols, // Front
        position - fs + level->cols  // Back
    };
    
    for (int i = 0; i < 5; i++) {
        if (bfs_is_valid_position(positions[i], level) && TEST_BIT(state->blocks, positions[i])) {
            return 1;
        }
    }
    return 0;
}

int will_fall(PuzzleState* outState, int position, const Level* lvl){
    int valid = bfs_is_valid_position(position, lvl);
    int block_or_goal = TEST_BIT(outState->blocks, position) || position == lvl->goal_location;
    int stable = bfs_is_block_stable(outState, position, lvl);
    return valid && block_or_goal && !stable;
}

int handle_block_falling(PuzzleState* outState, int* affected_blocks, int* blocks_to_move, int affected_blocks_count, const Level* lvl) {
    // Create queue for blocks that need to be checked for falling
    int bfs_queue[lvl->total_length];  // Max possible blocks to check
    int front = 0;
    int rear = 0;
    int fs = lvl->size;
    int cols = lvl->cols;

    // Add initially affected blocks to queue
    for (int i = 0; i < affected_blocks_count; i++) {
        if (affected_blocks[i] == -1){
            break;
        }
        bfs_queue[rear] = affected_blocks[i];
        rear++;
    }
    // First check all blocks above and adjacent to moved blocks
    for (int i = 0; i < lvl->cols; i++) {
        int block_pos = blocks_to_move[i];
        if (block_pos == -1){
            continue;
        }
        // Add block directly above
        int cell_above = block_pos + fs;  // Assuming 100 is floor height
        // If valid block above and unstable, add to queue
        if (will_fall(outState, cell_above, lvl)) {
            bfs_queue[rear++] = cell_above;
        }
        // Check edge-supported blocks
        int edge_blocks[4] = {
            cell_above - 1,      // left
            cell_above + 1,      // right
            cell_above - cols,     // front (assuming 10 is width)
            cell_above + cols      // back
        };
        // Add valid edge blocks to queue
        for (int j = 0; j < 4; j++) {
            if (will_fall(outState, edge_blocks[j], lvl)) {
                bfs_queue[rear++] = edge_blocks[j];
            }
        }
    }
    // Process queue until empty
    while (front < rear) {
        int current = bfs_queue[front++];
        int falling_position = current;
        int found_support = 0;
        // Check if block is goal (2)
        if (current == lvl->goal_location) {
            // Goal block is falling - level failed
            return 0;
        }
        // Remove block from current position
        CLEAR_BIT(outState->blocks, current);
        // Keep moving down until support found or bottom reached
        while (!found_support && falling_position >= fs) {  // 100 represents one floor down
            // Place block temporarily to check stability
            SET_BIT(outState->blocks, falling_position);            
            // Check if block is stable
            if (bfs_is_block_stable(outState, falling_position, lvl)) {
                found_support = 1;
            } else {
                // Remove block if not stable
                CLEAR_BIT(outState->blocks, falling_position);
                falling_position -= fs;  // Move down one level
            }
        }
        // Check blocks that might be affected by this fall
        int original_above = current + fs;  // Block directly above original position
        if (will_fall(outState, original_above, lvl)) {
            bfs_queue[rear++] = original_above;
        }
        // Check edge blocks that might be affected
        int edge_blocks[4] = {
            original_above - 1,   // Left
            original_above + 1,   // Right
            original_above - cols,  // Front (assuming 10 is the width)
            original_above + cols   // Back
        };
        for (int i = 0; i < 4; i++) {
            if (will_fall(outState, edge_blocks[i], lvl)) {
                bfs_queue[rear++] = edge_blocks[i];
            }
        }
    }
    return 1;
}

int push(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env, int block_offset){
    int first_block_index = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*lvl->cols;                          
    int* blocks_to_move = calloc(lvl->cols, sizeof(int));
    for(int i = 0; i < lvl->cols; i++) {
        blocks_to_move[i] = (i == 0) ? first_block_index : -1;
    }
    for (int i = 0; i < lvl->cols; i++){
        int b_address = first_block_index + i*block_offset;
        if(i!=0 && blocks_to_move[i-1] == -1){
            break;
        }
        if(TEST_BIT(outState->blocks, b_address)){
            blocks_to_move[i] = b_address;
        }
    }
    int affected_blocks[lvl->cols];
    int count = 0;
    for (int i = 0; i < lvl->cols; i++){
        // invert conditional 
        int b_index = blocks_to_move[i];
        if (b_index == -1){
            continue;
        }
        if(i==0){
            CLEAR_BIT(outState->blocks, b_index);
        }
        int grid_pos = b_index % lvl->size;
        int x = grid_pos % lvl->cols;
        int z = grid_pos / lvl->cols;
        // Check if movement would cross floor boundaries
        if ((x == 0 && block_offset == -1) ||           // Don't move left off floor
            (x == lvl->cols-1 && block_offset == 1) ||       // Don't move right off floor
            (z == 0 && block_offset == -lvl->cols) ||        // Don't move forward off floor
            (z == lvl->rows-1 && block_offset == lvl->cols)) {    // Don't move back off floor
            continue;
        }

        // If we get here, movement is safe
        SET_BIT(outState->blocks, b_index + block_offset);
        affected_blocks[count] = b_index + block_offset;
        count++;
    }
    outState->block_grabbed = -1;
    int result =  handle_block_falling(outState, affected_blocks, blocks_to_move,count, lvl);
    free(blocks_to_move);
    return result;
}

int pull(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env, int block_offset){
    int pull_block = outState->robot_position + BFS_DIRECTION_VECTORS_X[outState->robot_orientation] + BFS_DIRECTION_VECTORS_Z[outState->robot_orientation]*lvl->cols;
    int block_in_front = TEST_BIT(outState->blocks, pull_block);
    int block_behind = TEST_BIT(outState->blocks, outState->robot_position + block_offset);
    int cell_below_next_position = outState->robot_position + block_offset - lvl->size;
    int backwards_walkable = bfs_is_valid_position(cell_below_next_position, lvl) && TEST_BIT(outState->blocks, cell_below_next_position);
    if (block_behind){
        return 0;
    }
    if (block_in_front){
        CLEAR_BIT(outState->blocks, pull_block);
        SET_BIT(outState->blocks, outState->robot_position);
        if (backwards_walkable){
            outState->block_grabbed = outState->robot_position;
            outState->robot_position = outState->robot_position + block_offset;
        }
        else {
            outState->robot_position = cell_below_next_position;
            outState->robot_state = 1;
            outState->block_grabbed = -1;
        }
    }
    int blocks_to_move[10];
    for(int i = 0; i<10; i++){
	    blocks_to_move[i] = -1;
    }
    blocks_to_move[0] = pull_block;
    int affected_blocks[1] = {-1};
    return handle_block_falling(outState, affected_blocks, blocks_to_move, 1, lvl);
}

int shimmy_normal(PuzzleState* outState, int action, const Level* lvl, int local_direction, int mode, CTowerClimb* env){
    int next_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
    int above_next_cell = next_cell + lvl->size;
    if (bfs_is_valid_position(above_next_cell, lvl) && !TEST_BIT(outState->blocks, above_next_cell)){
        outState->robot_position = next_cell;
        return 1;
    }
    return 0;
}

int wrap_around(PuzzleState* outState, int action, const Level* lvl, int mode, CTowerClimb* env){
    int action_idx = (action == LEFT) ? 0 : 1;
    int grid_pos = outState->robot_position % lvl->size;
    int x = grid_pos % lvl->cols;
    int z = grid_pos / lvl->cols;
    int current_floor = outState->robot_position / lvl->size;
    int new_x = x + wrap_x[outState->robot_orientation][action_idx];
    int new_z = z + wrap_z[outState->robot_orientation][action_idx];
    int new_pos = new_x + new_z*lvl->cols + current_floor*lvl->size;
    if (TEST_BIT(outState->blocks, new_pos + lvl->size)){
        return 0;
    }
    outState->robot_position = new_pos;
    outState->robot_orientation = wrap_orientation[outState->robot_orientation][action_idx];
    return 1;
}

int climb_from_hang(PuzzleState* outState, int action, const Level* lvl, int next_cell, int mode, CTowerClimb* env){
    int climb_index = next_cell + lvl->size;
    int direct_above_index = outState->robot_position + lvl->size;
    int can_climb = bfs_is_valid_position(climb_index, lvl) && bfs_is_valid_position(direct_above_index, lvl) 
    && !TEST_BIT(outState->blocks, climb_index) && !TEST_BIT(outState->blocks, direct_above_index);
    if (can_climb){
        outState->robot_position = climb_index;
        outState->robot_state = 0;
        return 1;
    }
    return 0;
}

int applyAction(PuzzleState* outState, int action,  Level* lvl, int mode, CTowerClimb* env) {
    // necessary variables
    int next_dx = BFS_DIRECTION_VECTORS_X[outState->robot_orientation];
    int next_dz = BFS_DIRECTION_VECTORS_Z[outState->robot_orientation];   
    int grid_pos = outState->robot_position % lvl->size;
    int x = grid_pos % lvl->cols;
    int z = grid_pos / lvl->cols;
    int next_cell = outState->robot_position + next_dx + next_dz*lvl->cols;
    int next_below_cell = next_cell - lvl->size;
    int walkable = bfs_is_valid_position(next_below_cell, lvl) && TEST_BIT(outState->blocks, next_below_cell);
    int block_in_front = bfs_is_valid_position(next_cell, lvl) && (TEST_BIT(outState->blocks, next_cell) || next_cell == lvl->goal_location);
    int movement_action = (action >= 0 && action < 4);
    int move_orient_check = (action == outState->robot_orientation);
    int standing_and_holding_nothing = outState->robot_state == 0 && outState->block_grabbed == -1;
    int hanging = outState->robot_state == 1;
    // Handle movement actions with common orientation check
    if (standing_and_holding_nothing && movement_action) {
        // If orientation doesn't match action, just rotate
        if (!move_orient_check) {
            outState->robot_orientation = action;
            return 1;
        }
        // Now handle the actual movement cases
        if (walkable && !block_in_front) {
            return move(outState, action, mode, env, lvl);
        }
        if (block_in_front) {
            return climb(outState, action, mode, env, lvl);
        }
        if (!block_in_front && !walkable) {
            return drop(outState, action, mode, env, lvl);
        }
    }

    if(hanging && movement_action){
        if (action == UP){
            return climb_from_hang(outState, action, lvl, next_cell, mode, env);
        }
        if (action == DOWN){
            return 0;
        }
        int local_direction = outState->robot_orientation;
        if (action == LEFT){
            local_direction = (outState->robot_orientation + 3) % 4;
        }
        if (action == RIGHT){
            local_direction = (outState->robot_orientation + 1) % 4;
        }
        int shimmy_cell = next_cell + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
        int shimmy_path_cell = outState->robot_position + BFS_DIRECTION_VECTORS_X[local_direction] + BFS_DIRECTION_VECTORS_Z[local_direction]*lvl->cols;
        
        int basic_shimmy = bfs_is_valid_position(shimmy_cell, lvl) && bfs_is_valid_position(shimmy_path_cell, lvl) && TEST_BIT(outState->blocks, shimmy_cell) && !TEST_BIT(outState->blocks, shimmy_path_cell);
        int rotation_shimmy = bfs_is_valid_position(shimmy_path_cell, lvl) && TEST_BIT(outState->blocks, shimmy_path_cell);
        int in_bounds = x + next_dx >= 0 && x + next_dx < lvl->cols && z + next_dz >= 0 && z + next_dz < lvl->rows;
        int wrap_shimmy = in_bounds && !TEST_BIT(outState->blocks, shimmy_path_cell) && !TEST_BIT(outState->blocks, shimmy_cell);
        
        if(basic_shimmy){
            return shimmy_normal(outState, action, lvl, local_direction, mode, env);
        }
        else if(rotation_shimmy){
            //rotate shimmy
            static const int LEFT_TURNS[] = {3, 0, 1, 2};   // RIGHT->UP, DOWN->RIGHT, LEFT->DOWN, UP->LEFT
            static const int RIGHT_TURNS[] = {1, 2, 3, 0};  // RIGHT->DOWN, DOWN->LEFT, LEFT->UP, UP->RIGHT

            outState->robot_orientation = (action == LEFT) ? 
                LEFT_TURNS[outState->robot_orientation] : 
                RIGHT_TURNS[outState->robot_orientation];
            outState->robot_state = 1;
            return 1;
        }
        else if(wrap_shimmy){
            return wrap_around(outState, action, lvl, mode, env);
        }
    }
    // drop from hang action 
    if (action == DROP && !standing_and_holding_nothing) {
        return drop_from_hang(outState, action, mode, lvl);
    }
    // grab action
    if (action == GRAB && standing_and_holding_nothing 
    && block_in_front){
        if (outState->block_grabbed == -1){
            outState->block_grabbed = next_cell;
            return 1;
        } 
    } 
    if (action == GRAB && outState->block_grabbed != -1){
        outState->block_grabbed = -1;
        return 1;
    }
    
    // push or pull block 
    if (movement_action && block_in_front && outState->block_grabbed != -1){
        int result = 0;
        int block_offset = BFS_DIRECTION_VECTORS_X[action] + BFS_DIRECTION_VECTORS_Z[action] * lvl->cols;
        if (outState->robot_orientation == action){
            result = push(outState, action, lvl, mode, env, block_offset);
        } else if(outState->robot_orientation == (action+2)%4){
            result = pull(outState, action, lvl, mode, env, block_offset);
        } else {
            outState->robot_orientation = action;
            outState->block_grabbed = -1;
            result = 1;
        }
        // block fell on top of player
        if (TEST_BIT(outState->blocks, outState->robot_position)){
            if (mode == PLG_MODE){
                return MOVE_ILLEGAL;
            }
            if (mode == RL_MODE){
                return MOVE_DEATH;
            }
        }
        if (mode == RL_MODE && result == 1){
            env->rewards[0] = env->reward_move_block;
            env->log.episode_return += env->reward_move_block;
        }
        return result;
    }
    return 0;   
}

int c_step(CTowerClimb* env) {
    env->log.episode_length += 1.0;
    env->rewards[0] = 0.0;
    if(env->log.episode_length > 60){
         env->rewards[0] = 0;
         env->log.perf = 0;
         add_log(env->log_buffer, &env->log);
         return 1;
    }
    // Create next state
    int move_result = applyAction(env->state, env->actions[0], env->level, RL_MODE, env);
    if (move_result == MOVE_ILLEGAL) {
        illegal_move(env);
        return 0;
    }
    if (move_result == MOVE_DEATH){
        death(env);
        return 1;
    }
    
    // Check for goal state
    if (isGoal(env->state, env->level)) {
        env->rewards[0] = 1.0;
        env->log.episode_return +=1.0;
        env->log.perf = 1.0;
        add_log(env->log_buffer, &env->log);
        return 1;
    }
    
    // Update observations
    compute_observations(env);
    return 0;
}

typedef struct BFSNode {
    PuzzleState state;
    int depth;      // how many moves from start
    int parent;     // index in BFS array of who generated me
    int action;     // which action led here (if you want to reconstruct the path)
} BFSNode;

static BFSNode* queueBuffer = NULL;
static int front = 0, back = 0;

// hash table for visited states
typedef struct VisitedNode {
    PuzzleState state;
    uint64_t hashVal;
    struct VisitedNode* next;
} VisitedNode;

static VisitedNode* visitedTable[TABLE_SIZE];
// Helper to incorporate a 32-bit integer into the hash one byte at a time.
static inline uint64_t fnv1a_hash_int(uint64_t h, int value) {
    // Break the int into 4 bytes (assuming 32-bit int).
    // This ensures consistent hashing regardless of CPU endianness.
    unsigned char bytes[4];
    bytes[0] = (unsigned char)((value >>  0) & 0xFF);
    bytes[1] = (unsigned char)((value >>  8) & 0xFF);
    bytes[2] = (unsigned char)((value >> 16) & 0xFF);
    bytes[3] = (unsigned char)((value >> 24) & 0xFF);
    for (int i = 0; i < 4; i++) {
        h ^= bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

uint64_t hashPuzzleState(const PuzzleState *s) {
    uint64_t h = FNV_OFFSET;
    // 1) Hash the 125-byte bitmask
    for (int i = 0; i < BLOCK_BYTES; i++) {
        h ^= s->blocks[i];
        h *= FNV_PRIME;
    }
    // 2) Hash the int fields (position, orientation, state, block_grabbed)
    h = fnv1a_hash_int(h, s->robot_position);
    h = fnv1a_hash_int(h, s->robot_orientation);
    h = fnv1a_hash_int(h, s->robot_state);
    h = fnv1a_hash_int(h, s->block_grabbed);
    return h;
}
// Compares two puzzle states fully
int equalPuzzleState(const PuzzleState* a, const PuzzleState* b) {
    // compare bitmask
    if (memcmp(a->blocks, b->blocks, BLOCK_BYTES) != 0) return 0;
    // compare other fields
    if (a->robot_position != b->robot_position) return 0;
    if (a->robot_orientation != b->robot_orientation) return 0;
    if (a->robot_state != b->robot_state) return 0;
    if (a->block_grabbed != b->block_grabbed) return 0;
    return 1;
}

void resetVisited(void) {
    memset(visitedTable, 0, sizeof(visitedTable));
}

// Helper function to find a node in the hash table
static VisitedNode* findNode(const PuzzleState* s, uint64_t hv, size_t idx) {
    VisitedNode* node = visitedTable[idx];
    while (node) {
        if (node->hashVal == hv && equalPuzzleState(&node->state, s)) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

int isVisited(const PuzzleState* s) {
    uint64_t hv = hashPuzzleState(s);
    size_t idx = (size_t)(hv % TABLE_SIZE);
    return findNode(s, hv, idx) != NULL;
}

void markVisited(const PuzzleState* s) {
    uint64_t hv = hashPuzzleState(s);
    size_t idx = (size_t)(hv % TABLE_SIZE);
    // Return if already present
    if (findNode(s, hv, idx)) {
        return;
    }
    // Insert new node
    VisitedNode* node = (VisitedNode*)malloc(sizeof(VisitedNode));
    node->state.blocks = (unsigned char*)malloc(BLOCK_BYTES * sizeof(unsigned char));
    // Copy the blocks data
    memcpy(node->state.blocks, s->blocks, BLOCK_BYTES * sizeof(unsigned char));
    // Copy other fields
    node->state.robot_position = s->robot_position;
    node->state.robot_orientation = s->robot_orientation;
    node->state.robot_state = s->robot_state;
    node->state.block_grabbed = s->block_grabbed;
    node->hashVal = hv;
    node->next = visitedTable[idx];
    visitedTable[idx] = node;
}

static PuzzleState copyPuzzleState(const PuzzleState* src) 
{
    PuzzleState dst = {
	    .blocks = NULL,
	    .robot_position = src->robot_position,
	    .robot_orientation = src->robot_orientation,
	    .robot_state = src->robot_state,
	    .block_grabbed = src->block_grabbed
    };
    if(src->blocks) {
	    dst.blocks = (unsigned char*)calloc(BLOCK_BYTES, sizeof(unsigned char));
	    if (dst.blocks) {
		    memcpy(dst.blocks, src->blocks, BLOCK_BYTES);
	    }
    }
    return dst;
}

// This function fills out up to MAX_NEIGHBORS BFSNodes in 'outNeighbors'
// from a given BFSNode 'current'. It returns how many neighbors it produced.
int getNeighbors(const BFSNode* current, BFSNode* outNeighbors,  Level* lvl) {
    int count = 0;
    // We'll read the current BFSNode's puzzle state
    const PuzzleState* curState = &current->state;
    // Try each action
    for (int i = 0; i < 6; i++) {
        int action = i;
        // 1) Make a copy of the current puzzle state
        PuzzleState newState = copyPuzzleState(curState); 
        // 2) Attempt to apply the action to newState
        int success = applyAction(&newState, action, lvl, PLG_MODE, NULL);
        if (!success) {
            // Move was invalid, skip
	    free(newState.blocks);
            continue;
        }
        // 3) If valid, build a BFSNode
        BFSNode neighbor;
        neighbor.state = newState;
        neighbor.depth = current->depth + 1;
        neighbor.parent = -1;   // BFS sets or overwrites this later
        neighbor.action = action; // record which action led here
        // 4) Add to 'outNeighbors' array
        outNeighbors[count++] = neighbor;
        // If you only allow up to 6 total, we can break if we reach that
        if (count >= MAX_NEIGHBORS) break;
    }
    return count; // how many valid neighbors we produced
}

void freeQueueBuffer(BFSNode* queueBuffer, int back){
    if (!queueBuffer) return;
    for (int i = 0; i < back; i++) {
        free(queueBuffer[i].state.blocks); 
    }
    free(queueBuffer);
}

// Example BFS
int bfs(PuzzleState* start, int maxDepth, Level* lvl, int min_moves) {
    // Clear or init your BFS queue
    queueBuffer = (BFSNode*)malloc(MAX_BFS_SIZE * sizeof(BFSNode));
    if (!queueBuffer) {
        printf("Failed to allocate memory for BFS queue\n");
        return 0;
    }
    front = 0;
    back = 0;
    // Enqueue start node
    BFSNode startNode;
    startNode.state = copyPuzzleState(start);  // copy puzzle state
    startNode.depth = 0;
    startNode.parent = -1;
    startNode.action = -1;
    queueBuffer[back++] = startNode;
    // BFS loop
    while (front < back) {
        if (back >= MAX_BFS_SIZE) {
            printf("BFS queue overflow! Increase MAX_BFS_SIZE or optimize search.\n");
            freeQueueBuffer(queueBuffer, back);
            queueBuffer = NULL;
            return 0;
        }
        BFSNode current = queueBuffer[front];
        int currentIndex = front;
        front++;
        // If current.state is the goal, reconstruct path
        if (isGoal(&current.state, lvl)) {
            if(current.depth < min_moves){
                freeQueueBuffer(queueBuffer, back);
                queueBuffer = NULL;
                return 0;
            }
            // Store nodes in order
            BFSNode* path = (BFSNode*)malloc((current.depth + 1) * sizeof(BFSNode));
            BFSNode node = current;
            int idx = current.depth;
            // Walk backwards to get path
            while (idx >= 0) {
                path[idx] = node;
                if (node.parent != -1) {
                    node = queueBuffer[node.parent];
                }
                idx--;
            }
            free(path);
            freeQueueBuffer(queueBuffer, back);
            queueBuffer = NULL;
            return 1;
        }
        if (current.depth >= maxDepth) continue;
        // generate neighbors
        BFSNode neighbors[MAX_NEIGHBORS];
        int nCount = getNeighbors(&current, neighbors, lvl);
        for (int i = 0; i < nCount; i++) {
            PuzzleState* nxt = &neighbors[i].state;
            if (!isVisited(nxt)) {
                markVisited(nxt);
                neighbors[i].depth = current.depth + 1;
                neighbors[i].parent = currentIndex;
                queueBuffer[back++] = neighbors[i];
            } else {
                free(nxt->blocks);
            }
        }        
    }
    freeQueueBuffer(queueBuffer, back);
    queueBuffer = NULL;
    // If we exit while, no solution found within maxDepth
    //printf("No solution within %d moves.\n", maxDepth);
    return 0;
}

void cleanupVisited(void) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        VisitedNode* current = visitedTable[i];
        while (current != NULL) {
            VisitedNode* next = current->next;
            free(current->state.blocks);
            free(current);
            current = next;
        }
        visitedTable[i] = NULL;
    }
}
int verify_level(Level* level, int max_moves, int min_moves){
    // converting level to puzzle state
    PuzzleState* state = calloc(1, sizeof(PuzzleState));
    init_puzzle_state(state);
    levelToPuzzleState(level, state);
    // reset visited hash table
    resetVisited();
    markVisited(state);
    // Run BFS
    int solvable = bfs(state, max_moves, level, min_moves);
    cleanupVisited();
    free_puzzle_state(state);
    return solvable;
}

void gen_level(Level* lvl, int goal_level) {
    // Initialize an illegal level in case we need to return early
    int legal_width_size = 8;
    int legal_depth_size = 8;
    int area = depth_max * col_max;
    int spawn_created = 0;
    int spawn_index = -1;
    int goal_created =0;
    int goal_index = -1;
    for(int y= 0; y < row_max; y++){
        for(int z = depth_max - 1; z >= 0; z--){
            for(int x = 0; x< col_max; x++){
                int block_index = x + col_max * z + area * y;
                int within_legal_bounds = x>=1 && x < legal_width_size && z >= 1 && z < legal_depth_size && y>=1 && y < goal_level;
                int allowed_block_placement = within_legal_bounds && (z <= (legal_depth_size - y));
                if (allowed_block_placement){
                    int chance = (rand() % 2 ==0) ? 1 : 0;
                    lvl->map[block_index] = chance;
                    // create spawn point above an existing block
                    if (spawn_created == 0 && y == 2 && lvl->map[block_index - area] == 1){
                        spawn_created = 1;
                        spawn_index = block_index;
                        lvl->map[spawn_index] = 0;
                    }
                }
                if (!goal_created && y == goal_level && 
                    (lvl->map[block_index + col_max - area] == 1 || 
                     lvl->map[block_index - 1 - area] == 1 || 
                     lvl->map[block_index + 1 - area] == 1)) {
                    // 33% chance to place goal here, unless we're at the last valid position
                    if (rand() % 3 == 0 || (x == col_max-1 && z == 0)) {
                        goal_created = 1;
                        goal_index = block_index;
                        lvl->map[goal_index] = 2;
                    }
                }
            }
        }
    }
    if (!spawn_created || spawn_index < 0) {
        return;
    }
    if (!goal_created || goal_index < 0) {
        return;
    }
    lvl->rows = row_max;
    lvl->cols = col_max;
    lvl->size = row_max * col_max;
    lvl->total_length = row_max * col_max * depth_max;
    lvl->goal_location = goal_index;
    lvl->spawn_location = spawn_index;
}

void init_random_level(CTowerClimb* env, int goal_level, int max_moves, int min_moves, int seed) {
	time_t t;
    srand((unsigned) time(&t) + seed); // Increment seed for each level
    reset_level(env->level);
    gen_level(env->level, goal_level);
    // guarantee a map is created
    while(env->level->spawn_location == 0 || env->level->goal_location == 999 || verify_level(env->level,max_moves, min_moves) == 0){
        reset_level(env->level);
        gen_level(env->level,goal_level);
    }
    levelToPuzzleState(env->level, env->state);
}

void cy_init_random_level(Level* level, int goal_level, int max_moves, int min_moves, int seed) {
    time_t t;
    srand((unsigned) time(&t) + seed); // Increment seed for each level
    gen_level(level, goal_level);
    // guarantee a map is created
    while(level->spawn_location == 0 || level->goal_location == 999 || verify_level(level,max_moves, min_moves) == 0){
        gen_level(level, goal_level);
    }
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_GREY = (Color){128, 128, 128, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef enum {
    ANIM_IDLE,
    ANIM_RUNNING,
    ANIM_CLIMBING,
    ANIM_HANGING,
    ANIM_START_GRABBING,
    ANIM_GRABBING,
    ANIM_SHIMMY_RIGHT,
    ANIM_SHIMMY_LEFT,
} AnimationState;

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Texture2D background;
    Camera3D camera;
    Model robot;
    Light lights[MAX_LIGHTS];
    Shader shader; 
    ModelAnimation* animations;
    int animFrameCounter;
    AnimationState animState;
    int previousRobotPosition;
    Vector3 visualPosition;
    Vector3 targetPosition;
    int isMoving;
    float moveProgress;
    Model cube;
    float scale;
    int enable_animations;
};

Client* make_client(CTowerClimb* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1600;
    client->height = 900;
    SetConfigFlags(FLAG_MSAA_4X_HINT);  // Enable MSAA
    InitWindow(client->width, client->height, "PufferLib Ray Tower Climb");
    SetTargetFPS(60);
    // camera
    client->camera = (Camera3D){ 0 };
    client->camera.position = (Vector3){ 0.0f, 25.0f, 20.0f };  // Move camera further back and higher up
    client->camera.target = (Vector3){ 2.0f, 4.0f, 2.0f };     // Keep looking at same target point
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    // load background
    client->background = LoadTexture("resources/tower_climb/space2.jpg");
    client->puffers = LoadTexture("resources/puffers_128.png");
    // load robot & cube models
    client->robot = LoadModel("resources/tower_climb/small_astro.glb");
    client->cube = LoadModel("resources/tower_climb/spacerock.glb");
    BoundingBox bounds = GetModelBoundingBox(client->cube);
    float cubeSize = bounds.max.x - bounds.min.x;
    float scale = 1.0f / cubeSize;
    client->scale = scale; 
    int animCount = 0;
    client->animations = LoadModelAnimations("resources/tower_climb/small_astro.glb", &animCount);
    printf("Loaded %d animations\n", animCount);
    client->animState = ANIM_IDLE;
    client->animFrameCounter = 0;
    UpdateModelAnimation(client->robot, client->animations[4], 0); 
    // Load and configure shader
    char vsPath[256];
    char fsPath[256];
    sprintf(vsPath, "resources/tower_climb/shaders/gls%i/lighting.vs", GLSL_VERSION);
    sprintf(fsPath, "resources/tower_climb/shaders/gls%i/lighting.fs", GLSL_VERSION);
    client->shader = LoadShader(vsPath, fsPath);
    // Get shader locations
    client->shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(client->shader, "viewPos");
    // Set up ambient light
    int ambientLoc = GetShaderLocation(client->shader, "ambient");
    float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f };
    SetShaderValue(client->shader, ambientLoc, ambient, SHADER_UNIFORM_VEC4);
    // apply lighting shader
    client->robot.materials[0].shader = client->shader;
    client->cube.materials[0].shader = client->shader;
    // Create lights with much brighter colors and closer positions
    client->lights[0] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 7.0f, 5.0f },  // Front
        Vector3Zero(), 
        WHITE,  // ~12% intensity
        client->shader);
    // Very dim fill lights
    client->lights[1] = CreateLight(LIGHT_POINT, 
        (Vector3){ 3.0f, 5.0f, 8.0f },  // Right side
        Vector3Zero(), 
        (Color){ 40, 40, 40, 255 },  // ~15% intensity
        client->shader);
    
    client->lights[2] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 7.0f, 5.0f },  // Front
        Vector3Zero(), 
        (Color){ 30, 30, 30, 255 },  // ~12% intensity
        client->shader);
    
    client->lights[3] = CreateLight(LIGHT_POINT, 
        (Vector3){ 10.0f, 6.0f, -5.0f },  // Left side
        Vector3Zero(), 
        (Color){ 20, 20, 20, 255 },  // ~8% intensity
        client->shader);

    // Make sure both models' materials use the lighting shader
    for (int i = 0; i < client->robot.materialCount; i++) {
        client->robot.materials[i].shader = client->shader;
    }
    for (int i = 0; i < client->cube.materialCount; i++) {
        client->cube.materials[i].shader = client->shader;
    }
    client->animState = ANIM_IDLE;
    client->previousRobotPosition = env->state->robot_position;
    // Initialize visual position to match starting robot position
    int floor = env->state->robot_position / env->level->size;
    int grid_pos = env->state->robot_position % env->level->size;
    int x = grid_pos % env->level->cols;
    int z = grid_pos / env->level->cols;
    client->visualPosition = (Vector3){ 
        x * 1.0f,
        floor * 1.0f,
        z * 1.0f
    };
    client->targetPosition = client->visualPosition;  // Initialize target to match
    return client;
}

void orient_hang_offset(Client* client, CTowerClimb* env, int reverse){
    client->visualPosition.y -= 0.2f * reverse;
    if (env->state->robot_orientation == 0) { // Facing +x
        client->visualPosition.x += 0.4f * reverse;
    } else if (env->state->robot_orientation == 1) { // Facing +z
        client->visualPosition.z += 0.4f * reverse;
    } else if (env->state->robot_orientation == 2) { // Facing -x
        client->visualPosition.x -= 0.4f * reverse;
    } else if (env->state->robot_orientation == 3) { // Facing -z
        client->visualPosition.z -= 0.4f * reverse;
    }
}

// Animation configuration
typedef struct {
    int animationIndex;
    int frameRate;
    int maxFrames;
    int startFrame;           // Added startFrame configuration
    AnimationState nextState;
} AnimConfig;

static const AnimConfig ANIM_CONFIGS[] = {
    [ANIM_IDLE] = {4, 1, -1, 0, ANIM_IDLE},            // Loops from start
    [ANIM_CLIMBING] = {1, 6, -1, 0, ANIM_IDLE},        // Start from beginning
    [ANIM_HANGING] = {2, 0, 1, 0, ANIM_HANGING},       // Static frame
    [ANIM_START_GRABBING] = {3, 6, -2, 0, ANIM_GRABBING}, // Normal grab start
    [ANIM_GRABBING] = {3, 4, -2, -2, ANIM_GRABBING},   // Start at second-to-last frame
    [ANIM_RUNNING] = {5, 4, -1, 0, ANIM_IDLE},         // Start from beginning
    [ANIM_SHIMMY_RIGHT] = {7, 2, 87, 0, ANIM_HANGING}, // Start from beginning
    [ANIM_SHIMMY_LEFT] = {6, 2, 87, 0, ANIM_HANGING}   // Start from beginning
};

static void update_animation(Client* client, AnimationState newState) {
    if (!client->enable_animations) return;
    const AnimConfig* config = &ANIM_CONFIGS[newState];
    client->animState = newState;
    // Handle negative startFrame (counting from end)
    int startFrame = config->startFrame;
    if (startFrame < 0) {
        startFrame = client->animations[config->animationIndex].frameCount + startFrame;
    }
    client->animFrameCounter = startFrame;
    UpdateModelAnimation(client->robot, client->animations[config->animationIndex], startFrame);
    if (newState == ANIM_IDLE || newState == ANIM_GRABBING || newState == ANIM_HANGING || newState == ANIM_START_GRABBING) {
        client->visualPosition = client->targetPosition;
    }
}

static void update_position(Client* client, CTowerClimb* env) {
    int floor = env->state->robot_position / env->level->size;
    int grid_pos = env->state->robot_position % env->level->size;
    int x = grid_pos % env->level->cols;
    int z = grid_pos / env->level->cols;
    client->targetPosition = (Vector3){x * 1.0f, floor * 1.0f, z * 1.0f};
}

static void process_animation_frame(Client* client, CTowerClimb* env) {
    if (!client->enable_animations) return;
    const AnimConfig* config = &ANIM_CONFIGS[client->animState];
    if (!client->isMoving && client->animState != ANIM_IDLE) return;
    
    client->animFrameCounter += config->frameRate;
    UpdateModelAnimation(client->robot, client->animations[config->animationIndex], 
                        client->animFrameCounter);
    // Handle shimmy movement lerping
    if (client->isMoving && (client->animState == ANIM_SHIMMY_LEFT || 
                            client->animState == ANIM_SHIMMY_RIGHT)) {
        float progress = 0.1f;
        // Horizontal movement for UP/DOWN, vertical movement for LEFT/RIGHT
        bool facingNS = env->state->robot_orientation == UP || env->state->robot_orientation == DOWN;
        if (facingNS) {
            client->visualPosition.x = Lerp(client->visualPosition.x, client->targetPosition.x, progress);
        } else {
            client->visualPosition.z = Lerp(client->visualPosition.z, client->targetPosition.z, progress);
        }
    }
    // Check for animation completion
    int maxFrames = config->maxFrames;
    if (maxFrames < 0) {
        maxFrames = client->animations[config->animationIndex].frameCount + maxFrames;
    }
    // If we've reached the end of the animation, update the animation state
    if (maxFrames > 0 && client->animFrameCounter >= maxFrames) {
        client->isMoving = false;
        update_animation(client, config->nextState);
        client->visualPosition = client->targetPosition;
        if (config->nextState == ANIM_HANGING) {
            orient_hang_offset(client, env, 1);
        }
    }
}

static void handle_hanging_movement(Client* client, CTowerClimb* env) {
    bool is_wrap_shimmy = fabs(client->targetPosition.x - client->visualPosition.x) > 0.5f && 
                         fabs(client->targetPosition.z - client->visualPosition.z) > 0.5f;
    // First ensure we have the correct hanging offset if we just transitioned to hanging
    if ((int)client->visualPosition.x == client->visualPosition.x && (int)client->visualPosition.z == client->visualPosition.z) {
        orient_hang_offset(client, env, 1);
    }
    if (is_wrap_shimmy) {
        client->isMoving = false;
        update_animation(client, ANIM_HANGING);
        client->visualPosition = client->targetPosition;
        orient_hang_offset(client, env, 1);
        return;
    }
    // Determine movement direction based on orientation
    bool moving_right = false;
    switch (env->state->robot_orientation) {
        case UP:    moving_right = client->targetPosition.x > client->visualPosition.x; break;
        case DOWN:  moving_right = client->targetPosition.x < client->visualPosition.x; break;
        case RIGHT: moving_right = client->targetPosition.z < client->visualPosition.z; break;
        case LEFT:  moving_right = client->targetPosition.z > client->visualPosition.z; break;
    }
    if (client->targetPosition.y < client->visualPosition.y) {
        update_animation(client, ANIM_HANGING);
        orient_hang_offset(client, env, 1);
	client->isMoving = false;
    } else {
        update_animation(client, moving_right ? ANIM_SHIMMY_RIGHT : ANIM_SHIMMY_LEFT);
    }
}

static void update_camera(Client* client, CTowerClimb* env) {
    int floor = env->state->robot_position / env->level->size;
    int cameraFloor = (floor - 1) * 0.5;
    float targetCameraY = cameraFloor * 1.0f + 7.0f;
    float targetLookY = cameraFloor * 1.0f;
    float smoothSpeed = 0.025f;
    // Update camera position
    client->camera.position.y = Lerp(client->camera.position.y, targetCameraY, smoothSpeed);
    client->camera.target.y = Lerp(client->camera.target.y, targetLookY, smoothSpeed);
    client->camera.position = (Vector3){
        env->level->cols * 0.5f,
        client->camera.position.y,
        25.0f
    };
    client->camera.target = (Vector3){4.0f, client->camera.target.y, 1.0f};
}

static void draw_background(Client* client) {
    float scaleWidth = (float)client->width / client->background.width;
    float scaleHeight = (float)client->height / client->background.height;
    float scale = fmax(scaleWidth, scaleHeight);
    
    Rectangle dest = {
        .x = (client->width - client->background.width * scale) * 0.5f,
        .y = (client->height - client->background.height * scale) * 0.5f,
        .width = client->background.width * scale,
        .height = client->background.height * scale
    };
    Rectangle source = {0, 0, client->background.width, client->background.height};
    DrawTexturePro(client->background, source, dest, (Vector2){0, 0}, 0.0f, WHITE);
}

static void draw_level(Client* client, CTowerClimb* env) {
    int cols = env->level->cols;
    int sz = env->level->size;
    for(int i = 0; i < env->level->total_length; i++) {
        int floor = i / sz;
        int grid_pos = i % sz;
        int x = grid_pos % cols;
        int z = grid_pos / cols;
        Vector3 pos = {x * 1.0f, floor * 1.0f, z * 1.0f};
        if(TEST_BIT(env->state->blocks, i)) {
            DrawModel(client->cube, pos, client->scale, WHITE);
            Color wireColor = (i == env->state->block_grabbed) ? RED : BLACK;
            DrawCubeWires(pos, 1.0f, 1.0f, 1.0f, wireColor);
        }
        if (i == env->level->goal_location) {
            EndShaderMode();
            DrawCube(pos, 1.0f, 1.0f, 1.0f, PUFF_CYAN);
            BeginShaderMode(client->shader);
        }
    }
}

static void draw_robot(Client* client, CTowerClimb* env) {
    Vector3 pos = client->visualPosition;
    pos.y -= 0.5f;
    rlPushMatrix();
    rlTranslatef(pos.x, pos.y, pos.z);
    rlRotatef(90.0f, 1, 0, 0);
    rlRotatef(-90.0f + env->state->robot_orientation * 90.0f, 0, 0, 1);
    DrawModel(client->robot, (Vector3){0, 0, 0}, 0.5f, WHITE);
    rlPopMatrix();
}

static void render_scene(Client* client, CTowerClimb* env) {
    BeginDrawing();
    ClearBackground(BLACK);
    EndShaderMode();
    draw_background(client);
    BeginShaderMode(client->shader);
    BeginMode3D(client->camera);
    // Update shader camera position
    float cameraPos[3] = {
        client->camera.position.x,
        client->camera.position.y,
        client->camera.position.z
    };
    SetShaderValue(client->shader, client->shader.locs[SHADER_LOC_VECTOR_VIEW], 
                  cameraPos, SHADER_UNIFORM_VEC3);
    BeginBlendMode(BLEND_ALPHA);
    draw_level(client, env);
    EndBlendMode();
    draw_robot(client, env);
    EndMode3D();
    EndDrawing();
}

void c_render(Client* client, CTowerClimb* env) {
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    // Handle state transitions - drop animation
    if (env->state->robot_state == DEFAULT && client->animState == ANIM_HANGING && client->enable_animations) {
        update_animation(client, ANIM_IDLE);
        client->isMoving = false;
        client->visualPosition = client->targetPosition;
    }
    // grab animation
    if (env->state->block_grabbed != -1 && 
        client->animState != ANIM_GRABBING && 
        client->animState != ANIM_START_GRABBING && client->enable_animations) {
        update_animation(client, ANIM_START_GRABBING);
        client->isMoving = true;
    } else if (env->state->block_grabbed == -1 && client->animState == ANIM_GRABBING && client->enable_animations) {
        update_animation(client, ANIM_IDLE);
    }
    // Handle position changes
    if (env->state->robot_position != client->previousRobotPosition && client->enable_animations) {
        if (client->isMoving) client->visualPosition = client->targetPosition;
        client->isMoving = true;
        update_position(client, env);
        float verticalDiff = client->targetPosition.y - client->visualPosition.y;
        if (verticalDiff > 0.5) {
            orient_hang_offset(client, env, client->animState == ANIM_HANGING ? 0 : 1);
            update_animation(client, ANIM_CLIMBING);
        } else if (env->state->robot_state == HANGING) {
            handle_hanging_movement(client, env);
        } else {
            update_animation(client, verticalDiff < 0 ? ANIM_IDLE : ANIM_RUNNING);
            if (verticalDiff < 0) {
                client->isMoving = false;
                client->visualPosition = client->targetPosition;
            }
        }
        client->previousRobotPosition = env->state->robot_position;
    }
    if(!client->enable_animations) {
        update_position(client, env);
        client->visualPosition = client->targetPosition;
    }
    process_animation_frame(client, env);
    update_camera(client, env);
    render_scene(client, env);
}

void close_client(Client* client) {
    UnloadShader(client->shader);
    CloseWindow();
    UnloadModel(client->robot);
    free(client);
}
