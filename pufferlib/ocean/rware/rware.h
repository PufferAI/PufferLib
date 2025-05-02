#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"

#define NOOP 0
#define FORWARD 1
#define LEFT 2
#define RIGHT 3
#define TOGGLE_LOAD 4
#define TICK_RATE 1.0f/60.0f
#define NUM_DIRECTIONS 4

// warehouse states
#define EMPTY 0
#define SHELF 1
#define REQUESTED_SHELF 2
#define GOAL 3

// agent states
#define UNLOADED 0
#define HOLDING_REQUESTED_SHELF 1
#define HOLDING_EMPTY_SHELF 2

// observation types
#define SELF_OBS 3
#define VISION_OBS 24

// Facing directions
#define FACING_RIGHT 0
#define FACING_DOWN 1
#define FACING_LEFT 2
#define FACING_UP 3

// Reward Type
#define INDIVIDUAL 1
#define GLOBAL 2
static const int DIRECTIONS[NUM_DIRECTIONS] = {0, 1, 2, 3};
static const int DIRECTION_VECTORS[NUM_DIRECTIONS][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
static const int SURROUNDING_VECTORS[8][2] = {{0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}};
static const int tiny_map[110] = {
    0,0,0,0,0,0,0,0,0,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0, 
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0, 
    0,1,1,0,0,0,0,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,3,3,0,0,0,0   
};
static const int tiny_shelf_locations[32] = {
    11, 12, 17, 18,  // Row 1
    21, 22, 27, 28,  // Row 2
    31, 32, 37, 38,  // Row 3
    41, 42, 47, 48,  // Row 4
    51, 52, 57, 58,  // Row 5
    61, 62, 67, 68,  // Row 6
    71, 72, 77, 78,  // Row 7
    81, 82, 87, 88   // Row 8
};

static const int small_map[200] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

static const int small_shelf_locations[80] = {
    22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,  // Row 1
    42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,  // Row 2
    91,92,93,94,95,96,97,98,  // Row 4 (right side only)
    111,112,113,114,115,116,117,118,  // Row 5 (right side only)
    142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,  // Row 7 (both sides)
    162,163,164,165,166,167,168,169,171,172,173,174,175,176,177,178   // Row 8 (both sides)
};

static const int medium_map[320] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

static const int medium_shelf_locations[144]={
    22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,  // Row 1
    42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,  // Row 2
    82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,98,  // Row 4
    102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,  // Row 5
    151,152,153,154,155,156,157,158,  // Row 7 (right side only)
    171,172,173,174,175,176,177,178,  // Row 8 (right side only)
    202,203,204,205,206,207,208,209,211,212,213,214,215,216,217,218,  // Row 10
    222,223,224,225,226,227,228,229,231,232,233,234,235,236,237,238,  // Row 11
    262,263,264,265,266,267,268,269,271,272,273,274,275,276,277,278,  // Row 13
    282,283,284,285,286,287,288,289,291,292,293,294,295,296,297,298   // Row 14
};

static const int map_sizes[3] = {110, 200, 320};
static const int map_rows[3] = {11, 10, 16};
static const int map_cols[3] = {10, 20, 20};
static const int* maps[3] = {tiny_map, small_map, medium_map};
#define LOG_BUFFER_SIZE 1024

static inline int max(int a, int b) {
    return (a > b) ? a : b;
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
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;   
        log.perf += fmaxf(0.0, logs->logs[i].score - 0.01*logs->logs[i].episode_length);
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.perf /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct MovementGraph MovementGraph;
struct MovementGraph {
    int* target_positions;
    int* cycle_ids;
    int* weights;
    int num_cycles;
};


typedef struct CRware CRware;
struct CRware {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log* logs;
    float* scores;
    int reward_type;
    int width;
    int height;
    int map_choice;
    int* warehouse_states;
    int num_agents;
    int num_requested_shelves;
    int* agent_locations;
    int* agent_directions;
    int* agent_states;
    int human_agent_idx;
    int grid_square_size;
    int* original_shelve_locations;
    MovementGraph* movement_graph;
};

int find_agent_at_position(CRware* env, int position) {
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agent_locations[i] == position) {
            return i;
        }
    }
    return -1;
}

void place_agent(CRware* env, int agent_idx) {
    // map size fixed at top
    int map_size = map_sizes[env->map_choice - 1];
    
    int found_valid_position = 0;
    while (!found_valid_position) {
        int random_pos = rand() % map_size;
        
        // Skip if position is not empty
        if (env->warehouse_states[random_pos] != EMPTY) {
            continue;
        }

        // Skip if another agent is already here
        int agent_at_position = find_agent_at_position(env, random_pos);
        if (agent_at_position != -1) {
            continue;
        }

        // Position is valid, place the agent
        env->agent_locations[agent_idx] = random_pos;
        env->agent_directions[agent_idx] = rand() % 4;
        env->agent_states[agent_idx] = 0;
        found_valid_position = 1;
    }
}

int request_new_shelf(CRware* env) {
    int total_shelves;
    const int* shelf_locations;
    if (env->map_choice == 1) {
        total_shelves = 32;
        shelf_locations = tiny_shelf_locations;
    } else if (env->map_choice == 2) {
        total_shelves = 80;
        shelf_locations = small_shelf_locations;
    } else {
        total_shelves = 144;
        shelf_locations = medium_shelf_locations;
    }
    int random_index = rand() % total_shelves;
    int shelf_location = shelf_locations[random_index];
    if (env->warehouse_states[shelf_location] == SHELF ) {
        env->warehouse_states[shelf_location] = REQUESTED_SHELF;
        return 1;
    }
    return 0;
}

void generate_map(CRware* env,const int* map) {
    // seed new random
    srand(time(NULL));
    int map_size = map_sizes[env->map_choice - 1];
    memcpy(env->warehouse_states, map, map_size * sizeof(int));

    int requested_shelves_count = 0;
    while (requested_shelves_count < env->num_requested_shelves) {
        requested_shelves_count += request_new_shelf(env);
    }
    // set agents random locations
    for (int i = 0; i < env->num_agents; i++) {
        place_agent(env, i);
    }
}

MovementGraph* init_movement_graph(CRware* env) {
    MovementGraph* graph = (MovementGraph*)calloc(1, sizeof(MovementGraph));
    graph->target_positions = (int*)calloc(env->num_agents, sizeof(int));
    graph->cycle_ids = (int*)calloc(env->num_agents, sizeof(int));
    graph->weights = (int*)calloc(env->num_agents, sizeof(int));
    graph->num_cycles = 0;

    // Initialize arrays
    for (int i = 0; i < env->num_agents; i++) {
        graph->target_positions[i] = -1;   // No target
        graph->cycle_ids[i] = -1;          // Not in cycle
    }
    return graph;
}

void init(CRware* env) {
    int map_size = map_sizes[env->map_choice - 1];
    env->warehouse_states = (int*)calloc(map_size, sizeof(int));
    env->agent_locations = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_directions = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_states = (int*)calloc(env->num_agents, sizeof(int));
    env->scores = (float*)calloc(env->num_agents,sizeof(float));
    env->movement_graph = init_movement_graph(env);
    env->logs = (Log*)calloc(env->num_agents, sizeof(Log));
    if (env->map_choice == 1) {
        generate_map(env, tiny_map);
    }
    else if (env->map_choice == 2) {
        generate_map(env, small_map);
    }
    else {
        generate_map(env, medium_map);
    }
}

void allocate(CRware* env) {
    init(env);
    env->observations = (float*)calloc(env->num_agents*(SELF_OBS+VISION_OBS), sizeof(float));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(CRware* env) {
    free(env->warehouse_states);
    free(env->agent_locations);
    free(env->agent_directions);
    free(env->agent_states);
    free(env->movement_graph->target_positions);
    free(env->movement_graph->cycle_ids);
    free(env->movement_graph->weights);
    free(env->movement_graph);
    free(env->logs);
    free(env->scores);
}

void free_allocated(CRware* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

void compute_observations(CRware* env) {
    int surround_indices[8];
    int cols = map_cols[env->map_choice - 1];
    int rows = map_rows[env->map_choice - 1];
    float (*observations)[SELF_OBS+VISION_OBS] = (float(*)[SELF_OBS+VISION_OBS])env->observations;
    for (int i = 0; i < env->num_agents; i++) {
        // Agent location, direction, state
        float* obs = &observations[i][0];
        int agent_location = env->agent_locations[i];
        int current_x = agent_location % cols;
        int current_y = agent_location / cols;
        // Self observations
        obs[0] = env->agent_locations[i] / (float)(rows*cols);
        obs[1] = (env->agent_directions[i] + 1) / 4.0;
        obs[2] = env->agent_states[i];
	// Vision observations
        for (int j = 0; j < 8; j++) {
            int new_x = current_x + SURROUNDING_VECTORS[j][0];
            int new_y = current_y + SURROUNDING_VECTORS[j][1];
            surround_indices[j] = new_x + new_y * cols;
            // other robots location and rotation if on that spot
            for (int k = 0; k < env->num_agents; k++) {
                if(i==k){
                    continue;
                }
                if(env->agent_locations[k] == surround_indices[j]){
                    obs[3 + j*3] = 1;
                    obs[4 + j*3] = (env->agent_directions[k] + 1) / 4.0;
                    break;
                } else {
                    obs[3 + j*3] = 0;
                    obs[4 + j*3] = 0;
                    break;
                }
            }
            // boundary check
            if (new_x < 0 || new_x >= cols || new_y < 0 || new_y >= rows) {
                obs[5 + j*3] = 0;
            } else {
                obs[5 + j*3] = (env->warehouse_states[surround_indices[j]] + 1) / 4.0;
            }
        }
    }
}

void c_reset(CRware* env) {
     
	env->dones[0] = 0;
    // set agents in center
    env->human_agent_idx = 0;
    if (env->map_choice == 1) {
        generate_map(env, tiny_map);
    } else if (env->map_choice == 2) {
        generate_map(env, small_map);
    } else {
        generate_map(env, medium_map);
    }
    for(int x = 0;x<env->num_agents; x++){
	    env->scores[x] = 0.0;
        env->logs[x] = (Log){0};
    }
    compute_observations(env);
    
}

int get_direction(CRware* env, int action, int agent_idx) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    int current_direction = env->agent_directions[agent_idx];
    if (action == FORWARD) {
        return current_direction;
    }
    else if (action == LEFT) {
        // Rotate counter-clockwise
        return (current_direction + 3) % NUM_DIRECTIONS;
    }
    else if (action == RIGHT) {
        // Rotate clockwise
        return (current_direction + 1) % NUM_DIRECTIONS;
    }
    return current_direction;
}

int get_new_position(CRware* env, int agent_idx) {
    int cols = map_cols[env->map_choice - 1];
    int rows = map_rows[env->map_choice - 1];
    int current_position_x = env->agent_locations[agent_idx] % cols;
    int current_position_y = env->agent_locations[agent_idx] / cols;
    int current_direction = env->agent_directions[agent_idx];
    int new_position_x = current_position_x + DIRECTION_VECTORS[current_direction][0];
    int new_position_y = current_position_y + DIRECTION_VECTORS[current_direction][1];
    int new_position = new_position_x + new_position_y * cols;
    // check boundary
    if (new_position_x < 0 || new_position_x >= cols || new_position_y < 0 || new_position_y >= rows) {
        return -1;
    }
    // check if holding shelf and next position is a shelf
    int agent_state = env->agent_states[agent_idx];
    int warehouse_state = env->warehouse_states[new_position];
    if ((agent_state == HOLDING_EMPTY_SHELF || agent_state == HOLDING_REQUESTED_SHELF) && 
    (warehouse_state == SHELF || warehouse_state == REQUESTED_SHELF)) {
        return -1;
    }
    // check if agent is trying to move into a goal with empty shelf
    if (agent_state == HOLDING_EMPTY_SHELF && warehouse_state == GOAL) {
        return -1;
    }
    return new_position;
}

int detect_cycle_for_agent(CRware* env, int start_agent) {
    MovementGraph* graph = env->movement_graph;
    
    // If already processed or no target, skip
    if (graph->cycle_ids[start_agent] != -1 || 
        graph->target_positions[start_agent] == -1) {
        return -1;
    }

    // Initialize tortoise and hare
    int tortoise = find_agent_at_position(env, graph->target_positions[start_agent]);
    if (tortoise == -1) return -1;
    int hare = tortoise;

    // Move hare ahead by one step
    hare = find_agent_at_position(env, graph->target_positions[hare]);
    if (hare == -1) return -1;

    // Loop to detect cycle
    while (tortoise != hare) {
        tortoise = find_agent_at_position(env, graph->target_positions[tortoise]);
        if (tortoise == -1) return -1;

        hare = find_agent_at_position(env, graph->target_positions[hare]);
        if (hare == -1) return -1;
        hare = find_agent_at_position(env, graph->target_positions[hare]);
        if (hare == -1) return -1;
    }

    // Find the start of the cycle
    tortoise = start_agent;
    while (tortoise != hare) {
        tortoise = find_agent_at_position(env, graph->target_positions[tortoise]);
        hare = find_agent_at_position(env, graph->target_positions[hare]);
    }

    int cycle_start = tortoise;

    // Mark all agents in the cycle
    int cycle_id = graph->num_cycles++;
    int current = cycle_start;
    do {
        graph->cycle_ids[current] = cycle_id;
        current = find_agent_at_position(env, graph->target_positions[current]);
    } while (current != cycle_start);

    return cycle_id;
}

void detect_cycles(CRware* env) {
    for (int i = 0; i < env->num_agents; i++) {
        if(env->movement_graph->cycle_ids[i] == -1) {
            detect_cycle_for_agent(env, i);
        }
    }
}

void calculate_weights(CRware* env) {
    MovementGraph* graph = env->movement_graph;
    
    // First pass: identify leaf nodes (agents not targeted by others)
    for (int i = 0; i < env->num_agents; i++) {
        if (graph->cycle_ids[i] != -1) continue;  // Skip agents in cycles
        
        bool is_leaf = true;
        for (int j = 0; j < env->num_agents; j++) {
            if (graph->target_positions[j] != env->agent_locations[i]) continue;
            is_leaf = false;
            break;
        }
        
        if (is_leaf) {
            graph->weights[i] = 1;  // Leaf node
        }
    }

    bool changed = true;
    while(changed) {
        changed = false;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != -1) continue;
            
            // Find agents targeting this agent's position
            int max_child_weight = 0;
            for (int j = 0; j < env->num_agents; j++) {
                if (graph->target_positions[j] != env->agent_locations[i]) continue;
                max_child_weight = max(max_child_weight, graph->weights[j]);
            }
            
            if (max_child_weight == 0 || graph->weights[i] == max_child_weight + 1) {
                continue;
            }
            graph->weights[i] = max_child_weight + 1;
            changed = true;
        }
    }
}

void update_movement_graph(CRware* env, int agent_idx) {
    MovementGraph* graph = env->movement_graph;
    int new_position = get_new_position(env, agent_idx);
    if (new_position == -1) {
        return;
    }
    graph->target_positions[agent_idx] = new_position;

    // reset cycle and weights
    for (int i = 0; i < env->num_agents; i++) {
        graph->cycle_ids[i] = -1;
        graph->weights[i] = 0;
    }
    graph->num_cycles = 0;

    // detect cycles with Floyd algorithm
    detect_cycles(env);

    // calculate weights for tree
    calculate_weights(env);
}

void move_agent(CRware* env, int agent_idx) {
    
    int new_position = get_new_position(env, agent_idx);
    // check boundary
    if (new_position == -1) {
        return;
    }
    // if reach goal
    int agent_state = env->agent_states[agent_idx];
    int agent_location = env->agent_locations[agent_idx];
    int new_position_state = env->warehouse_states[new_position];
    int current_position_state = env->warehouse_states[agent_location];
    if (new_position_state == GOAL && agent_state== HOLDING_REQUESTED_SHELF) {
        if (current_position_state != GOAL) {
            env->warehouse_states[agent_location] = 0;
        }
        env->agent_locations[agent_idx] = new_position;
        return;
    }
    // if agent is holding requested shelf
    if (agent_state == HOLDING_REQUESTED_SHELF) {
        if (current_position_state != GOAL) {
            env->warehouse_states[agent_location] = 0;
        }
        env->warehouse_states[new_position] = REQUESTED_SHELF;
    }
    // if agent is holding empty shelf
    if (agent_state == HOLDING_EMPTY_SHELF) {
        if (current_position_state != GOAL){
            env->warehouse_states[agent_location] = 0;
        }
        env->warehouse_states[new_position] = SHELF;
    }
    env->agent_locations[agent_idx] = new_position;
    env->movement_graph->target_positions[agent_idx] = -1;
}

void pickup_shelf(CRware* env, int agent_idx) {
    // pickup shelf
    const int* map = maps[env->map_choice - 1];
    int agent_location = env->agent_locations[agent_idx];
    int agent_state = env->agent_states[agent_idx];
    int current_position_state = env->warehouse_states[agent_location];
    int original_map_state = map[agent_location];
    if ((current_position_state == REQUESTED_SHELF) && (agent_state==UNLOADED)) {
        env->rewards[agent_idx] = 0.5;
	env->logs[agent_idx].episode_return += 0.5;
	env->agent_states[agent_idx]=HOLDING_REQUESTED_SHELF;
    }
    // return empty shelf
    else if (agent_state == HOLDING_EMPTY_SHELF && current_position_state == original_map_state 
    && original_map_state != GOAL) {
        env->agent_states[agent_idx]=UNLOADED;
        env->warehouse_states[agent_location] = original_map_state;
        env->rewards[agent_idx] = 1.0;

        env->logs[agent_idx].score = 1.0;
        env->logs[agent_idx].episode_return += 1.0;

	    env->scores[agent_idx] = 0;
        add_log(env->log_buffer, &env->logs[agent_idx]);
        env->logs[agent_idx] = (Log){0};
    }
    // drop shelf at goal
    else if (agent_state == HOLDING_REQUESTED_SHELF && current_position_state == GOAL) {
        env->agent_states[agent_idx]=HOLDING_EMPTY_SHELF;
        env->rewards[agent_idx] = 0.5;
        env->logs[agent_idx].episode_return += 0.5;
        env->logs[agent_idx].score = 1.0;
        int shelf_count = 0;
        while (shelf_count < 1) {
            shelf_count += request_new_shelf(env);
        }
    }
}

void process_cycle_movements(CRware* env, MovementGraph* graph) {
    for (int cycle = 0; cycle < graph->num_cycles; cycle++) {
        int cycle_size = 0;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] == cycle) {
                cycle_size++;
            }
        }
        if (cycle_size == 2) continue;

        bool can_move_cycle = true;
        // Verify all agents in cycle can move
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != cycle) continue;
            int new_pos = get_new_position(env, i);
            if (new_pos == -1) {
                can_move_cycle = false;
                break;
            }
        }
        
        // Move all agents in cycle if possible
        if (!can_move_cycle) continue;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != cycle) continue;
            if (env->actions[i] != FORWARD) continue;            
            move_agent(env, i);
        }
    }
}

void process_tree_movements(CRware* env, MovementGraph* graph) {
    int max_weight = 0;
    for (int i = 0; i < env->num_agents; i++) {
        if (graph->cycle_ids[i] == -1 && graph->weights[i] > max_weight) {
            max_weight = graph->weights[i];
        }
    }
    // Process from highest weight to lowest
    for (int weight = max_weight; weight > 0; weight--) {
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != -1 || graph->weights[i] != weight) continue;
            if (env->actions[i] != FORWARD) continue;

            int new_pos = get_new_position(env, i);
            if (new_pos == -1) continue;

            int target_agent = find_agent_at_position(env, new_pos);
            if (target_agent != -1) continue;
            move_agent(env, i);
        }
    }
}

void c_step(CRware* env) {
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    MovementGraph* graph = env->movement_graph;
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i].episode_length += 1;
        int action = env->actions[i];
        
	// Handle direction changes and non-movement actions
        if (action != NOOP && action != TOGGLE_LOAD) {
            env->agent_directions[i] = get_direction(env, action, i);
        }
        if (action == TOGGLE_LOAD) {
            pickup_shelf(env, i);
        }
        if (action == FORWARD) {
            update_movement_graph(env, i);
        }
    }
    int is_movement=0;
    for(int i=0; i<env->num_agents; i++) {
        if (env->actions[i] == FORWARD) is_movement++;
    }
    if (is_movement>=1) {
        // Process movements in cycles first
        process_cycle_movements(env, graph);
        // process tree movements
        process_tree_movements(env, graph);
    }

    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_GREY = (Color){128, 128, 128, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
};

Client* make_client(CRware* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "PufferLib Ray RWare");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/puffers_128.png");
    return client;
}

void c_render(Client* client, CRware* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    
    int map_size = map_sizes[env->map_choice - 1];
    int cols = map_cols[env->map_choice - 1];
    for (int i = 0; i < map_size; i++) {
        int state = env->warehouse_states[i];
        Color color;
        if (state == SHELF) {
            color = PUFF_CYAN;
        } else if (state == REQUESTED_SHELF) {
            color = RED;
        } else if (state == GOAL) {
            color = STONE_GRAY;
        } else {
            color = PUFF_BACKGROUND;
        }

        int x_pos = i % cols * env->grid_square_size;
        int y_pos = i / cols * env->grid_square_size;
        DrawRectangle(
            x_pos,  // x position
            y_pos,  // y position
            env->grid_square_size, 
            env->grid_square_size, 
            color
        );
        
        DrawRectangleLines(
            x_pos,
            y_pos,
            env->grid_square_size,
            env->grid_square_size,
            PUFF_GREY
        );

        DrawRectangleLines(
            x_pos + 1,
            y_pos + 1,
            env->grid_square_size - 2,
            env->grid_square_size - 2,
            state != EMPTY ? WHITE : BLANK
        );
        // draw agent
        for (int j = 0; j < env->num_agents; j++) {
            if (env->agent_locations[j] != i) {
                continue;
            }
            int starting_sprite_x = 0;
            int rotation = 90*env->agent_directions[j];
            if (rotation == 180) {
                starting_sprite_x = 128;
                rotation = 0;
            }
            DrawTexturePro(
                client->puffers,
                (Rectangle){starting_sprite_x, 0, 128, 128},
                (Rectangle){
                    x_pos + env->grid_square_size/2,
                    y_pos + env->grid_square_size/2,
                    env->grid_square_size,
                    env->grid_square_size
                },
                (Vector2){env->grid_square_size/2, env->grid_square_size/2},
                rotation,
                //env->agent_states[j] != UNLOADED ? RED : WHITE
                WHITE
            );
            // put a number on top of the agent
            DrawText(
                TextFormat("%d", j),
                x_pos + env->grid_square_size/2,
                y_pos + env->grid_square_size/2,
                20,
                WHITE
            );
        }
    }
    ClearBackground(PUFF_BACKGROUND);    
    EndDrawing();
}
void close_client(Client* client) {
    CloseWindow();
    free(client);
}
