/* FourRooms: Classic four room navigation environment
 * Agent has 7x7 partial observation and must navigate to goal
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char RIGHT = 2;
const unsigned char FORWARD = 3;

const unsigned char EMPTY = 0;
const unsigned char WALL = 1;
const unsigned char AGENT = 2;
const unsigned char GOAL = 3;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log;
    unsigned char* observations; // 7x7 partial view
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size; // Grid size (default 19)
    int tick;
    int agent_x, agent_y;
    int agent_dir; // 0=East, 1=South, 2=West, 3=North
    int goal_x, goal_y;
    unsigned char* grid; // Full grid state
    int seed; // Environment-specific seed
} FourRooms;

void add_log(FourRooms* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void generate_observation(FourRooms* env) {
    // Generate 7x7 observation centered on agent's view direction
    int view_size = 7;
    int half_view = view_size / 2;
    
    // Calculate the center of the view based on agent's direction
    int center_x = env->agent_x;
    int center_y = env->agent_y;
    
    // Shift center forward in the direction the agent is facing
    if (env->agent_dir == 0) center_x += half_view; // East
    else if (env->agent_dir == 1) center_y += half_view; // South
    else if (env->agent_dir == 2) center_x -= half_view; // West
    else if (env->agent_dir == 3) center_y -= half_view; // North
    
    for (int i = 0; i < view_size; i++) {
        for (int j = 0; j < view_size; j++) {
            int world_x = center_x - half_view + j;
            int world_y = center_y - half_view + i;
            
            int obs_idx = i * view_size + j;
            
            // Check bounds
            if (world_x < 0 || world_x >= env->size || world_y < 0 || world_y >= env->size) {
                env->observations[obs_idx] = WALL;
            } else {
                int grid_idx = world_y * env->size + world_x;
                env->observations[obs_idx] = env->grid[grid_idx];
            }
        }
    }
}

void create_four_rooms_grid(FourRooms* env) {
    int size = env->size;
    
    // Clear grid
    memset(env->grid, EMPTY, size * size * sizeof(unsigned char));
    
    // Create outer walls
    for (int i = 0; i < size; i++) {
        env->grid[0 * size + i] = WALL; // Top wall
        env->grid[(size-1) * size + i] = WALL; // Bottom wall
        env->grid[i * size + 0] = WALL; // Left wall
        env->grid[i * size + (size-1)] = WALL; // Right wall
    }
    
    int room_w = size / 2;
    int room_h = size / 2;
    
    // Create vertical separating wall (with gap)
    for (int y = 0; y < size; y++) {
        env->grid[y * size + room_w] = WALL;
    }
    
    // Create horizontal separating wall (with gap)
    for (int x = 0; x < size; x++) {
        env->grid[room_h * size + x] = WALL;
    }
    
    // Create 4 gaps in the separating walls
    // Gap in vertical wall (top half)
    int gap_y1 = 1 + rand() % (room_h - 2);
    env->grid[gap_y1 * size + room_w] = EMPTY;
    
    // Gap in vertical wall (bottom half)
    int gap_y2 = room_h + 1 + rand() % (room_h - 2);
    env->grid[gap_y2 * size + room_w] = EMPTY;
    
    // Gap in horizontal wall (left half)
    int gap_x1 = 1 + rand() % (room_w - 2);
    env->grid[room_h * size + gap_x1] = EMPTY;
    
    // Gap in horizontal wall (right half)
    int gap_x2 = room_w + 1 + rand() % (room_w - 2);
    env->grid[room_h * size + gap_x2] = EMPTY;
}

void c_reset(FourRooms* env) {
    // Set environment-specific seed
    srand(env->seed + env->tick);  // Adding tick provides variation across resets
    
    create_four_rooms_grid(env);
    
    // Place agent randomly in valid position
    do {
        env->agent_x = 1 + rand() % (env->size - 2);
        env->agent_y = 1 + rand() % (env->size - 2);
    } while (env->grid[env->agent_y * env->size + env->agent_x] != EMPTY);
    
    // Place goal randomly in valid position (different from agent)
    do {
        env->goal_x = 1 + rand() % (env->size - 2);
        env->goal_y = 1 + rand() % (env->size - 2);
    } while (env->grid[env->goal_y * env->size + env->goal_x] != EMPTY ||
             (env->goal_x == env->agent_x && env->goal_y == env->agent_y));
    
    // Set agent and goal on grid
    env->grid[env->agent_y * env->size + env->agent_x] = AGENT;
    env->grid[env->goal_y * env->size + env->goal_x] = GOAL;
    
    // Random initial direction
    env->agent_dir = rand() % 4;
    env->tick = 0;
    
    generate_observation(env);
}

void c_step(FourRooms* env) {
    env->tick += 1;
    
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    
    // Clear agent from current position
    env->grid[env->agent_y * env->size + env->agent_x] = EMPTY;
    
    int new_x = env->agent_x;
    int new_y = env->agent_y;
    int new_dir = env->agent_dir;
    
    if (action == LEFT) {
        new_dir = (env->agent_dir + 3) % 4; // Turn left
    } else if (action == RIGHT) {
        new_dir = (env->agent_dir + 1) % 4; // Turn right
    } else if (action == FORWARD) {
        // Move forward in current direction
        if (env->agent_dir == 0) new_x += 1; // East
        else if (env->agent_dir == 1) new_y += 1; // South
        else if (env->agent_dir == 2) new_x -= 1; // West
        else if (env->agent_dir == 3) new_y -= 1; // North
        
        // Check if move is valid (not into wall)
        if (new_x >= 0 && new_x < env->size && new_y >= 0 && new_y < env->size &&
            env->grid[new_y * env->size + new_x] != WALL) {
            env->agent_x = new_x;
            env->agent_y = new_y;
        }
    }
    
    env->agent_dir = new_dir;
    
    // Check if agent reached goal
    if (env->agent_x == env->goal_x && env->agent_y == env->goal_y) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        // Update seed for next episode to ensure variety
        env->seed += 1;
        c_reset(env);
        return;
    }
    
    // Place agent back on grid
    env->grid[env->agent_y * env->size + env->agent_x] = AGENT;
    
    // Check timeout
    if (env->tick >= env->size * env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0;
        add_log(env);
        // Update seed for next episode to ensure variety
        env->seed += 1;
        c_reset(env);
        return;
    }
    
    generate_observation(env);
}

void c_render(FourRooms* env) {
    if (!IsWindowReady()) {
        InitWindow(32*env->size, 32*env->size, "PufferLib FourRooms");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){40, 40, 40, 255});

    int px = 32;
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            int cell = env->grid[y * env->size + x];
            Color color = (Color){60, 60, 60, 255}; // Empty
            
            if (cell == WALL) color = (Color){80, 80, 80, 255};
            else if (cell == AGENT) color = (Color){0, 187, 187, 255};
            else if (cell == GOAL) color = (Color){0, 187, 0, 255};
            
            if (cell != EMPTY) {
                DrawRectangle(x*px, y*px, px, px, color);
            }
        }
    }

    EndDrawing();
}

void c_close(FourRooms* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env->grid) {
        free(env->grid);
    }
}
