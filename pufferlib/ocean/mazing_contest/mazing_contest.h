#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "raylib.h"

#define GRID_WIDTH 10
#define GRID_HEIGHT 10
#define GRID_SIZE (GRID_WIDTH * GRID_HEIGHT)
#define MAX_TOWERS 80

#define ACTION_BUILD_WALL_START 0
#define ACTION_BUILD_WALL_END (GRID_SIZE - 1)
#define ACTION_BUILD_THUNDERCLAP_START GRID_SIZE
#define ACTION_BUILD_THUNDERCLAP_END (2 * GRID_SIZE - 1)
#define TOTAL_ACTIONS (2 * GRID_SIZE)

#define NEIGHBORS_COUNT 4
#define GRID_CELL_WORLD_SIZE 32.0f
#define GRID_CELL_HALF_SIZE 16.0f
#define THUNDERCLAP_EFFECT_DURATION 30

#define MAX_EDGE_TOUCHES 9
#define EFFICIENCY_RATIO_CAP 1.0f
#define MAX_EFFICIENCY_REWARD 10.0f
#define THUNDERCLAP_PENALTY_PER_TOWER 20.0f
#define OVERLAPPING_SLOW_PENALTY 5.0f

#define TOWER_EDGES_COUNT 4
#define THUNDERCLAP_REWARD_MULTIPLIER 2.0f
#define TOUCH_REWARD_EXPONENTIAL_BASE 4.0f
#define MIN_TOUCH_COUNT_FOR_REWARD 2
#define MIN_DISTANCE_FROM_ENTRANCE_EXIT 2
#define FALLBACK_GOAL_POSITION (GRID_WIDTH / 2)


#define GRID_INDEX(x, y) ((y) * GRID_WIDTH + (x))



typedef enum {
    PHASE_BUILD = 0,
    PHASE_RUN = 1
} GamePhase;

typedef enum {
    TOWER_WALL = 0,
    TOWER_THUNDERCLAP = 1
} TowerType;

typedef struct {
    float x, y;
    
    int active;
    int grid_x, grid_y;
    int cooldown;
    int last_activated_tick;
    int touch_count;
    int currently_touching_edges[MAX_EDGE_TOUCHES];
    int previously_touching_edges[MAX_EDGE_TOUCHES];
    
    TowerType type;
    int touched_by_runner;
    int is_random_obstacle;
    int caused_new_slow;
} Tower;

typedef struct {
    int active;
    float x, y;
    int duration;
    int max_duration;
} ThunderclapEffect;

typedef struct {
    float x, y;
    float speed;
    int slowed_until_tick;
    int stuck_counter;
} Runner;

typedef struct {
    float episode_return;
    int episode_length;
    int n;
    int towers_built;
    int total_gold_spent;
    float best_time;
    int rounds_completed;
    float average_path_length;
    int walls_built;
    int final_path_length;
    float path_length_rewards;
    int thunderclap_towers_built;
    float thunderclap_slowdown_time;
    float thunderclap_rewards;
    int wall_touches;
    float wall_touch_rewards;
    int wall_touch_1;
    int wall_touch_2;
    int wall_touch_3;
    int wall_touch_4;
    int wall_touch_5;
    int wall_touch_6;
    int goal_visits;
    float efficiency_rewards;
    float efficiency_percentage;
} Log;

typedef struct Client Client;
struct Client {
    int cell_size;
    int width;
    int height;
};

typedef struct {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int moves_left;

    GamePhase phase;
    int phase_timer;
    int round_number;
    int maze_completed;
    int goal_visited;

    int gold;
    int lumber;
    float runner_time;
    float best_time;
    float episode_return;

    int entrance_x, entrance_y;
    int exit_x, exit_y;
    int goal_x, goal_y;
    int previous_path_length;
    int rounds_completed;

    int num_towers;
    int num_thunderclap_effects;

    int build_time_limit;
    int max_moves;
    int max_rounds;
    int min_gold;
    int max_gold;
    int min_lumber;
    int max_lumber;
    int min_random_walls;
    int max_random_walls;
    int thunderclap_slowdown_duration;
    int thunderclap_cooldown;
    int stuck_teleport_ticks;

    float grid_cell_size;
    float path_length_reward_multiplier;
    float wall_touch_base_reward;
    float thunderclap_slowdown_factor;
    float thunderclap_range;
    float runner_base_speed;
    float thunderclap_slowdown_reward;
    float fps;
    float gold_normalization;
    float lumber_normalization;
    float phase_normalization;
    float build_time_normalization;
    float runner_time_normalization;
    float cell_size_render;
    float total_path_length_rewards;
    float total_path_length;
    float total_thunderclap_rewards;
    float total_wall_touch_rewards;
    float total_efficiency_rewards;
    float total_efficiency_percentage;

    Runner runner;
    Client *client;

    Tower towers[MAX_TOWERS];
    ThunderclapEffect thunderclap_effects[MAX_TOWERS];
    int grid[GRID_WIDTH][GRID_HEIGHT];
    int tower_grid[GRID_WIDTH][GRID_HEIGHT];
} MazingContest;

#define OBS_SIZE (GRID_SIZE + 8)

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static int get_tower_cost(TowerType type) {
    switch (type) {
        case TOWER_WALL: return 1;
        case TOWER_THUNDERCLAP: return 0;
        default: return 1;
    }
}

static int get_tower_lumber_cost(TowerType type) {
    switch (type) {
        case TOWER_WALL: return 0;
        case TOWER_THUNDERCLAP: return 1;
        default: return 0;
    }
}

static void grid_to_world(int grid_x, int grid_y, float* world_x, float* world_y) {
    *world_x = grid_x * GRID_CELL_WORLD_SIZE + GRID_CELL_HALF_SIZE;
    *world_y = grid_y * GRID_CELL_WORLD_SIZE + GRID_CELL_HALF_SIZE;
}

static void world_to_grid(float world_x, float world_y, int* grid_x, int* grid_y) {
    *grid_x = (int)(world_x / GRID_CELL_WORLD_SIZE);
    *grid_y = (int)(world_y / GRID_CELL_WORLD_SIZE);
}

static int is_valid_position(int x, int y) {
    return x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT;
}

static float fast_distance_normalize(float dx, float dy) {
    return sqrtf(dx * dx + dy * dy);
}

// ============================================================================
// PATHFINDING FUNCTIONS
// ============================================================================

static int find_path(const MazingContest* env, int start_x, int start_y, int end_x, int end_y);
static int calculate_path_length(const MazingContest* env, int start_x, int start_y, int end_x, int end_y);
static int find_next_step(const MazingContest* env, int start_x, int start_y, int end_x, int end_y, int* next_x, int* next_y);

static int calculate_total_path_length(const MazingContest* env) {
    int entrance_to_goal = calculate_path_length(env, env->entrance_x, env->entrance_y, env->goal_x, env->goal_y);
    int goal_to_exit = calculate_path_length(env, env->goal_x, env->goal_y, env->exit_x, env->exit_y);
    return entrance_to_goal + goal_to_exit;
}

static int is_path_valid(const MazingContest* env) {
    return find_path(env, env->entrance_x, env->entrance_y, env->goal_x, env->goal_y) &&
           find_path(env, env->goal_x, env->goal_y, env->exit_x, env->exit_y);
}

static void get_runner_target(const MazingContest* env, int* target_x, int* target_y) {
    if (!env->goal_visited) {
        *target_x = env->goal_x;
        *target_y = env->goal_y;
    } else {
        *target_x = env->exit_x;
        *target_y = env->exit_y;
    }
}

static int find_next_step(const MazingContest* env, int start_x, int start_y, int end_x, int end_y, int* next_x, int* next_y) {
    if (start_x == end_x && start_y == end_y) {
        *next_x = start_x;
        *next_y = start_y;
        return 1;
    }
    
    static int visited[GRID_SIZE];
    static int parent_x[GRID_SIZE];
    static int parent_y[GRID_SIZE];
    static int queue_x[GRID_SIZE];
    static int queue_y[GRID_SIZE];
    
    memset(visited, 0, sizeof(visited));
    
    int queue_front = 0, queue_back = 0;
    
    queue_x[queue_back] = start_x;
    queue_y[queue_back] = start_y;
    queue_back++;
    int start_idx = GRID_INDEX(start_x, start_y);
    visited[start_idx] = 1;
    parent_x[start_idx] = -1;
    parent_y[start_idx] = -1;
    
    static const int dx[NEIGHBORS_COUNT] = {0, 0, 1, -1};
    static const int dy[NEIGHBORS_COUNT] = {1, -1, 0, 0};
    
    while (queue_front < queue_back) {
        int x = queue_x[queue_front];
        int y = queue_y[queue_front];
        queue_front++;
        
        if (x == end_x && y == end_y) {
            int path_x = end_x, path_y = end_y;
            int prev_x = path_x, prev_y = path_y;
            
            while (true) {
                int path_idx = GRID_INDEX(path_x, path_y);
                if (parent_x[path_idx] == -1) break;
                
                prev_x = path_x;
                prev_y = path_y;
                int temp_x = parent_x[path_idx];
                int temp_y = parent_y[path_idx];
                path_x = temp_x;
                path_y = temp_y;
                
                if (path_x == start_x && path_y == start_y) {
                    *next_x = prev_x;
                    *next_y = prev_y;
                    return 1;
                }
            }
            
            *next_x = start_x;
            *next_y = start_y;
            return 1;
        }
        
        for (int i = 0; i < NEIGHBORS_COUNT; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
                int neighbor_idx = GRID_INDEX(nx, ny);
                if (!visited[neighbor_idx] && 
                    (env->grid[nx][ny] == 0 || (nx == end_x && ny == end_y))) {
                    visited[neighbor_idx] = 1;
                    parent_x[neighbor_idx] = x;
                    parent_y[neighbor_idx] = y;
                    queue_x[queue_back] = nx;
                    queue_y[queue_back] = ny;
                    queue_back++;
                }
            }
        }
    }
    
    return 0;
}

static int find_path(const MazingContest* env, int start_x, int start_y, int end_x, int end_y) {
    int dummy_x, dummy_y;
    return find_next_step(env, start_x, start_y, end_x, end_y, &dummy_x, &dummy_y);
}

static int calculate_path_length(const MazingContest* env, int start_x, int start_y, int end_x, int end_y) {
    if (start_x == end_x && start_y == end_y) {
        return 0;
    }
    
    
    static int visited[GRID_SIZE];
    static int parent_x[GRID_SIZE];
    static int parent_y[GRID_SIZE];
    static int queue_x[GRID_SIZE];
    static int queue_y[GRID_SIZE];
    
    memset(visited, 0, sizeof(visited));
    
    int queue_front = 0, queue_back = 0;
    
    queue_x[queue_back] = start_x;
    queue_y[queue_back] = start_y;
    queue_back++;
    int start_idx = GRID_INDEX(start_x, start_y);
    visited[start_idx] = 1;
    parent_x[start_idx] = -1;
    parent_y[start_idx] = -1;
    
    static const int dx[NEIGHBORS_COUNT] = {0, 0, 1, -1};
    static const int dy[NEIGHBORS_COUNT] = {1, -1, 0, 0};
    
    while (queue_front < queue_back) {
        int x = queue_x[queue_front];
        int y = queue_y[queue_front];
        queue_front++;
        
        if (x == end_x && y == end_y) {
            int path_length = 0;
            int curr_x = x, curr_y = y;
            
            while (curr_x != start_x || curr_y != start_y) {
                path_length++;
                int curr_idx = GRID_INDEX(curr_x, curr_y);
                if (parent_x[curr_idx] == -1) break;
                int temp_x = parent_x[curr_idx];
                int temp_y = parent_y[curr_idx];
                curr_x = temp_x;
                curr_y = temp_y;
            }
            
            return path_length;
        }
        
        for (int i = 0; i < NEIGHBORS_COUNT; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
                int neighbor_idx = GRID_INDEX(nx, ny);
                if (!visited[neighbor_idx] && 
                    (env->grid[nx][ny] == 0 || (nx == end_x && ny == end_y))) {
                    visited[neighbor_idx] = 1;
                    parent_x[neighbor_idx] = x;
                    parent_y[neighbor_idx] = y;
                    queue_x[queue_back] = nx;
                    queue_y[queue_back] = ny;
                    queue_back++;
                }
            }
        }
    }
    
    return -1;
}

// ============================================================================
// GAME LOGIC FUNCTIONS
// ============================================================================

static void place_tower(MazingContest* env, int x, int y, TowerType type) {
    if (!is_valid_position(x, y) || env->grid[x][y] != 0) return;
    
    if ((x == env->entrance_x && y == env->entrance_y) ||
        (x == env->exit_x && y == env->exit_y)) {
        return;
    }
    
    if (env->gold < get_tower_cost(type) || env->lumber < get_tower_lumber_cost(type)) return;
    
    env->grid[x][y] = (type == TOWER_WALL) ? 1 : 2;
    
    if (!is_path_valid(env)) {
        env->grid[x][y] = 0;
        return;
    }
    
    Tower* tower = &env->towers[env->num_towers];
    tower->active = 1;
    tower->type = type;
    tower->grid_x = x;
    tower->grid_y = y;
    grid_to_world(x, y, &tower->x, &tower->y);
    tower->cooldown = 0;
    tower->caused_new_slow = 0;
    tower->last_activated_tick = -1;
    tower->touched_by_runner = 0;
    tower->touch_count = 0;
    tower->is_random_obstacle = 0;
    
    memset(tower->currently_touching_edges, 0, sizeof(tower->currently_touching_edges));
    memset(tower->previously_touching_edges, 0, sizeof(tower->previously_touching_edges));
    
    env->tower_grid[x][y] = env->num_towers;
    env->num_towers++;
    
    if (env->phase == PHASE_BUILD) {
        int current_path_length = calculate_total_path_length(env);
        
        if (current_path_length > env->previous_path_length) {
            int path_increase = current_path_length - env->previous_path_length;
            float reward = path_increase * env->path_length_reward_multiplier;
            env->rewards[0] += reward;
            env->total_path_length_rewards += reward;
        }
        env->previous_path_length = current_path_length;
    }
    
    env->gold -= get_tower_cost(type);
    env->lumber -= get_tower_lumber_cost(type);
}

static void place_random_obstacle(MazingContest* env, int x, int y, TowerType type) {
    if (!is_valid_position(x, y) || env->grid[x][y] != 0) return;
    
    if ((x == env->entrance_x && y == env->entrance_y) ||
        (x == env->exit_x && y == env->exit_y)) {
        return;
    }
    
    env->grid[x][y] = (type == TOWER_WALL) ? 1 : 2;
    
    if (!is_path_valid(env)) {
        env->grid[x][y] = 0;
        return;
    }
    
    Tower* tower = &env->towers[env->num_towers];
    tower->active = 1;
    tower->type = type;
    tower->grid_x = x;
    tower->grid_y = y;
    grid_to_world(x, y, &tower->x, &tower->y);
    tower->cooldown = 0;
    tower->caused_new_slow = 0;
    tower->last_activated_tick = -1;
    tower->touched_by_runner = 0;
    tower->touch_count = 0;
    tower->is_random_obstacle = 1;
    
    memset(tower->currently_touching_edges, 0, sizeof(tower->currently_touching_edges));
    memset(tower->previously_touching_edges, 0, sizeof(tower->previously_touching_edges));
    
    env->tower_grid[x][y] = env->num_towers;
    env->num_towers++;
}

static void check_tower_proximity(MazingContest* env) {
    int runner_x, runner_y;
    world_to_grid(env->runner.x, env->runner.y, &runner_x, &runner_y);
    
    // Only check towers that are close to the runner
    for (int t = 0; t < env->num_towers; t++) {
        Tower* tower = &env->towers[t];
        if (!tower->active) continue;
        
        // Fast spatial check - only process towers within proximity
        int dx = abs(tower->grid_x - runner_x);
        int dy = abs(tower->grid_y - runner_y);
        if (dx > 1 || dy > 1) {
            // Tower is too far, just reset its touching state
            memset(tower->currently_touching_edges, 0, sizeof(tower->currently_touching_edges));
            continue;
        }
        
        memcpy(tower->previously_touching_edges, tower->currently_touching_edges, sizeof(tower->currently_touching_edges));
        memset(tower->currently_touching_edges, 0, sizeof(tower->currently_touching_edges));
    }
    
    static const int offsets[NEIGHBORS_COUNT][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < NEIGHBORS_COUNT; i++) {
        int check_x = runner_x + offsets[i][0];
        int check_y = runner_y + offsets[i][1];
        
        if (check_x < 0 || check_x >= GRID_WIDTH || check_y < 0 || check_y >= GRID_HEIGHT) continue;
        
        int tower_idx = env->tower_grid[check_x][check_y];
        if (tower_idx == -1) continue;
        
        Tower* tower = &env->towers[tower_idx];
        if (!tower->active) continue;
        
        int edge_index = i;
        int was_touching_this_edge = tower->previously_touching_edges[edge_index];
        tower->currently_touching_edges[edge_index] = 1;
        
        if (!was_touching_this_edge) {
            if (!tower->touched_by_runner) {
                tower->touched_by_runner = 1;
                tower->touch_count = 1;
            } else {
                tower->touch_count++;
            }
            
            if (tower->touch_count >= MIN_TOUCH_COUNT_FOR_REWARD) {
                float exponential_multiplier = powf(TOUCH_REWARD_EXPONENTIAL_BASE, (float)(tower->touch_count - MIN_TOUCH_COUNT_FOR_REWARD));
                float base_reward = env->wall_touch_base_reward;
                
                if (tower->type == TOWER_THUNDERCLAP) {
                    base_reward *= THUNDERCLAP_REWARD_MULTIPLIER;
                }
                
                float reward = base_reward * exponential_multiplier;
                env->rewards[0] += reward;
                env->total_wall_touch_rewards += reward;
            }
        }
    }
}

static void move_runner(MazingContest* env) {
    Runner* runner = &env->runner;
    
    int current_x, current_y;
    world_to_grid(runner->x, runner->y, &current_x, &current_y);
    
    if (!env->goal_visited &&
        current_x == env->goal_x && current_y == env->goal_y) {
        env->goal_visited = 1;
        env->log.goal_visits++;
        return;
    }
    
    if (env->goal_visited &&
        current_x == env->exit_x && current_y == env->exit_y) {
        env->maze_completed = 1;
        return;
    }
    
    int next_x, next_y;
    int target_x, target_y;
    get_runner_target(env, &target_x, &target_y);
    
    if (find_next_step(env, current_x, current_y, target_x, target_y, &next_x, &next_y)) {
        float target_world_x, target_world_y;
        grid_to_world(next_x, next_y, &target_world_x, &target_world_y);
        
        float speed = runner->speed;
        if (env->tick < runner->slowed_until_tick) {
            speed *= env->thunderclap_slowdown_factor;
            
            env->rewards[0] += env->thunderclap_slowdown_reward;
            env->total_thunderclap_rewards += env->thunderclap_slowdown_reward;
            
            env->log.thunderclap_slowdown_time += 1.0f / env->fps;
        }
        
        float dx = target_world_x - runner->x;
        float dy = target_world_y - runner->y;
        float dist = fast_distance_normalize(dx, dy);
        
        if (dist > speed) {
            runner->x += (dx / dist) * speed;
            runner->y += (dy / dist) * speed;
        } else {
            runner->x = target_world_x;
            runner->y = target_world_y;
        }
        
        runner->stuck_counter = 0;
    } else {
        runner->stuck_counter++;
        if (runner->stuck_counter > env->stuck_teleport_ticks) {
            grid_to_world(env->exit_x, env->exit_y, &runner->x, &runner->y);
            env->maze_completed = 1;
        }
    }
}

static void update_thunderclap_effects(MazingContest* env) {
    for (int i = 0; i < env->num_thunderclap_effects; i++) {
        ThunderclapEffect* effect = &env->thunderclap_effects[i];
        if (!effect->active) continue;
        
        effect->duration--;
        if (effect->duration <= 0) {
            effect->active = 0;
            *effect = env->thunderclap_effects[env->num_thunderclap_effects - 1];
            env->num_thunderclap_effects--;
            i--;
        }
    }
}

static void update_thunderclap_towers(MazingContest* env) {
    const float runner_x = env->runner.x;
    const float runner_y = env->runner.y;
    const float range_sq = env->thunderclap_range * env->thunderclap_range;
    
    const int was_slowed_at_start = (env->tick < env->runner.slowed_until_tick);
    int new_slow_credit_given = 0;
    
    for (int i = env->num_towers - 1; i >= 0; i--) {
        Tower* tower = &env->towers[i];
        if (!tower->active || tower->type != TOWER_THUNDERCLAP || tower->cooldown > 0) {
            if (tower->cooldown > 0) tower->cooldown--;
            continue;
        }
        
        const float dx = tower->x - runner_x;
        const float dy = tower->y - runner_y;
        const float dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= range_sq) {
            env->runner.slowed_until_tick = env->tick + env->thunderclap_slowdown_duration;
            tower->cooldown = env->thunderclap_cooldown;
            tower->last_activated_tick = env->tick;
            
            if (!was_slowed_at_start && !new_slow_credit_given) {
                tower->caused_new_slow = 1;
                new_slow_credit_given = 1;
            } else if (was_slowed_at_start) {
                env->rewards[0] -= OVERLAPPING_SLOW_PENALTY;
                env->total_thunderclap_rewards -= OVERLAPPING_SLOW_PENALTY;
            }
            
            if (env->num_thunderclap_effects < MAX_TOWERS) {
                ThunderclapEffect* effect = &env->thunderclap_effects[env->num_thunderclap_effects];
                effect->active = 1;
                effect->x = tower->x;
                effect->y = tower->y;
                effect->duration = THUNDERCLAP_EFFECT_DURATION;
                effect->max_duration = THUNDERCLAP_EFFECT_DURATION;
                env->num_thunderclap_effects++;
            }
        }
    }
}

void compute_observations(MazingContest* env) {
    float* obs = env->observations;
    int idx = 0;
    
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            obs[idx++] = env->grid[x][y] / 2.0f;
        }
    }
    
    obs[idx++] = (float)env->gold / env->gold_normalization;
    obs[idx++] = (float)env->lumber / env->lumber_normalization;
    
    obs[idx++] = env->phase / env->phase_normalization;
    obs[idx++] = env->phase_timer / env->build_time_normalization;
    obs[idx++] = env->runner_time / env->runner_time_normalization;
    
    obs[idx++] = 1.0f;
    obs[idx++] = (float)env->goal_x / (float)GRID_WIDTH;
    obs[idx++] = (float)env->goal_y / (float)GRID_HEIGHT;
}

void add_log(MazingContest* env) {
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.n += 1;
    
    int player_towers = 0;
    int thunderclap_towers = 0;
    
    env->log.total_gold_spent += (env->max_gold - env->gold);
    env->log.best_time = env->best_time;
    env->log.rounds_completed += env->rounds_completed;
    
    env->log.final_path_length = calculate_total_path_length(env);
    env->log.path_length_rewards += env->total_path_length_rewards;
    
    env->total_path_length += env->log.final_path_length;
    if (env->rounds_completed > 0) {
        env->log.average_path_length = env->total_path_length / env->rounds_completed;
    }
    
    for (int i = 0; i < env->num_towers; i++) {
        Tower* tower = &env->towers[i];
        
        if (!tower->is_random_obstacle) {
            player_towers++;
            if (tower->type == TOWER_WALL) {
                env->log.walls_built++;
            } else if (tower->type == TOWER_THUNDERCLAP) {
                thunderclap_towers++;
            }
        }
        
        if (tower->active && (tower->type == TOWER_WALL || tower->type == TOWER_THUNDERCLAP)) {
            env->log.wall_touches += tower->touch_count;
            
            if (tower->touch_count == 1) {
                env->log.wall_touch_1++;
            } else if (tower->touch_count == 2) {
                env->log.wall_touch_2++;
            } else if (tower->touch_count == 3) {
                env->log.wall_touch_3++;
            } else if (tower->touch_count == 4) {
                env->log.wall_touch_4++;
            } else if (tower->touch_count == 5) {
                env->log.wall_touch_5++;
            } else if (tower->touch_count >= 6) {
                env->log.wall_touch_6++;
            }
        }
    }
    
    env->log.towers_built += player_towers;
    env->log.thunderclap_towers_built += thunderclap_towers;
    
    env->log.wall_touch_rewards += env->total_wall_touch_rewards;
    
    env->log.efficiency_rewards += env->total_efficiency_rewards;
    if (env->rounds_completed > 0) {
        env->log.efficiency_percentage = (env->total_efficiency_percentage / env->rounds_completed) * 100.0f;
    }
    env->log.thunderclap_rewards += env->total_thunderclap_rewards;
}

void place_obstacles(MazingContest* env);
void place_goal(MazingContest* env);

static void init_round(MazingContest* env) {
    memset(env->grid, 0, sizeof(env->grid));
    
    env->num_towers = 0;
    memset(env->towers, 0, sizeof(env->towers));
    
    memset(env->tower_grid, -1, sizeof(env->tower_grid));
    
    env->num_thunderclap_effects = 0;
    memset(env->thunderclap_effects, 0, sizeof(env->thunderclap_effects));
    
    env->entrance_x = 0;
    env->entrance_y = GRID_HEIGHT / 2;
    env->exit_x = GRID_WIDTH - 1;
    env->exit_y = GRID_HEIGHT / 2;
    
    int gold_range = env->max_gold - env->min_gold + 1;
    int lumber_range = env->max_lumber - env->min_lumber + 1;
    
    env->gold = env->min_gold + (rand() % gold_range);
    env->lumber = env->min_lumber + (rand() % lumber_range);
    
    env->phase = PHASE_BUILD;
    env->phase_timer = 0;
    env->runner_time = 0.0f;
    env->maze_completed = 0;
    
    grid_to_world(env->entrance_x, env->entrance_y, &env->runner.x, &env->runner.y);
    env->runner.speed = env->runner_base_speed;
    env->runner.slowed_until_tick = 0;
    env->runner.stuck_counter = 0;
    
    place_obstacles(env);
    
    place_goal(env);
    env->goal_visited = 0;
    
    
    env->previous_path_length = calculate_total_path_length(env);
}

void place_obstacles(MazingContest* env) {
    static int valid_positions[GRID_SIZE][2];
    int valid_count = 0;
    
    for (int x = 1; x < GRID_WIDTH - 1; x++) {
        for (int y = 1; y < GRID_HEIGHT - 1; y++) {
            if (env->grid[x][y] == 0 && 
                !(x == env->entrance_x && y == env->entrance_y) &&
                !(x == env->exit_x && y == env->exit_y)) {
                valid_positions[valid_count][0] = x;
                valid_positions[valid_count][1] = y;
                valid_count++;
            }
        }
    }
    
    if (valid_count == 0) return;
    
    int num_walls = env->min_random_walls;
    if (env->max_random_walls > env->min_random_walls) {
        num_walls += rand() % (env->max_random_walls - env->min_random_walls + 1);
    }
    
    for (int i = 0; i < num_walls && i < valid_count; i++) {
        int rand_idx = i + rand() % (valid_count - i);
        int temp_x = valid_positions[i][0];
        int temp_y = valid_positions[i][1];
        valid_positions[i][0] = valid_positions[rand_idx][0];
        valid_positions[i][1] = valid_positions[rand_idx][1];
        valid_positions[rand_idx][0] = temp_x;
        valid_positions[rand_idx][1] = temp_y;
        
        place_random_obstacle(env, valid_positions[i][0], valid_positions[i][1], TOWER_WALL);
    }
}

void place_goal(MazingContest* env) {
    static int valid_positions[GRID_SIZE][2];
    int valid_count = 0;
    
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (env->grid[x][y] != 0) continue;
            
            if ((x == env->entrance_x && y == env->entrance_y) ||
                (x == env->exit_x && y == env->exit_y)) continue;
            
            int dist_to_entrance = abs(x - env->entrance_x) + abs(y - env->entrance_y);
            int dist_to_exit = abs(x - env->exit_x) + abs(y - env->exit_y);
            
            if (dist_to_entrance >= MIN_DISTANCE_FROM_ENTRANCE_EXIT && dist_to_exit >= MIN_DISTANCE_FROM_ENTRANCE_EXIT) {
                valid_positions[valid_count][0] = x;
                valid_positions[valid_count][1] = y;
                valid_count++;
            }
        }
    }
    
    if (valid_count == 0) {
        env->goal_x = FALLBACK_GOAL_POSITION;
        env->goal_y = FALLBACK_GOAL_POSITION;
        return;
    }
    
    for (int attempts = 0; attempts < valid_count; attempts++) {
        int rand_idx = rand() % valid_count;
        int test_x = valid_positions[rand_idx][0];
        int test_y = valid_positions[rand_idx][1];
        
        if (find_path(env, env->entrance_x, env->entrance_y, test_x, test_y) &&
            find_path(env, test_x, test_y, env->exit_x, env->exit_y)) {
            env->goal_x = test_x;
            env->goal_y = test_y;
            return;
        }
        
        valid_positions[rand_idx][0] = valid_positions[valid_count - 1][0];
        valid_positions[rand_idx][1] = valid_positions[valid_count - 1][1];
        valid_count--;
    }
    
    env->goal_x = FALLBACK_GOAL_POSITION;
    env->goal_y = FALLBACK_GOAL_POSITION;
}

void c_reset(MazingContest* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->total_path_length = 0.0f;
    env->rounds_completed = 0;
    env->round_number = 1;
    env->best_time = 0.0f;
    env->moves_left = env->max_moves;
    env->total_path_length_rewards = 0.0f;
    env->total_thunderclap_rewards = 0.0f;
    env->total_wall_touch_rewards = 0.0f;
    
    init_round(env);
    compute_observations(env);
}

void c_step(MazingContest* env) {
    env->tick++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    
    int action = (int)env->actions[0];
    
    if (env->phase == PHASE_BUILD) {
        env->phase_timer++;
        
        if (action >= ACTION_BUILD_WALL_START && action <= ACTION_BUILD_WALL_END) {
            int pos = action - ACTION_BUILD_WALL_START;
            int x = pos % GRID_WIDTH;
            int y = pos / GRID_WIDTH;
            place_tower(env, x, y, TOWER_WALL);
        } 
        else if (action >= ACTION_BUILD_THUNDERCLAP_START && action <= ACTION_BUILD_THUNDERCLAP_END) {
            int pos = action - ACTION_BUILD_THUNDERCLAP_START;
            int x = pos % GRID_WIDTH;
            int y = pos / GRID_WIDTH;
            place_tower(env, x, y, TOWER_THUNDERCLAP);
        }
        
        int can_build_walls = (env->gold >= get_tower_cost(TOWER_WALL));
        int can_build_thunderclap = (env->lumber >= get_tower_lumber_cost(TOWER_THUNDERCLAP));
        int time_limit_reached = (env->phase_timer >= env->build_time_limit);
        
        if ((!can_build_walls && !can_build_thunderclap) || time_limit_reached) {
            int path_exists = is_path_valid(env);
            
            if (path_exists) {
                env->phase = PHASE_RUN;
                env->phase_timer = 0;
            } else {
                env->runner_time = 0.0f;
                
                env->rounds_completed++;
                env->round_number++;
                
                if (env->rounds_completed >= env->max_rounds) {
                    env->terminals[0] = 1;
                    add_log(env);
                    c_reset(env);
                    return;
                }
                
                init_round(env);
            }
        }
    } else if (env->phase == PHASE_RUN) {
        env->phase_timer++;
        env->runner_time = env->phase_timer / env->fps;
        
        move_runner(env);
        check_tower_proximity(env);
        update_thunderclap_towers(env);
        update_thunderclap_effects(env);
        
        if (env->maze_completed) {
            if (env->runner_time > env->best_time) {
                env->best_time = env->runner_time;
            }
            
            int total_blockades = 0;
            int actual_touches = 0;
            int ineffective_thunderclaps = 0;
            
            for (int i = 0; i < env->num_towers; i++) {
                Tower* tower = &env->towers[i];
                if (tower->active) {
                    total_blockades++;
                    actual_touches += tower->touch_count;
                    
                    if (!tower->is_random_obstacle && tower->type == TOWER_THUNDERCLAP && tower->caused_new_slow == 0) {
                        ineffective_thunderclaps++;
                    }
                }
            }
            
            if (total_blockades > 0) {
                const int possible_edges = total_blockades * TOWER_EDGES_COUNT;
                const float efficiency_ratio = (float)actual_touches / (float)possible_edges;
                
                const float capped_ratio = efficiency_ratio > EFFICIENCY_RATIO_CAP ? EFFICIENCY_RATIO_CAP : efficiency_ratio;
                const float efficiency_reward = capped_ratio * MAX_EFFICIENCY_REWARD;
                
                env->rewards[0] += efficiency_reward;
                
                env->total_efficiency_rewards += efficiency_reward;
                env->total_efficiency_percentage += efficiency_ratio;
            }
            if (ineffective_thunderclaps > 0) {
                const float penalty = ineffective_thunderclaps * THUNDERCLAP_PENALTY_PER_TOWER;
                env->rewards[0] -= penalty;
                env->total_thunderclap_rewards -= penalty;
            }
            
            env->rounds_completed++;
            env->round_number++;
            
            if (env->rounds_completed >= env->max_rounds) {
                env->terminals[0] = 1;
                
                add_log(env);
                c_reset(env);
                return;
            }
            
            init_round(env);
        }
    }
    
    env->moves_left--;
    
    if (env->moves_left == 0) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
    
    env->episode_return += env->rewards[0];
    compute_observations(env);
}

// ============================================================================
// MEMORY MANAGEMENT FUNCTIONS
// ============================================================================

void allocate(MazingContest* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->total_path_length = 0.0f;
    env->total_thunderclap_rewards = 0.0f;
    env->total_wall_touch_rewards = 0.0f;
    env->total_efficiency_rewards = 0.0f;
    env->total_efficiency_percentage = 0.0f;
    env->rounds_completed = 0;
    env->client = NULL;
    
    memset(env->tower_grid, -1, sizeof(env->tower_grid));
    
    srand(time(NULL));
    
    if (env->observations == NULL) {
        env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    }
    if (env->actions == NULL) {
        env->actions = (float*)calloc(1, sizeof(float));
    }
    if (env->rewards == NULL) {
        env->rewards = (float*)calloc(1, sizeof(float));
    }
    if (env->terminals == NULL) {
        env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    }
    
    memset(&env->log, 0, sizeof(Log));
    
    env->moves_left = env->max_moves;
    
    init(env);
    init_round(env);
}

// ============================================================================
// RENDERING FUNCTIONS
// ============================================================================

#define UI_PANEL_HEIGHT 110
#define UI_PANEL_MARGIN 10
#define UI_TEXT_LARGE 18
#define UI_TEXT_MEDIUM 14
#define UI_TEXT_SMALL 10
#define UI_TEXT_TINY 8
#define UI_Y_OFFSET 15
#define UI_LINE_HEIGHT 20

#define RUNNER_RADIUS_OUTER 15
#define RUNNER_RADIUS_MIDDLE 10
#define RUNNER_RADIUS_INNER 6
#define RUNNER_RADIUS_OUTLINE 8

#define EFFECT_MAX_RADIUS 60.0f
#define EFFECT_CORE_RADIUS 8
#define EFFECT_GLOW_RADIUS 12
#define EFFECT_LIGHTNING_COUNT 8

// Color definitions
static const Color COLOR_BG_GRADIENT_TOP = {10, 15, 25, 255};
static const Color COLOR_BG_GRADIENT_BOTTOM = {25, 35, 50, 255};
static const Color COLOR_GRID_BASE_1 = {20, 25, 35, 255};
static const Color COLOR_GRID_BASE_2 = {25, 30, 40, 255};
static const Color COLOR_GRID_LINES = {100, 150, 200, 100};

static const Color COLOR_WALL_SHADOW = {0, 0, 0, 80};
static const Color COLOR_WALL_GRAD_TOP = {140, 140, 140, 255};
static const Color COLOR_WALL_GRAD_BOTTOM = {80, 80, 80, 255};
static const Color COLOR_WALL_LINE_OUTER = {180, 180, 180, 255};
static const Color COLOR_WALL_LINE_INNER = {60, 60, 60, 255};

static const Color COLOR_THUNDER_SHADOW = {0, 0, 0, 100};
static const Color COLOR_THUNDER_GRAD_TOP = {138, 43, 226, 255};
static const Color COLOR_THUNDER_GRAD_BOTTOM = {75, 0, 130, 255};
static const Color COLOR_THUNDER_LINE = {186, 85, 211, 255};
static const Color COLOR_THUNDER_READY = {255, 255, 255, 150};

static const Color COLOR_OBSTACLE_MARKER = {255, 255, 255, 200};
static const Color COLOR_ENTRANCE = {0, 255, 100, 200};
static const Color COLOR_EXIT = {255, 50, 50, 200};
static const Color COLOR_GOAL_ACTIVE = {255, 215, 0, 255};
static const Color COLOR_GOAL_VISITED = {255, 215, 0, 100};

static const Color COLOR_RUNNER_NORMAL = {0, 150, 255, 255};
static const Color COLOR_RUNNER_SLOWED = {255, 0, 255, 255};

static const Color COLOR_UI_BG = {10, 15, 25, 240};
static const Color COLOR_UI_ACCENT = {100, 150, 255, 255};
static const Color COLOR_UI_ACCENT_DARK = {50, 100, 200, 255};

static const Color COLOR_PHASE_BUILD = {255, 200, 0, 255};
static const Color COLOR_PHASE_RUN = {0, 255, 0, 255};
static const Color COLOR_PHASE_RESULT = {255, 100, 100, 255};

static const Color COLOR_GOLD = {255, 215, 0, 255};
static const Color COLOR_LUMBER = {139, 69, 19, 255};
static const Color COLOR_SCORE = {255, 255, 100, 255};
static const Color COLOR_INFO = {150, 200, 255, 255};
static const Color COLOR_TIMER_NORMAL = {255, 255, 255, 255};
static const Color COLOR_TIMER_WARNING = {255, 100, 100, 255};
static const Color COLOR_TIMER_RUNNING = {100, 255, 100, 255};

static void render_grid_background(MazingContest* env, float cell_size) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        int y_pos = y * cell_size;
        for (int x = 0; x < GRID_WIDTH; x++) {
            int x_pos = x * cell_size;
            Color base_color = ((x + y) & 1) ? COLOR_GRID_BASE_2 : COLOR_GRID_BASE_1;
            DrawRectangle(x_pos, y_pos, cell_size, cell_size, base_color);
        }
    }
}

static void render_grid_lines(float cell_size) {
    for (int x = 0; x <= GRID_WIDTH; x++) {
        DrawLine(x * cell_size, 0, x * cell_size, GRID_HEIGHT * cell_size, COLOR_GRID_LINES);
    }
    for (int y = 0; y <= GRID_HEIGHT; y++) {
        DrawLine(0, y * cell_size, GRID_WIDTH * cell_size, y * cell_size, COLOR_GRID_LINES);
    }
}

static void render_wall_tower(int x_pos, int y_pos, float cell_size, Tower* tower) {
    DrawRectangle(x_pos + 2, y_pos + 2, cell_size - 2, cell_size - 2, COLOR_WALL_SHADOW);
    DrawRectangleGradientV(x_pos, y_pos, cell_size, cell_size, COLOR_WALL_GRAD_TOP, COLOR_WALL_GRAD_BOTTOM);
    DrawRectangleLines(x_pos, y_pos, cell_size, cell_size, COLOR_WALL_LINE_OUTER);
    DrawRectangleLines(x_pos + 1, y_pos + 1, cell_size - 2, cell_size - 2, COLOR_WALL_LINE_INNER);
    
    if (tower && tower->is_random_obstacle) {
        int quarter = (int)cell_size >> 2;
        for (int gx = 0; gx < cell_size; gx += quarter) {
            DrawLine(x_pos + gx, y_pos, x_pos + gx, y_pos + cell_size, COLOR_OBSTACLE_MARKER);
        }
        for (int gy = 0; gy < cell_size; gy += quarter) {
            DrawLine(x_pos, y_pos + gy, x_pos + cell_size, y_pos + gy, COLOR_OBSTACLE_MARKER);
        }
    }
}

static void render_thunder_tower(int x_pos, int y_pos, float cell_size, Tower* tower) {
    DrawRectangle(x_pos + 2, y_pos + 2, cell_size - 2, cell_size - 2, COLOR_THUNDER_SHADOW);
    DrawRectangleGradientV(x_pos, y_pos, cell_size, cell_size, COLOR_THUNDER_GRAD_TOP, COLOR_THUNDER_GRAD_BOTTOM);
    DrawRectangleLines(x_pos, y_pos, cell_size, cell_size, COLOR_THUNDER_LINE);
    
    if (tower) {
        if (tower->is_random_obstacle) {
            int offset = 4;
            int end_pos = cell_size - 4;
            DrawLine(x_pos + offset, y_pos + offset, x_pos + end_pos, y_pos + end_pos, COLOR_OBSTACLE_MARKER);
            DrawLine(x_pos + end_pos, y_pos + offset, x_pos + offset, y_pos + end_pos, COLOR_OBSTACLE_MARKER);
            DrawLine(x_pos + offset + 1, y_pos + offset, x_pos + end_pos - 1, y_pos + end_pos, COLOR_OBSTACLE_MARKER);
            DrawLine(x_pos + end_pos - 1, y_pos + offset, x_pos + offset + 1, y_pos + end_pos, COLOR_OBSTACLE_MARKER);
        }
        
        if (tower->cooldown == 0 && !tower->is_random_obstacle) {
            DrawRectangleLines(x_pos - 1, y_pos - 1, cell_size + 2, cell_size + 2, COLOR_THUNDER_READY);
            DrawRectangleLines(x_pos - 2, y_pos - 2, cell_size + 4, cell_size + 4, 
                             (Color){COLOR_THUNDER_READY.r, COLOR_THUNDER_READY.g, COLOR_THUNDER_READY.b, 80});
        }
    }
}

static void render_tower_touch_count(int x_pos, int y_pos, float cell_size, int touch_count, bool is_thunder) {
    if (touch_count <= 0) return;
    
    char touch_text[4];
    snprintf(touch_text, sizeof(touch_text), "%d", touch_count);
    
    Color touch_color;
    if (is_thunder) {
        touch_color = touch_count >= 6 ? (Color){255, 255, 100, 255} :
                     touch_count >= 4 ? (Color){255, 200, 100, 255} :
                     touch_count >= 2 ? (Color){200, 255, 255, 255} : WHITE;
    } else {
        touch_color = touch_count >= 6 ? (Color){255, 255, 0, 255} :
                     touch_count >= 4 ? (Color){255, 165, 0, 255} : WHITE;
    }
    
    int text_width = MeasureText(touch_text, UI_TEXT_SMALL);
    int text_x = x_pos + (cell_size - text_width) / 2;
    int text_y = y_pos + (cell_size - UI_TEXT_SMALL) / 2;
    
    DrawText(touch_text, text_x, text_y, UI_TEXT_SMALL, touch_color);
}

static void render_special_locations(MazingContest* env, float cell_size) {
    int entrance_x = env->entrance_x * cell_size;
    int entrance_y = env->entrance_y * cell_size;
    int exit_x = env->exit_x * cell_size;
    int exit_y = env->exit_y * cell_size;
    int goal_x = env->goal_x * cell_size;
    int goal_y = env->goal_y * cell_size;
    
    DrawRectangle(entrance_x + 3, entrance_y + 3, cell_size - 6, cell_size - 6, COLOR_ENTRANCE);
    DrawRectangleLines(entrance_x + 2, entrance_y + 2, cell_size - 4, cell_size - 4, 
                      (Color){COLOR_ENTRANCE.r, COLOR_ENTRANCE.g, COLOR_ENTRANCE.b, 255});
    DrawRectangleLines(entrance_x + 1, entrance_y + 1, cell_size - 2, cell_size - 2, 
                      (Color){255, 255, 255, 150});
    DrawText("START", entrance_x + 5, entrance_y + 12, UI_TEXT_TINY, WHITE);
    
    DrawRectangle(exit_x + 3, exit_y + 3, cell_size - 6, cell_size - 6, COLOR_EXIT);
    DrawRectangleLines(exit_x + 2, exit_y + 2, cell_size - 4, cell_size - 4, 
                      (Color){COLOR_EXIT.r, COLOR_EXIT.g, COLOR_EXIT.b, 255});
    DrawRectangleLines(exit_x + 1, exit_y + 1, cell_size - 2, cell_size - 2, 
                      (Color){255, 255, 255, 150});
    DrawText("EXIT", exit_x + 8, exit_y + 12, UI_TEXT_TINY, WHITE);
    
    Color goal_color = env->goal_visited ? COLOR_GOAL_VISITED : COLOR_GOAL_ACTIVE;
    DrawRectangle(goal_x + 3, goal_y + 3, cell_size - 6, cell_size - 6, goal_color);
    DrawRectangleLines(goal_x + 2, goal_y + 2, cell_size - 4, cell_size - 4, COLOR_GOAL_ACTIVE);
    DrawRectangleLines(goal_x + 1, goal_y + 1, cell_size - 2, cell_size - 2, 
                      (Color){255, 255, 255, 150});
    DrawText("GOAL", goal_x + 5, goal_y + 12, UI_TEXT_TINY, WHITE);
}

static void render_thunderclap_effects(MazingContest* env, float cell_size) {
    for (int i = 0; i < env->num_thunderclap_effects; i++) {
        ThunderclapEffect* effect = &env->thunderclap_effects[i];
        if (!effect->active) continue;
        
        int screen_x = (int)(effect->x * cell_size / env->grid_cell_size);
        int screen_y = (int)(effect->y * cell_size / env->grid_cell_size);
        
        float intensity = (float)effect->duration / effect->max_duration;
        int alpha = (int)(255 * intensity);
        
        float radius = (1.0f - intensity) * EFFECT_MAX_RADIUS;
        DrawCircleLines(screen_x, screen_y, radius, (Color){255, 255, 0, alpha});
        DrawCircleLines(screen_x, screen_y, radius * 0.8f, (Color){255, 150, 0, alpha});
        DrawCircleLines(screen_x, screen_y, radius * 0.6f, (Color){255, 50, 0, alpha});
        
        DrawCircle(screen_x, screen_y, EFFECT_CORE_RADIUS * intensity, (Color){255, 255, 255, alpha});
        DrawCircle(screen_x, screen_y, EFFECT_GLOW_RADIUS * intensity, (Color){255, 255, 0, alpha/2});
        
        if (intensity > 0.5f) {
            for (int j = 0; j < EFFECT_LIGHTNING_COUNT; j++) {
                int offset = (int)(intensity * 3);
                DrawLine(screen_x - 20 + (j * 5), screen_y + offset, 
                        screen_x - 15 + (j * 5), screen_y - offset, 
                        (Color){255, 255, 100, alpha/2});
            }
        }
    }
}

static void render_runner(MazingContest* env, float cell_size) {
    if (env->phase != PHASE_RUN && env->phase != PHASE_BUILD) return;
    
    Color runner_color = (env->tick < env->runner.slowed_until_tick) ? COLOR_RUNNER_SLOWED : COLOR_RUNNER_NORMAL;
    
    if (env->phase == PHASE_BUILD) {
        runner_color = (Color){runner_color.r, runner_color.g, runner_color.b, 150};
    }
    
    int screen_x = (int)(env->runner.x * cell_size / env->grid_cell_size);
    int screen_y = (int)(env->runner.y * cell_size / env->grid_cell_size);
    
    DrawCircle(screen_x, screen_y, RUNNER_RADIUS_OUTER, (Color){runner_color.r, runner_color.g, runner_color.b, 30});
    DrawCircle(screen_x, screen_y, RUNNER_RADIUS_MIDDLE, (Color){runner_color.r, runner_color.g, runner_color.b, 80});
    DrawCircle(screen_x, screen_y, RUNNER_RADIUS_INNER, runner_color);
    DrawCircleLines(screen_x, screen_y, RUNNER_RADIUS_OUTLINE, (Color){255, 255, 255, 200});
}

static void render_ui_panel(MazingContest* env) {
    int ui_y = GRID_HEIGHT * env->cell_size_render + UI_Y_OFFSET;
    
    DrawRectangle(UI_PANEL_MARGIN, ui_y - 5, env->client->width - 2 * UI_PANEL_MARGIN, UI_PANEL_HEIGHT, COLOR_UI_BG);
    DrawRectangleGradientH(UI_PANEL_MARGIN, ui_y - 5, env->client->width - 2 * UI_PANEL_MARGIN, 3, COLOR_UI_ACCENT, COLOR_UI_ACCENT_DARK);
    DrawRectangleLines(UI_PANEL_MARGIN, ui_y - 5, env->client->width - 2 * UI_PANEL_MARGIN, UI_PANEL_HEIGHT, COLOR_UI_ACCENT);
    
    const char* phase_text = env->phase == PHASE_BUILD ? "⚒ BUILD PHASE ⚒" : 
                             env->phase == PHASE_RUN ? "⚡ RUN PHASE ⚡" : "📊 RESULT";
    Color phase_color = env->phase == PHASE_BUILD ? COLOR_PHASE_BUILD : 
                        env->phase == PHASE_RUN ? COLOR_PHASE_RUN : COLOR_PHASE_RESULT;
    DrawText(phase_text, 20, ui_y + 5, UI_TEXT_LARGE, phase_color);
    
    DrawText(TextFormat("💰 Gold: %d", env->gold), 20, ui_y + 30, UI_TEXT_MEDIUM, COLOR_GOLD);
    DrawText(TextFormat("🌲 Lumber: %d", env->lumber), 20, ui_y + 50, UI_TEXT_MEDIUM, COLOR_LUMBER);
    
    DrawText(TextFormat("Round: %d", env->round_number), 200, ui_y + 30, UI_TEXT_MEDIUM, COLOR_INFO);
    DrawText(TextFormat("Score: %.1f", env->episode_return), 200, ui_y + 50, UI_TEXT_MEDIUM, COLOR_SCORE);
    
    if (env->phase == PHASE_BUILD) {
        int time_left = (env->build_time_limit - env->phase_timer) / 60;
        Color timer_color = time_left > 5 ? COLOR_TIMER_NORMAL : COLOR_TIMER_WARNING;
        DrawText(TextFormat("⏱ Build Time: %ds", time_left), 20, ui_y + 70, UI_TEXT_MEDIUM, timer_color);
    } else if (env->phase == PHASE_RUN) {
        DrawText(TextFormat("⏱ Runner Time: %.1fs", env->runner_time), 20, ui_y + 70, UI_TEXT_MEDIUM, COLOR_TIMER_RUNNING);
    } else {
        DrawText(TextFormat("⏱ Final: %.1fs (Best: %.1fs)", env->runner_time, env->best_time), 
                20, ui_y + 70, UI_TEXT_MEDIUM, WHITE);
    }
}

static Client* make_client(MazingContest* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cell_size = env->cell_size_render;
    client->width = GRID_WIDTH * client->cell_size;
    client->height = GRID_HEIGHT * client->cell_size + 100;
    
    InitWindow(client->width, client->height, "PufferLib Mazing Contest");
    SetTargetFPS(60);
    
    return client;
}

void c_close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(MazingContest* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    
    DrawRectangleGradientV(0, 0, env->client->width, env->client->height, 
                          COLOR_BG_GRADIENT_TOP, COLOR_BG_GRADIENT_BOTTOM);
    
    float cell_size = env->client->cell_size;
    
    render_grid_background(env, cell_size);
    
    for (int y = 0; y < GRID_HEIGHT; y++) {
        int y_pos = y * cell_size;
        for (int x = 0; x < GRID_WIDTH; x++) {
            int x_pos = x * cell_size;
            int grid_val = env->grid[x][y];
            int tower_idx = env->tower_grid[x][y];
            Tower* tower = (tower_idx != -1) ? &env->towers[tower_idx] : NULL;
            
            if (grid_val == 1) {
                render_wall_tower(x_pos, y_pos, cell_size, tower);
                if (tower && tower->touch_count > 0) {
                    render_tower_touch_count(x_pos, y_pos, cell_size, tower->touch_count, false);
                }
            } else if (grid_val == 2) {
                render_thunder_tower(x_pos, y_pos, cell_size, tower);
                if (tower && tower->touch_count > 0) {
                    render_tower_touch_count(x_pos, y_pos, cell_size, tower->touch_count, true);
                }
            }
        }
    }
    
    render_grid_lines(cell_size);
    render_special_locations(env, cell_size);
    render_thunderclap_effects(env, cell_size);
    render_runner(env, cell_size);
    render_ui_panel(env);
    
    EndDrawing();
}

void c_close(MazingContest* env) {
    if (env->client) {
        c_close_client(env->client);
        env->client = NULL;
    }
}

void init(MazingContest *env) {
    env->log = (Log){0};
    env->tick = 0;
    env->build_time_limit = 600;
    env->max_moves = 3000000;
    env->max_rounds = 1;
    env->min_gold = 10;
    env->max_gold = 30;
    env->min_lumber = 2;
    env->max_lumber = 4;
    env->gold = 50;
    env->lumber = 1;
    env->phase = PHASE_BUILD;
    env->phase_timer = 0;
    env->round_number = 1;
    env->runner_time = 0.0f;
    env->best_time = 0.0f;
    env->maze_completed = 0;
    env->grid_cell_size = 32.0f;
    env->path_length_reward_multiplier = 1.4f;
    env->wall_touch_base_reward = 0.1f;
    env->thunderclap_slowdown_factor = 0.3f;
    env->thunderclap_slowdown_duration = 60;
    env->thunderclap_range = 50.0f;
    env->thunderclap_cooldown = 60;
    env->runner_base_speed = 6.0f;
    env->thunderclap_slowdown_reward = 0.06f;
    env->stuck_teleport_ticks = 120;
    env->fps = 60.0f;
    env->gold_normalization = 50.0f;
    env->lumber_normalization = 5.0f;
    env->phase_normalization = 2.0f;
    env->build_time_normalization = 600.0f;
    env->runner_time_normalization = 30.0f;
    env->cell_size_render = 50.0f;
    env->min_random_walls = 8;
    env->max_random_walls = 20;
}