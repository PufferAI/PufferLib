#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// Simulation constants
#define TRAJECTORY_LENGTH 91 // Discretized Waymo scenarios
#define SIM_DT 0.1f

// Agent limits
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif

// Dynamics models
#define CLASSIC 0

// Collision State
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Grid Map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation Space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
#define EGO_FEATURES 7
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)

// Observation normalization
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Observation scaling factors
#define OBS_GOAL_SCALE 0.005f
#define OBS_SPEED_SCALE 0.01f
#define OBS_POSITION_SCALE 0.02f

// Distance thresholds
#define MIN_DISTANCE_TO_GOAL 5.0f
#define COLLISION_DIST_SQ 225.0f
#define OBS_DIST_SQ 2500.0f
#define COLLISION_BOX_SCALE 0.7f

// Action space
#define NUM_ACCEL_BINS 7
#define NUM_STEER_BINS 13

static const float ACCELERATION_VALUES[NUM_ACCEL_BINS] = {
    -4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f
};

static const float STEERING_VALUES[NUM_STEER_BINS] = {
    -1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
     0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f
};

// Geometry helpers
static const float offsets[4][2] = {
    {-1,  1},  // top-left
    { 1,  1},  // top-right
    { 1, -1},  // bottom-right
    {-1, -1}   // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},
    {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}
};

// Rendering Colors
const Color STONE_GRAY       = (Color){80, 80, 80, 255};
const Color PUFF_RED         = (Color){187, 0, 0, 255};
const Color PUFF_CYAN        = (Color){0, 187, 187, 255};
const Color PUFF_WHITE       = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND  = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color ROAD_COLOR       = (Color){35, 35, 37, 255};

// Forward declarations
typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Entity Entity;

struct Log {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float offroad_rate;
    float collision_rate;
    float clean_collision_rate;
    float completion_rate;
    float dnf_rate;
    float n;
};

struct Entity {
    int type;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* traj_heading;
    int* traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int mark_as_expert;
    int collision_state;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int valid;
    int reached_goal;
    int respawn_timestep;
    int collided_before_goal;
    int reached_goal_this_episode;
    int active_agent;
};

void free_entity(Entity* entity) {
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

// Utility
float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

float clipSpeed(float speed) {
    if (speed > MAX_SPEED) return MAX_SPEED;
    if (speed < -MAX_SPEED) return -MAX_SPEED;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI) heading -= 2 * M_PI;
    if (heading < -M_PI) heading += 2 * M_PI;
    return heading;
}

struct Drive {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Log log;
    Log* logs;
    int num_agents;
    int max_agents;
    int active_agent_count;
    int* active_agent_indices;
    int human_agent_idx;
    Entity* entities;
    int num_entities;
    int num_actors;  // Total agents (active + static)
    int num_objects;
    int num_roads;
    int static_agent_count;
    int* static_agent_indices;
    int expert_static_agent_count;
    int* expert_static_agent_indices;
    int timestep;
    int dynamics_model;
    float* map_corners;
    int* grid_cells;
    int grid_cols;
    int grid_rows;
    int vision_range;
    int* neighbor_offsets;
    int* neighbor_cache_entities;
    int* neighbor_cache_indices;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char* map_name;
    float world_mean_x;
    float world_mean_y;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    unsigned int rng;
};

void add_log(Drive* env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* e = &env->entities[env->active_agent_indices[i]];
        if (e->reached_goal_this_episode) {
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int clean_collided = env->logs[i].clean_collision_rate;
        env->log.clean_collision_rate += clean_collided;
        if (e->reached_goal_this_episode && !e->collided_before_goal) {
            env->log.score += 1.0f;
            env->log.perf += 1.0f;
        }
        if (!offroad && !collided && !e->reached_goal_this_episode) {
            env->log.dnf_rate += 1.0f;
        }
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

// Map loading
Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;

    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float*)malloc(size * sizeof(float));
        entities[i].traj_y = (float*)malloc(size * sizeof(float));
        entities[i].traj_z = (float*)malloc(size * sizeof(float));

        int is_actor = (entities[i].type == VEHICLE ||
                        entities[i].type == PEDESTRIAN ||
                        entities[i].type == CYCLIST);
        if (is_actor) {
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float*)malloc(size * sizeof(float));
            entities[i].traj_vy = (float*)malloc(size * sizeof(float));
            entities[i].traj_vz = (float*)malloc(size * sizeof(float));
            entities[i].traj_heading = (float*)malloc(size * sizeof(float));
            entities[i].traj_valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (is_actor) {
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
    }
    fclose(file);
    return entities;
}

// Position initialization
void set_start_position(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity* e = &env->entities[i];
        e->x = e->traj_x[0];
        e->y = e->traj_y[0];
        e->z = e->traj_z[0];

        if (e->type > CYCLIST || e->type == NONE) {
            continue;
        }
        if (!is_active) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->reached_goal = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[0];
            e->vy = e->traj_vy[0];
            e->vz = e->traj_vz[0];
        }
        e->heading = e->traj_heading[0];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[0];
        e->collision_state = NO_COLLISION;
        e->respawn_timestep = -1;
    }
}

// Grid Map
int getGridIndex(Drive* env, float x1, float y1) {
    if (env->map_corners[0] >= env->map_corners[2] ||
        env->map_corners[1] >= env->map_corners[3]) {
        return -1;
    }
    float relativeX = x1 - env->map_corners[0];
    float relativeY = y1 - env->map_corners[1];
    int gridX = (int)(relativeX / GRID_CELL_SIZE);
    int gridY = (int)(relativeY / GRID_CELL_SIZE);
    if (gridX < 0 || gridX >= env->grid_cols || gridY < 0 || gridY >= env->grid_rows) {
        return -1;
    }
    return (gridY * env->grid_cols) + gridX;
}

void add_entity_to_grid(Drive* env, int grid_index, int entity_idx, int geometry_idx) {
    if (grid_index == -1) return;

    int base_index = grid_index * SLOTS_PER_CELL;
    int count = env->grid_cells[base_index];
    if (count >= MAX_ENTITIES_PER_CELL) return;

    env->grid_cells[base_index + count * 2 + 1] = entity_idx;
    env->grid_cells[base_index + count * 2 + 2] = geometry_idx;
    env->grid_cells[base_index] = count + 1;
}

void init_grid_map(Drive* env) {
    float top_left_x, top_left_y, bottom_right_x, bottom_right_y;
    int first_valid_point = 0;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type >= ROAD_LANE && env->entities[i].type <= ROAD_EDGE) {
            Entity* e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION) continue;
                if (e->traj_y[j] == INVALID_POSITION) continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = 1;
                    continue;
                }
                if (e->traj_x[j] < top_left_x) top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x) bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] < top_left_y) top_left_y = e->traj_y[j];
                if (e->traj_y[j] > bottom_right_y) bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->map_corners = (float*)calloc(4, sizeof(float));
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;

    float grid_width = bottom_right_x - top_left_x;
    float grid_height = bottom_right_y - top_left_y;
    env->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_cols * env->grid_rows;
    env->grid_cells = (int*)calloc(grid_cell_count * SLOTS_PER_CELL, sizeof(int));

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type >= ROAD_LANE && env->entities[i].type <= ROAD_EDGE) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j);
            }
        }
    }
}

void init_neighbor_offsets(Drive* env) {
    int vr = env->vision_range;
    env->neighbor_offsets = (int*)calloc(vr * vr * 2, sizeof(int));

    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0;
    int steps_to_take = 1, steps_taken = 0, segments_completed = 0;
    int total = 0, max_offsets = vr * vr;
    int curr_idx = 0;

    env->neighbor_offsets[curr_idx++] = 0;
    env->neighbor_offsets[curr_idx++] = 0;
    total++;

    while (total < max_offsets) {
        x += dx[dir];
        y += dy[dir];
        if (abs(x) <= vr / 2 && abs(y) <= vr / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        if (steps_taken != steps_to_take) continue;
        steps_taken = 0;
        dir = (dir + 1) % 4;
        segments_completed++;
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive* env) {
    int vr = env->vision_range;
    int count = 0;
    int cell_count = env->grid_cols * env->grid_rows;

    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        env->neighbor_cache_indices[i] = count;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
            int grid_index = env->grid_cols * y + x;
            count += env->grid_cells[grid_index * SLOTS_PER_CELL] * 2;
        }
    }
    env->neighbor_cache_indices[cell_count] = count;
    env->neighbor_cache_entities = (int*)calloc(count, sizeof(int));

    for (int i = 0; i < cell_count; i++) {
        int neighbor_cache_base_index = 0;
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
            int grid_index = env->grid_cols * y + x;
            int grid_count = env->grid_cells[grid_index * SLOTS_PER_CELL];
            int base_index = env->neighbor_cache_indices[i];
            int src_idx = grid_index * SLOTS_PER_CELL + 1;
            int dst_idx = base_index + neighbor_cache_base_index;
            memcpy(&env->neighbor_cache_entities[dst_idx],
                   &env->grid_cells[src_idx],
                   grid_count * 2 * sizeof(int));
            neighbor_cache_base_index += grid_count * 2;
        }
    }
}

int get_neighbor_cache_entities(Drive* env, int cell_idx, int* entities, int max_entities) {
    if (cell_idx < 0 || cell_idx >= (env->grid_cols * env->grid_rows)) {
        return 0;
    }
    int base_index = env->neighbor_cache_indices[cell_idx];
    int end_index = env->neighbor_cache_indices[cell_idx + 1];
    int count = end_index - base_index;
    int pairs = count / 2;
    if (pairs > max_entities) {
        pairs = max_entities;
        count = pairs * 2;
    }
    memcpy(entities, env->neighbor_cache_entities + base_index, count * sizeof(int));
    return pairs;
}

// Mean Centering
void set_means(Drive* env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_valid[j]) {
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= ROAD_LANE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type >= ROAD_LANE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION) continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

// Expert Movement
void move_expert(Drive* env, float* actions, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

// Collision Detection
bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];
    float cross = dx1 * dy2 - dy1 * dx2;
    if (cross == 0) return false;

    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive* env, float x, float y, int* entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    int index = getGridIndex(env, x, y);
    if (index == -1) return 0;
    int cellsX = env->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_cols || ny < 0 || ny >= env->grid_rows) continue;
        int neighborIndex = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
        int count = env->grid_cells[neighborIndex];
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count] = env->grid_cells[neighborIndex + 1 + j * 2];
            entity_list[entity_list_count + 1] = env->grid_cells[neighborIndex + 2 + j * 2];
            entity_list_count += 2;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity* car1, Entity* car2) {
    float cos1 = car1->heading_x, sin1 = car1->heading_y;
    float cos2 = car2->heading_x, sin2 = car2->heading_y;
    float hl1 = car1->length * 0.5f, hw1 = car1->width * 0.5f;
    float hl2 = car2->length * 0.5f, hw2 = car2->width * 0.5f;

    float car1_corners[4][2] = {
        {car1->x + (hl1 * cos1 - hw1 * sin1), car1->y + (hl1 * sin1 + hw1 * cos1)},
        {car1->x + (hl1 * cos1 + hw1 * sin1), car1->y + (hl1 * sin1 - hw1 * cos1)},
        {car1->x + (-hl1 * cos1 - hw1 * sin1), car1->y + (-hl1 * sin1 + hw1 * cos1)},
        {car1->x + (-hl1 * cos1 + hw1 * sin1), car1->y + (-hl1 * sin1 - hw1 * cos1)}
    };

    float car2_corners[4][2] = {
        {car2->x + (hl2 * cos2 - hw2 * sin2), car2->y + (hl2 * sin2 + hw2 * cos2)},
        {car2->x + (hl2 * cos2 + hw2 * sin2), car2->y + (hl2 * sin2 - hw2 * cos2)},
        {car2->x + (-hl2 * cos2 - hw2 * sin2), car2->y + (-hl2 * sin2 + hw2 * cos2)},
        {car2->x + (-hl2 * cos2 + hw2 * sin2), car2->y + (-hl2 * sin2 - hw2 * cos2)}
    };

    float axes[4][2] = {
        {cos1, sin1}, {-sin1, cos1},
        {cos2, sin2}, {-sin2, cos2}
    };

    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;
        for (int j = 0; j < 4; j++) {
            float proj1 = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj1);
            max1 = fmaxf(max1, proj1);
            float proj2 = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj2);
            max2 = fmaxf(max2, proj2);
        }
        if (max1 < min2 || min1 > max2) return 0;
    }
    return 1;
}

int collision_check(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    if (agent->x == INVALID_POSITION) return -1;

    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] = agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    int collided = NO_COLLISION;
    int car_collided_with_index = -1;

    // Check road edge collisions via grid
    int entity_list[MAX_ENTITIES_PER_CELL * 2 * 25];
    int list_size = checkNeighbors(env, agent->x, agent->y, entity_list,
                                   MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i += 2) {
        if (entity_list[i] == -1 || entity_list[i] == agent_idx) continue;
        Entity* entity = &env->entities[entity_list[i]];
        if (entity->type != ROAD_EDGE) continue;
        int geometry_idx = entity_list[i + 1];
        float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
        float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) break;
    }

    // Check vehicle-vehicle collisions
    for (int i = 0; i < MAX_AGENTS; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1 || index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        float dx = entity->x - agent->x;
        float dy = entity->y - agent->y;
        if ((dx * dx + dy * dy) > COLLISION_DIST_SQ) continue;
        if (check_aabb_collision(agent, entity)) {
            collided = VEHICLE_COLLISION;
            car_collided_with_index = index;
            break;
        }
    }

    agent->collision_state = collided;

    // Spawn immunity: agent just respawned
    if (collided == VEHICLE_COLLISION && agent->active_agent == 1 &&
        agent->respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    if (collided == OFFROAD) return -1;
    if (car_collided_with_index == -1) return -1;

    // Spawn immunity: collided-with agent just respawned
    if (env->entities[car_collided_with_index].respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    return car_collided_with_index;
}

// Agent Selection
int valid_active_agent(Drive* env, int agent_idx) {
    if (agent_idx < 0 || agent_idx >= env->num_entities) return 0;
    Entity* e = &env->entities[agent_idx];
    if (e->type != VEHICLE || e->traj_valid[0] != 1) return 0;
    float cos_heading = cosf(e->traj_heading[0]);
    float sin_heading = sinf(e->traj_heading[0]);
    float goal_x = e->goal_position_x - e->traj_x[0];
    float goal_y = e->goal_position_y - e->traj_y[0];
    float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
    float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);

    e->width *= COLLISION_BOX_SCALE;
    e->length *= COLLISION_BOX_SCALE;

    if (distance_to_goal >= MIN_DISTANCE_TO_GOAL &&
        e->mark_as_expert == 0 &&
        env->active_agent_count < env->max_agents) {
        return distance_to_goal;
    }
    return 0;
}

void set_active_agents(Drive* env) {
    env->active_agent_count = 0;
    env->static_agent_count = 0;
    env->num_actors = 1;
    env->expert_static_agent_count = 0;

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    if (env->max_agents == 0) {
        env->max_agents = MAX_AGENTS;
    }

    // First agent: last object (SDC equivalent)
    int first_agent_id = env->num_objects - 1;
    float distance_to_goal = valid_active_agent(env, first_agent_id);
    if (distance_to_goal) {
        env->active_agent_count = 1;
        active_agent_indices[0] = first_agent_id;
        env->entities[first_agent_id].active_agent = 1;
        env->num_actors = 1;
    } else {
        env->active_agent_count = 0;
        env->num_actors = 0;
    }

    for (int i = 0; i < env->num_objects - 1 && env->num_actors < MAX_AGENTS; i++) {
        if (env->entities[i].type != VEHICLE) continue;
        if (env->entities[i].traj_valid[0] != 1) continue;
        env->num_actors++;

        float dist = valid_active_agent(env, i);
        if (dist > 0) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->entities[i].active_agent = 0;
            if (env->entities[i].mark_as_expert == 1 ||
                (dist >= MIN_DISTANCE_TO_GOAL && env->active_agent_count == env->max_agents)) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int*)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int*)malloc(env->expert_static_agent_count * sizeof(int));
    memcpy(env->active_agent_indices, active_agent_indices, env->active_agent_count * sizeof(int));
    memcpy(env->static_agent_indices, static_agent_indices, env->static_agent_count * sizeof(int));
    memcpy(env->expert_static_agent_indices, expert_static_agent_indices, env->expert_static_agent_count * sizeof(int));
}

// Trajectory Validation
void remove_bad_trajectories(Drive* env) {
    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));

    for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            move_expert(env, env->actions, env->active_agent_indices[i]);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION) continue;
            move_expert(env, env->actions, expert_idx);
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = NO_COLLISION;
            int collided_with = collision_check(env, agent_idx);
            if (env->entities[agent_idx].collision_state > NO_COLLISION && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1) continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_idx = env->static_agent_indices[j];
            if (static_idx != collided_with_indices[i]) continue;
            env->entities[static_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

// Initialization / Cleanup
void init(Drive* env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary(env->map_name, env);
    env->dynamics_model = CLASSIC;
    set_means(env);
    init_grid_map(env);
    env->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    env->neighbor_cache_indices = (int*)calloc((env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_active_agents(env);
    remove_bad_trajectories(env);
    set_start_position(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
}

void c_close(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    free(env->map_corners);
    free(env->grid_cells);
    free(env->neighbor_offsets);
    free(env->neighbor_cache_entities);
    free(env->neighbor_cache_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
}

void allocate(Drive* env) {
    init(env);
    env->observations = (float*)calloc(env->active_agent_count * OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float*)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (float*)calloc(env->active_agent_count, sizeof(float));
}

void free_allocated(Drive* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

// Dynamics
void move_dynamics(Drive* env, int action_idx, int agent_idx) {
    if (env->dynamics_model != CLASSIC) return;

    Entity* agent = &env->entities[agent_idx];
    float (*action_array)[2] = (float(*)[2])env->actions;
    int acceleration_index = action_array[action_idx][0];
    int steering_index = action_array[action_idx][1];
    float acceleration = ACCELERATION_VALUES[acceleration_index];
    float steering = STEERING_VALUES[steering_index];

    float x = agent->x;
    float y = agent->y;
    float heading = agent->heading;
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);

    speed = speed + 0.5f * acceleration * SIM_DT;
    speed = clipSpeed(speed);

    float beta = tanh(0.5 * tanf(steering));
    float yaw_rate = (speed * cosf(beta) * tanf(steering)) / agent->length;
    float new_vx = speed * cosf(heading + beta);
    float new_vy = speed * sinf(heading + beta);

    x += new_vx * SIM_DT;
    y += new_vy * SIM_DT;
    heading += yaw_rate * SIM_DT;

    agent->x = x;
    agent->y = y;
    agent->heading = heading;
    agent->heading_x = cosf(heading);
    agent->heading_y = sinf(heading);
    agent->vx = new_vx;
    agent->vy = new_vy;
}

// Observations
void compute_observations(Drive* env) {
    memset(env->observations, 0, OBS_SIZE * env->active_agent_count * sizeof(float));
    float (*observations)[OBS_SIZE] = (float(*)[OBS_SIZE])env->observations;

    for (int i = 0; i < env->active_agent_count; i++) {
        float* obs = &observations[i][0];
        Entity* ego = &env->entities[env->active_agent_indices[i]];
        if (ego->type > CYCLIST) break;

        if (ego->respawn_timestep != -1) {
            obs[6] = 1;
        }

        float cos_h = ego->heading_x;
        float sin_h = ego->heading_y;
        float ego_speed = sqrtf(ego->vx * ego->vx + ego->vy * ego->vy);

        // Goal in ego frame
        float goal_x = ego->goal_position_x - ego->x;
        float goal_y = ego->goal_position_y - ego->y;
        float rel_goal_x = goal_x * cos_h + goal_y * sin_h;
        float rel_goal_y = -goal_x * sin_h + goal_y * cos_h;

        // Ego features
        obs[0] = rel_goal_x * OBS_GOAL_SCALE;
        obs[1] = rel_goal_y * OBS_GOAL_SCALE;
        obs[2] = ego_speed * OBS_SPEED_SCALE;
        obs[3] = ego->width / MAX_VEH_WIDTH;
        obs[4] = ego->length / MAX_VEH_LEN;
        obs[5] = (ego->collision_state > NO_COLLISION) ? 1 : 0;

        // Partner observations
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1) continue;
            if (env->entities[index].type > CYCLIST) break;
            if (index == env->active_agent_indices[i]) continue;

            Entity* other = &env->entities[index];
            if (ego->respawn_timestep != -1) continue;
            if (other->respawn_timestep != -1) continue;

            float dx = other->x - ego->x;
            float dy = other->y - ego->y;
            if ((dx * dx + dy * dy) > OBS_DIST_SQ) continue;

            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;

            obs[obs_idx + 0] = rel_x * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = rel_y * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = other->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other->length / MAX_VEH_LEN;
            obs[obs_idx + 4] = other->heading_x * ego->heading_x + other->heading_y * ego->heading_y;
            obs[obs_idx + 5] = other->heading_y * ego->heading_x - other->heading_x * ego->heading_y;
            float other_speed = sqrtf(other->vx * other->vx + other->vy * other->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // Road observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS * 2];
        int grid_idx = getGridIndex(env, ego->x, ego->y);
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k * 2];
            int geometry_idx = entity_list[k * 2 + 1];
            Entity* entity = &env->entities[entity_idx];

            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego->x;
            float rel_y = mid_y - ego->y;
            float x_obs = rel_x * cos_h + rel_y * sin_h;
            float y_obs = -rel_x * sin_h + rel_y * cos_h;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);

            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float hypot = sqrtf(dx * dx + dy * dy);
            float dx_norm = dx, dy_norm = dy;
            if (hypot > 0) { dx_norm /= hypot; dy_norm /= hypot; }

            float cos_angle = dx_norm * cos_h + dy_norm * sin_h;
            float sin_angle = -dx_norm * sin_h + dy_norm * cos_h;

            obs[obs_idx + 0] = x_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = y_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = 0.1f / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - (float)ROAD_LANE;
            obs_idx += ROAD_FEATURES;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * ROAD_FEATURES;
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void c_reset(Drive* env) {
    env->timestep = 0;
    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].reached_goal = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;

        collision_check(env, agent_idx);
    }
    compute_observations(env);
}

void respawn_agent(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    e->x = e->traj_x[0];
    e->y = e->traj_y[0];
    e->heading = e->traj_heading[0];
    e->heading_x = cosf(e->heading);
    e->heading_y = sinf(e->heading);
    e->vx = e->traj_vx[0];
    e->vy = e->traj_vy[0];
    e->reached_goal = 0;
    e->respawn_timestep = env->timestep;
}

void c_step(Drive* env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(float));
    env->timestep++;

    if (env->timestep == TRAJECTORY_LENGTH) {
        add_log(env);
        c_reset(env);
        return;
    }

    // Move expert static agents
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->entities[expert_idx].x == INVALID_POSITION) continue;
        move_expert(env, env->actions, expert_idx);
    }

    // Apply dynamics for active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = NO_COLLISION;
        move_dynamics(env, i, agent_idx);
    }

    // Collision detection and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = NO_COLLISION;
        collision_check(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        if (collision_state > NO_COLLISION) {
            if (collision_state == VEHICLE_COLLISION && env->entities[agent_idx].respawn_timestep == -1) {
                env->rewards[i] = env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].clean_collision_rate = 1.0f;
                env->logs[i].collision_rate = 1.0f;
            } else if (collision_state == OFFROAD) {
                env->rewards[i] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
            }
            if (!env->entities[agent_idx].reached_goal_this_episode) {
                env->entities[agent_idx].collided_before_goal = 1;
            }
        }

        float distance_to_goal = relative_distance_2d(
            env->entities[agent_idx].x, env->entities[agent_idx].y,
            env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        if (distance_to_goal < MIN_DISTANCE_TO_GOAL) {
            if (env->entities[agent_idx].respawn_timestep != -1) {
                env->rewards[i] += env->reward_goal_post_respawn;
                env->logs[i].episode_return += env->reward_goal_post_respawn;
            } else {
                env->rewards[i] += 1.0f;
                env->logs[i].episode_return += 1.0f;
            }
            env->entities[agent_idx].reached_goal = 1;
            env->entities[agent_idx].reached_goal_this_episode = 1;
        }
    }

    // Respawn agents that reached goal
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->entities[agent_idx].reached_goal) {
            respawn_agent(env, agent_idx);
        }
    }

    compute_observations(env);
}

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client* make_client(Drive* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand_r(&env->rng) % 4) + 1;
    }
    // Get initial target position from first active agent
    float map_center_x = (env->map_corners[0] + env->map_corners[2]) / 2.0f;
    float map_center_y = (env->map_corners[1] + env->map_corners[3]) / 2.0f;
    Vector3 target_pos = {
       0,
        0,  // Y is up
        1   // Z is depth
    };
    
    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){ 
        0,           // Same X as target
        120.0f,   // 20 units above target
        175.0f    // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, -1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;
    
    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }
    
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }
    
    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;
        
        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };
        
        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive* env, int agent_index) {
    float diamond_height = 3.0f;
    float diamond_width = 1.5f;
    float diamond_z = 8.0f;

    Vector3 top = {0, 0, diamond_z + diamond_height / 2};
    Vector3 bot = {0, 0, diamond_z - diamond_height / 2};
    Vector3 fwd = {0, diamond_width / 2, diamond_z};
    Vector3 bck = {0, -diamond_width / 2, diamond_z};
    Vector3 lft = {-diamond_width / 2, 0, diamond_z};
    Vector3 rgt = {diamond_width / 2, 0, diamond_z};

    DrawTriangle3D(top, fwd, rgt, PUFF_CYAN);
    DrawTriangle3D(top, rgt, bck, PUFF_CYAN);
    DrawTriangle3D(top, bck, lft, PUFF_CYAN);
    DrawTriangle3D(top, lft, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, rgt, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, bck, rgt, PUFF_CYAN);
    DrawTriangle3D(bot, lft, bck, PUFF_CYAN);
    DrawTriangle3D(bot, fwd, lft, PUFF_CYAN);

    if (!IsKeyDown(KEY_LEFT_CONTROL)) return;

    float (*observations)[OBS_SIZE] = (float(*)[OBS_SIZE])env->observations;
    float* agent_obs = &observations[agent_index][0];

    // Draw goal
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, GREEN);

    // Draw partner observations
    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }
        float x = agent_obs[obs_idx] / OBS_POSITION_SCALE;
        float y = agent_obs[obs_idx + 1] / OBS_POSITION_SCALE;
        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);

        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float angle = atan2f(theta_y, theta_x);
        // Draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;

        float ax = x + arrow_length * cosf(angle);
        float ay = y + arrow_length * sinf(angle);
        DrawLine3D((Vector3){x, y, 1}, (Vector3){ax, ay, 1}, PUFF_WHITE);

        float arrow_size = 2.0f;
        float dx = ax - x, dy = ay - y;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 0) {
            dx /= len; dy /= len;
            float px = -dy * arrow_size, py = dx * arrow_size;
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size + px, ay - dy * arrow_size + py, 1}, PUFF_WHITE);
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size - px, ay - dy * arrow_size - py, 1}, PUFF_WHITE);
        }
        obs_idx += PARTNER_FEATURES;
    }

    // Draw road edge observations
    int map_start_idx = EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1);
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
        int idx = map_start_idx + k * ROAD_FEATURES;
        if (agent_obs[idx] == 0 && agent_obs[idx + 1] == 0) continue;
        int entity_type = (int)agent_obs[idx + 6];
        if (entity_type + ROAD_LANE != ROAD_EDGE) continue;

        float x_mid = agent_obs[idx] / OBS_POSITION_SCALE;
        float y_mid = agent_obs[idx + 1] / OBS_POSITION_SCALE;
        float rel_angle = atan2f(agent_obs[idx + 5], agent_obs[idx + 4]);
        float seg_len = agent_obs[idx + 2] * MAX_ROAD_SEGMENT_LENGTH;

        float x_start = x_mid - seg_len * cosf(rel_angle);
        float y_start = y_mid - seg_len * sinf(rel_angle);
        float x_end = x_mid + seg_len * cosf(rel_angle);
        float y_end = y_mid + seg_len * sinf(rel_angle);

        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_mid, y_mid, 1}, PUFF_CYAN);
        DrawCube((Vector3){x_mid, y_mid, 1}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};
    Color CURB_SIDE = (Color){180, 180, 180, 255};
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    float curb_height = 0.5f;
    float curb_width = 0.3f;

    Vector3 direction = {end_x - start_x, end_y - start_y, 0};
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
    Vector3 nd = {direction.x / length, direction.y / length, 0};
    Vector3 perp = {-nd.y, nd.x, 0};

    Vector3 b1 = {start_x - perp.x * curb_width / 2, start_y - perp.y * curb_width / 2, 1.0f};
    Vector3 b2 = {start_x + perp.x * curb_width / 2, start_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b3 = {end_x + perp.x * curb_width / 2, end_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b4 = {end_x - perp.x * curb_width / 2, end_y - perp.y * curb_width / 2, 1.0f};

    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    DrawTriangle3D(b1, t1, b2, CURB_SIDE); DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE); DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE); DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE); DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void c_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    BeginDrawing();
    ClearBackground(ROAD_COLOR);
    BeginMode3D(client->camera);
    handle_camera_controls(client);

    // Map bounds
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);

    for (int i = 0; i < env->num_entities; i++) {
        // Draw vehicles
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN) {
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if ((!is_active_agent && !is_static_agent) || env->entities[i].respawn_timestep != -1) {
                continue;
            }

            Vector3 position = {env->entities[i].x, env->entities[i].y, 1};
            float heading = env->entities[i].heading;
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);

            Model car_model = client->cars[5];
            if (is_active_agent) {
                car_model = client->cars[client->car_assignments[i % MAX_AGENTS]];
            }
            if (agent_index == env->human_agent_idx) {
                // Human-controlled agent uses default model
            }
            if (is_active_agent && env->entities[i].collision_state > NO_COLLISION) {
                car_model = client->cars[0];
            }

            if (agent_index == env->human_agent_idx && !env->entities[agent_index].reached_goal) {
                draw_agent_obs(env, agent_index);
            }

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size = {
                bounds.max.x - bounds.min.x,
                bounds.max.y - bounds.min.y,
                bounds.max.z - bounds.min.z
            };
            Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
            rlPopMatrix();

            // Draw collision box
            float cos_h = env->entities[i].heading_x;
            float sin_h = env->entities[i].heading_y;
            float hl = env->entities[i].length * 0.5f;
            float hw = env->entities[i].width * 0.5f;
            Vector3 corners[4] = {
                {position.x + (hl * cos_h - hw * sin_h), position.y + (hl * sin_h + hw * cos_h), position.z},
                {position.x + (hl * cos_h + hw * sin_h), position.y + (hl * sin_h - hw * cos_h), position.z},
                {position.x + (-hl * cos_h - hw * sin_h), position.y + (-hl * sin_h + hw * cos_h), position.z},
                {position.x + (-hl * cos_h + hw * sin_h), position.y + (-hl * sin_h - hw * cos_h), position.z}
            };
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], PURPLE);
            }

            // FPV camera
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                if (env->entities[agent_index].reached_goal) {
                    env->human_agent_idx = rand_r(&env->rng) % env->active_agent_count;
                }
                client->camera.position = (Vector3){
                    position.x - 25.0f * cosf(heading),
                    position.y - 25.0f * sinf(heading),
                    position.z + 15
                };
                client->camera.target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f
                };
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }

            if (!is_active_agent || env->entities[i].valid == 0) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 1}, 0.5f, DARKGREEN);
            }
        }

        // Draw road elements
        if (env->entities[i].type < ROAD_LANE || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            if (env->entities[i].type != ROAD_EDGE) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                draw_road_edge(env,
                    env->entities[i].traj_x[j], env->entities[i].traj_y[j],
                    env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1]);
            }
        }
    }

    // Grid overlay
    float grid_start_x = env->map_corners[0];
    float grid_start_y = env->map_corners[1];
    for (int i = 0; i < env->grid_cols; i++) {
        for (int j = 0; j < env->grid_rows; j++) {
            float x = grid_start_x + i * GRID_CELL_SIZE;
            float y = grid_start_y + j * GRID_CELL_SIZE;
            DrawCubeWires(
                (Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
        }
    }
    EndMode3D();

    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x, client->camera.position.y, client->camera.position.z), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x, client->camera.target.y, client->camera.target.z), 10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
             10, client->height - 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Acceleration: %d", env->actions[env->human_agent_idx * 2]), 10, 110, 20, PUFF_WHITE);
    DrawText(TextFormat("Steering: %d", env->actions[env->human_agent_idx * 2 + 1]), 10, 130, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client* client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}
