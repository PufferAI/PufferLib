#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include "raylib.h"
#include "simplex.h"
#include "raymath.h"
#include "rlgl.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

#define MAX_DIRT_HEIGHT 32.0f
#define BUCKET_MAX_HEIGHT 1.0f
#define DOZER_MAX_V 2.0f
#define DOZER_CAPACITY 100.0f
#define BUCKET_OFFSET 2.0f
#define BUCKET_WIDTH 2.5f
#define BUCKET_LENGTH 0.8f
#define BUCKET_HEIGHT 1.0f
#define SCOOP_SIZE 1
#define VISION 5
#define OBSERVATION_SIZE (2*VISION + 1)
#define TOTAL_OBS (OBSERVATION_SIZE*OBSERVATION_SIZE + 4)
#define DOZER_STEP_HEIGHT 5.0f 
struct timespec ts;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float quadrant_progress;
};

typedef struct Dozer {
    float x;
    float y;
    float z;
    float v;
    float heading;
    float bucket_height;
    float bucket_tilt;
    float load;
    int target_quadrant;
    int* load_indices;
    float quadrant_progress;
    float highest_quadrant_progress;
    float target_quadrant_delta;
} Dozer;
 
typedef struct Client Client;
typedef struct Terraform {
    Log log;
    Log* agent_logs;
    Client* client;
    Dozer* dozers;
    float* observations;
    int* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals;
    int size;
    int tick;
    float* orig_map;
    float* map;
    float* target_map;
    int num_agents;
    int reset_frequency;
    float reward_scale;
    float initial_total_delta;  
    float current_total_delta; 
    float delta_progress;
    int* stuck_count;
    int* grid_indices;
    int num_quadrants;
    float* quadrant_deltas;
    float* current_quadrant_deltas;
    float quadrants_solved;
    int* complete_quadrants;
    int* in_progress_quadrants;
    float* volume_deltas;
    float* quadrant_volume_deltas;
    float* quadrant_centroids;
} Terraform;

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y, float glob_scale) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] += (1.0/pow(2, oct))*noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = glob_scale * scale * (map[adr] - min_value);
            if (map[adr] < 34.0f) {
                map[adr] = 0.0f;
            } else {
                map[adr] -= 34.0f;
            }
        }
    }
}

int map_idx(Terraform* env, float x, float y) {
    return env->size*(int)y + (int)x;
}

void calculate_total_delta(Terraform* env) {
    env->initial_total_delta = 0.0f;
    env->current_total_delta = 0.0f;
    // Calculate total volume in original and target maps
    float original_volume = 0.0f;
    float target_volume = 0.0f;
    for (int i = 0; i < env->size * env->size; i++) {
        original_volume += env->orig_map[i];
        target_volume += env->target_map[i];
    }
    
    float scale_factor = target_volume / original_volume;
    for (int i = 0; i < env->size * env->size; i++) {
        if(env->orig_map[i] * scale_factor > MAX_DIRT_HEIGHT) {
            env->orig_map[i] = MAX_DIRT_HEIGHT;
        } else {
            env->orig_map[i] *= scale_factor;
        }
    }

    for (int i = 0; i < env->size * env->size; i++) {
        float delta = fabsf(env->orig_map[i] - env->target_map[i]);
        env->initial_total_delta += delta;
        env->quadrant_deltas[env->grid_indices[i]] += delta;
        env->quadrant_volume_deltas[env->grid_indices[i]] += (env->orig_map[i] - env->target_map[i]);
        env->volume_deltas[env->grid_indices[i]] += (env->orig_map[i] - env->target_map[i]);
    }
    memcpy(env->current_quadrant_deltas, env->quadrant_deltas, env->num_quadrants*sizeof(float));
    env->current_total_delta = env->initial_total_delta;
    env->delta_progress = 0.0f;
}

void assign_grid_indices(Terraform* env) {
    int num_quads_x = (env->size + 10) / 11;
    int num_quads_y = (env->size + 10) / 11;
    for (int i = 0; i < env->size*env->size; i++) {
        int y = i / env->size;
        int x = i % env->size;
        int quad_x = x / 11;
        int quad_y = y / 11;
        int grid_index = quad_y * num_quads_x + quad_x;
        env->grid_indices[i] = grid_index;
    }
    env->num_quadrants = num_quads_x * num_quads_y;
}

void assign_quadrant_centroids(Terraform* env) {
    int num_quads_x = (env->size + 10) / 11;
    int num_quads_y = (env->size + 10) / 11;
    for (int i = 0; i < env->num_quadrants; i++) {
        float centroid_x = (i % num_quads_x) * 11 + 5;
        float centroid_y = (i / num_quads_x) * 11 + 5;
        env->quadrant_centroids[i*2] = centroid_x;
        env->quadrant_centroids[i*2 + 1] = centroid_y;
    }
}

void init(Terraform* env) {
    env->orig_map = calloc(env->size*env->size, sizeof(float));
    env->map = calloc(env->size*env->size, sizeof(float));
    env->target_map = calloc(env->size*env->size, sizeof(float));
    env->grid_indices = calloc(env->size*env->size, sizeof(int));
    assign_grid_indices(env);
    env->quadrant_centroids = calloc(env->num_quadrants*2, sizeof(float));
    assign_quadrant_centroids(env);
    env->quadrant_deltas = calloc(env->num_quadrants, sizeof(float));
    env->complete_quadrants = calloc(env->num_quadrants, sizeof(int));
    env->in_progress_quadrants = calloc(env->num_quadrants*(env->num_agents+1), sizeof(int));
    env->current_quadrant_deltas = calloc(env->num_quadrants, sizeof(float));
    env->quadrant_volume_deltas = calloc(env->num_quadrants, sizeof(float));
    env->volume_deltas = calloc(env->num_quadrants, sizeof(float));
    env->agent_logs = calloc(env->num_agents, sizeof(Log));
    // for (int i = 0; i < env->size*env->size; i++) {
    //     env->target_map[i] = 1;  // Initialize all to empty
    // }

    // Calculate grid dimensions for quadrants
    const int QUADRANT_SIZE = 11;
    const int MIN_SPACING = 3;
    const int TOTAL_SPACE = QUADRANT_SIZE + MIN_SPACING;

    // Calculate how many quadrants we can fit in each dimension
    int num_quadrants_x = (env->size - MIN_SPACING) / TOTAL_SPACE;
    int num_quadrants_y = (env->size - MIN_SPACING) / TOTAL_SPACE;

    // Place quadrants in a grid pattern
    for (int grid_y = 0; grid_y < num_quadrants_y; grid_y++) {
        for (int grid_x = 0; grid_x < num_quadrants_x; grid_x++) {
            // Calculate starting position for this quadrant
            int start_x = MIN_SPACING + grid_x * TOTAL_SPACE;
            int start_y = MIN_SPACING + grid_y * TOTAL_SPACE;
            
            // Place the 11x11 quadrant
            for (int y = 0; y < QUADRANT_SIZE; y++) {
                for (int x = 0; x < QUADRANT_SIZE; x++) {
                    int pos_x = start_x + x;
                    int pos_y = start_y + y;
                    if (pos_x < env->size && pos_y < env->size) {
                        env->target_map[pos_y * env->size + pos_x] = 1;  // Mark as target
                    }
                }
            }
        }
    } 
    env->dozers = calloc(env->num_agents, sizeof(Dozer));
    for (int i = 0; i < env->num_agents; i++) {
        env->dozers[i].load_indices = calloc((2*SCOOP_SIZE + 1)*(2*SCOOP_SIZE + 1), sizeof(int));
        for (int j = 0; j < (2*SCOOP_SIZE + 1)*(2*SCOOP_SIZE + 1); j++) {
            env->dozers[i].load_indices[j] = -1;
        }
        env->dozers[i].quadrant_progress = 0.0f;
        env->dozers[i].highest_quadrant_progress = 0.0f;
    }
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned int base_seed = (unsigned int)(ts.tv_nsec ^ ts.tv_sec ^ getpid());
    unsigned int seed1 = base_seed;
    unsigned int seed2 = base_seed + 99991;
    srand(seed1);
    int offset_x1 = rand() % 10000;
    int offset_y1 = rand() % 10000;
    srand(seed2);
    int offset_x2 = rand() % 10000;
    int offset_y2 = rand() % 10000;
    perlin_noise(env->orig_map, env->size, env->size, 1.0/(env->size / 4.0), 8, offset_x1, offset_y1, MAX_DIRT_HEIGHT+20);
    // perlin_noise(env->target_map, env->size, env->size, 1.0/(env->size / 4.0), 8, offset_x2, offset_y2, MAX_DIRT_HEIGHT+55);
    env->returns = calloc(env->num_agents, sizeof(float));
    calculate_total_delta(env);
    env->stuck_count = calloc(env->num_agents, sizeof(int));
    env->tick = rand() % 512;
    env->quadrants_solved = 0.0f;
}

void free_initialized(Terraform* env) {
    free(env->orig_map);
    free(env->map);
    for (int i = 0; i < env->num_agents; i++) {
        free(env->dozers[i].load_indices);
    }
    free(env->dozers);
    free(env->returns);
    free(env->target_map);
    free(env->stuck_count);
    free(env->quadrant_deltas);
    free(env->complete_quadrants);
    free(env->grid_indices);
    free(env->current_quadrant_deltas);
    free(env->in_progress_quadrants);
    free(env->agent_logs);
    free(env->quadrant_volume_deltas);
    free(env->volume_deltas);
    free(env->quadrant_centroids);
}

void add_log(Terraform* env, Log* agent_log) {
    env->log.perf += agent_log->perf;
    env->log.score += agent_log->score;
    env->log.episode_length += agent_log->episode_length;
    env->log.episode_return += agent_log->episode_return;
    env->log.n++;
    env->log.quadrant_progress += agent_log->quadrant_progress;
}

void compute_all_observations(Terraform* env) {
    int dialate = 1;
    int max_obs = 319;
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations; 
    int channel_diff_offset = (2*VISION+1)*(2*VISION+1);
    for (int i = 0; i < env->num_agents; i++) {
        int obs_idx = 0;
        float* obs = &observations[i][obs_idx];
        int x_offset = env->dozers[i].x - dialate*VISION;
        int y_offset = env->dozers[i].y - dialate*VISION;
        for (int y = 0; y < 2 * dialate * VISION + 1; y += dialate) {  // ROW loop (Y-axis)
            for (int x = 0; x < 2 * dialate * VISION + 1; x += dialate) {  // COLUMN loop (X-axis)
                int map_x = x_offset + x;
                int map_y = y_offset + y;

                if (map_x < 0 || map_x >= env->size || map_y < 0 || map_y >= env->size) {
                    obs[obs_idx] = 0;
                    obs[obs_idx + channel_diff_offset] = 0;
                    obs_idx++;
                    continue;
                }
                int map_idx = map_y * env->size + map_x; 
                obs[obs_idx] = ((float)env->map[map_idx]) / MAX_DIRT_HEIGHT;
                float diff = ((float)(env->target_map[map_idx] - env->map[map_idx])) / (MAX_DIRT_HEIGHT * 2.0f);
                obs[obs_idx + channel_diff_offset] = diff;
                obs_idx++;
            }
        }
        obs_idx += channel_diff_offset;
        
        Dozer* dozer = &env->dozers[i];
        obs[obs_idx++] = dozer->x / env->size;
        obs[obs_idx++] = dozer->y / env->size;
        obs[obs_idx++] = (dozer->v) / (DOZER_MAX_V);
        // This is -5?
        obs[obs_idx++] = (dozer->heading) / (2*PI);
        obs[obs_idx++] = dozer->load / DOZER_CAPACITY;
        // float goal_x = env->quadrant_centroids[dozer->target_quadrant*2];
        // float goal_y = env->quadrant_centroids[dozer->target_quadrant*2+1];
        // float rel_x = goal_x - dozer->x;
        // float rel_y = goal_y - dozer->y;
        // float max_dist = sqrtf(2) * env->size;
        // obs[obs_idx++] = rel_x / max_dist;
        // obs[obs_idx++] = rel_y / max_dist;

        // Current and target quadrant - 249
        // obs[obs_idx++] = (float)dozer->target_quadrant / env->num_quadrants;
        // obs[obs_idx++] = (float)env->grid_indices[map_idx(env, dozer->x, dozer->y)] / env->num_quadrants;
        // relative directions to target quadrant center - 251
        for (int q = 0; q < env->num_quadrants; q++) {
            obs[obs_idx++] = env->quadrant_volume_deltas[q] / 121.0f;
        }
        float location_conv[env->num_quadrants];
        memset(location_conv, 0, env->num_quadrants*sizeof(float));
        location_conv[env->grid_indices[map_idx(env, dozer->x, dozer->y)]] = 1.0f;
        memcpy(obs + obs_idx, location_conv, env->num_quadrants*sizeof(float));
        obs_idx += env->num_quadrants;
    }
}

void c_reset(Terraform* env) {
    memcpy(env->map, env->orig_map, env->size*env->size*sizeof(float));
    memset(env->observations, 0, env->num_agents*319*sizeof(float));
    memset(env->returns, 0, env->num_agents*sizeof(float));
    env->tick = 0;
    env->current_total_delta = env->initial_total_delta;
    env->delta_progress = 0.0f;
    env->quadrants_solved = 0.0f;
    memset(env->stuck_count, 0, env->num_agents*sizeof(int));
    memcpy(env->quadrant_volume_deltas, env->volume_deltas, env->num_quadrants*sizeof(float));
    memset(env->complete_quadrants, 0, env->num_quadrants*sizeof(int));

    int num_quadrants_to_precomplete = rand() % 5 + 25; // e.g. 30 to 34
    
    // Create array of available quadrants
    int available[env->num_quadrants];
    int num_available = 0;
    for (int i = 0; i < env->num_quadrants; i++) {
        if (!env->complete_quadrants[i]) {
            available[num_available++] = i;
        }
    }

    // Complete exactly num_quadrants_to_precomplete quadrants
    // for (int i = 0; i < num_quadrants_to_precomplete && num_available > 0; i++) {
    //     // Pick random quadrant from remaining available ones
    //     int idx = rand() % num_available;
    //     int quad = available[idx];
        
    //     // Complete the quadrant
    //     for (int j = 0; j < env->size*env->size; j++) {
    //         if(env->grid_indices[j] == quad) {
    //             env->map[j] = env->target_map[j];
    //         }
    //     }

    //     env->complete_quadrants[quad] = 1;
    //     env->current_quadrant_deltas[quad] = 0.0f;
    //     env->quadrant_volume_deltas[quad] = 0.0f;
    //     env->quadrants_solved++;

    //     // Remove used quadrant by swapping with last available one
    //     available[idx] = available[--num_available];
    // }

    // // adjust remaining volume
    // float remaining_target_sum = 0.0f;
    // float remaining_map_sum = 0.0f;
    // for(int i = 0; i < env->size*env->size; i++) {
    //     int quad = env->grid_indices[i];
    //     if (!env->complete_quadrants[quad]) {
    //         remaining_target_sum += env->target_map[i];
    //         remaining_map_sum += env->map[i];
    //     }
    // }
    // // 2. Compute scale factor
    // float scale = (remaining_map_sum > 0.0f) ? (remaining_target_sum / remaining_map_sum) : 1.0f;
    // for (int i = 0; i < env->size * env->size; i++) {
    //     int quad = env->grid_indices[i];
    //     if (!env->complete_quadrants[quad]) {
    //         env->map[i] *= scale;
    //         if (env->map[i] > MAX_DIRT_HEIGHT) env->map[i] = MAX_DIRT_HEIGHT;
    //     }
    // }
    int available_quadrants[env->num_quadrants - (int)env->quadrants_solved];
    int available_quadrants_count = 0;
    for (int i = 0; i < env->num_quadrants; i++) {
        if (!env->complete_quadrants[i]) {
            available_quadrants[available_quadrants_count++] = i;
        }
    }
    for (int i = 0; i < env->num_agents; i++) {
        Dozer temp = {0};
        env->agent_logs[i] = (Log){0};
        temp.load_indices = env->dozers[i].load_indices;
        env->dozers[i] = temp;
        do {
            env->dozers[i].x = rand() % env->size;
            env->dozers[i].y = rand() % env->size;
        } while (env->map[map_idx(env, env->dozers[i].x, env->dozers[i].y)] != 0.0f);
        for (int j = 0; j < (2*SCOOP_SIZE + 1)*(2*SCOOP_SIZE + 1); j++) {
            env->dozers[i].load_indices[j] = -1;
        }
    }
    compute_all_observations(env);
}

void illegal_action(Terraform* env, int agent_idx) {
    env->rewards[agent_idx] += -0.05f;
    env->returns[agent_idx] += -0.05f;
    env->agent_logs[agent_idx].episode_return += -0.05f;
}

float scoop_dirt(Terraform* env, float x, float y, int bucket_atn, int agent_idx, Dozer* dozer){
    int scoop_idx = map_idx(env, x, y);
    float map_height = env->map[scoop_idx];
    float target_height = env->target_map[scoop_idx];
    float delta_pre = fabsf(map_height - target_height);
    float load_pre = dozer->load;

    if (bucket_atn == 0) {
        return 0.0f;
    } else if (bucket_atn == 1) { // Load
        // Can't load while backing up
        if (dozer->v < 0) {
            illegal_action(env, agent_idx);
            return 0.0f;
        }

        if (dozer->load >= DOZER_CAPACITY) {
            illegal_action(env, agent_idx);
            return 0.0f;
        }
        // if it aint broken dont fix it penalty
        // if (env->complete_quadrants[env->grid_indices[scoop_idx]]) {
        //     env->rewards[agent_idx] += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->returns[agent_idx] += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->agent_logs[agent_idx].episode_return += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->complete_quadrants[env->grid_indices[scoop_idx]] = 0;
        //     env->quadrants_solved -= 1.0f;
        // }

        // Load up to 1 unit of dirt
        float load_amount = 1.0f;
        if (map_height <= 1.0f) {
            load_amount = map_height;
        }                    

        // Don't overload the bucket
        if (dozer->load + load_amount > DOZER_CAPACITY) {
            load_amount = DOZER_CAPACITY - dozer->load;
        }

        dozer->load += load_amount;
        env->map[scoop_idx] -= load_amount;
        map_height -= load_amount;
        env->quadrant_volume_deltas[env->grid_indices[scoop_idx]] -= load_amount;
    } else if (bucket_atn == 2) { // Unload
        // Can't unload while moving forward
        if (dozer->v > 0) {
            illegal_action(env, agent_idx);
            return 0.0f;
        }

        if (dozer->load == 0) {
            illegal_action(env, agent_idx);
            return 0.0f;
        }
        // if it aint broken dont fix it penalty
        // if (env->complete_quadrants[env->grid_indices[scoop_idx]]) {
        //     env->rewards[agent_idx] += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->returns[agent_idx] += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->agent_logs[agent_idx].episode_return += (-1.0f / (SCOOP_SIZE*2 + 1));
        //     env->complete_quadrants[env->grid_indices[scoop_idx]] = 0;
        //     env->quadrants_solved -= 1.0f;
        // }

        float unload_amount = 1.0f;
        if (dozer->load < unload_amount) {
            unload_amount = dozer->load;
        }

        if (map_height + unload_amount > MAX_DIRT_HEIGHT) {
            unload_amount = MAX_DIRT_HEIGHT - map_height;
        }

        dozer->load -= unload_amount;
        env->map[scoop_idx] += unload_amount;
        map_height += unload_amount;
        env->quadrant_volume_deltas[env->grid_indices[scoop_idx]] += unload_amount;
    }

    // Reward for terraforming towards target map
    float delta_post = fabsf(map_height - target_height);
    float load_post = dozer->load;
    env->current_total_delta += (delta_post - delta_pre);
    float normalize_value = (2*SCOOP_SIZE + 1)*(2*SCOOP_SIZE + 1) + 1;
    float reward = (delta_pre + env->reward_scale*load_pre) - (delta_post + env->reward_scale*load_post);
    reward /= normalize_value;
    return reward;
}

void c_step(Terraform* env) {
    env->tick += 1;
    if ((env->reset_frequency && env->tick % env->reset_frequency == 0) || env->current_total_delta < 0.01f) {
        if(env->current_total_delta < 0.01f) {
            for (int i = 0; i < env->num_agents; i++) {
                env->rewards[i] = 1.0f;
                env->returns[i] = 1.0f;
                env->agent_logs[i].episode_return += 1.0f;
            }
        }
        for (int i = 0; i < env->num_agents; i++) {
            env->agent_logs[i].episode_length = env->tick;
            env->agent_logs[i].score = env->delta_progress;
            env->agent_logs[i].perf = env->delta_progress;
            add_log(env, &env->agent_logs[i]);
        }
        c_reset(env);
        return;
    }

    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    int (*actions)[3] = (int(*)[3])env->actions; 
    for (int i = 0; i < env->num_agents; i++) {
        env->agent_logs[i].episode_length = env->tick;
        Dozer* dozer = &env->dozers[i];
        int* atn = actions[i];
        float accel = ((float)atn[0] - 2.0f) / 2.0f; // Discrete(5) -> [-1, 1]
        float steer = ((float)atn[1] - 2.0f) / 10.0f; // Discrete(5) -> [-0.2, 0.2]
        int bucket_atn = atn[2];

        float cx = dozer->x + BUCKET_OFFSET*cosf(dozer->heading);
        float cy = dozer->y + BUCKET_OFFSET*sinf(dozer->heading);
        float total_change = 0.0f;
        int load_idx = 0;
        for (int x = cx - SCOOP_SIZE; x < cx + SCOOP_SIZE; x++) {
            for (int y = cy - SCOOP_SIZE; y < cy + SCOOP_SIZE; y++) {
                if (x < 0 || x >= env->size || y < 0 || y >= env->size) {
                    env->dozers[i].load_indices[load_idx] = -1;
                    load_idx++;
                    continue;
                }
                env->dozers[i].load_indices[load_idx] = map_idx(env, x, y);
                load_idx++;
                total_change += scoop_dirt(env, x, y, bucket_atn, i, dozer);
                
            }
        }
        env->rewards[i] += total_change;
        env->returns[i] += total_change;
        env->agent_logs[i].episode_return += total_change;

        dozer->heading += steer;
        if (dozer->heading > 2*PI) {
            dozer->heading -= 2*PI;
        }
        if (dozer->heading < 0) {
            dozer->heading += 2*PI;
        }

        dozer->v += accel;
        if (dozer->v > DOZER_MAX_V) {
            dozer->v = DOZER_MAX_V;
        }
        if (dozer->v < -DOZER_MAX_V) {
            dozer->v = -DOZER_MAX_V;
        }
        int idx = map_idx(env, dozer->x, dozer->y);
        float dozer_height = env->map[idx];

        // Raytrace collision
        for (int d=0; d<dozer->v; d++) {
            float x = dozer->x + d*cosf(dozer->heading);
            float y = dozer->y + d*sinf(dozer->heading);
            if (x < 0 || x >= env->size-1 || y < 0 || y >= env->size-1) {
                continue;
            }

            int dst_idx = map_idx(env, x, y);
            float dst_height = env->map[dst_idx];
            if (fabsf(dozer_height - dst_height) > DOZER_STEP_HEIGHT) {
                dozer->v = 0;
            }
        }

        // Box collision around final destination
        float dst_x = dozer->x + dozer->v*cosf(dozer->heading);
        float dst_y = dozer->y + dozer->v*sinf(dozer->heading);
        for (int x=(int)(dst_x-1.0f); x<=(int)(dst_x+1.0f); x++) {
            for (int y=(int)(dst_y-1.0f); y<=(int)(dst_y+1.0f); y++) {
                if (x < 0 || x >= env->size-1 || y < 0 || y >= env->size-1) {
                    continue;
                }
                int dst_idx = map_idx(env, x, y);
                float dst_height = env->map[dst_idx];
                if (fabsf(dozer_height - dst_height) > DOZER_STEP_HEIGHT) {
                    dozer->v = 0;
                    env->stuck_count[i]++;
                }
            }
        }

        dozer->x += dozer->v*cosf(dozer->heading);
        dozer->y += dozer->v*sinf(dozer->heading);
        if (dozer->x < 0) {
            dozer->x = 0;
        }
        if (dozer->x >= env->size) {
            dozer->x = env->size - 1;
        }
        if (dozer->y < 0) {
            dozer->y = 0;
        }
        if (dozer->y >= env->size) {
            dozer->y = env->size - 1;
        }

        // Teleportitis
        if (env->tick % 512 == 0) {
             do {
                 env->dozers[i].x = rand() % env->size;
                 env->dozers[i].y = rand() % env->size;
                 env->stuck_count[i] = 0;
             } while (env->map[map_idx(env, env->dozers[i].x, env->dozers[i].y)] != 0.0f);
        }
 
    }
    int marked_to_skip[env->num_agents];
    memset(marked_to_skip, 0, env->num_agents*sizeof(int));
    for(int i = 0; i < env->num_agents; i++) {
        if(marked_to_skip[i]) {
            continue;
        }
        // compute delta progress
        if (env->initial_total_delta > 0) {
            env->delta_progress = 1.0f - (env->current_total_delta / env->initial_total_delta);
            env->delta_progress = fmaxf(0.0f, fminf(1.0f, env->delta_progress));
        }
    }
   
    //printf("observations\n");
    compute_all_observations(env);
    //int action = env->actions[0];
}

void c_close(Terraform* env) {
    free_initialized(env);
}

Mesh* create_heightmap_mesh(float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    // NOTE: One vertex per pixel
    Mesh* mesh = (Mesh*)calloc(1, sizeof(Mesh));
    mesh->triangleCount = (mapX - 1)*(mapZ - 1)*2;    // One quad every four pixels

    mesh->vertexCount = mesh->triangleCount*3;

    mesh->vertices = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->normals = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->texcoords = (float *)RL_MALLOC(mesh->vertexCount*2*sizeof(float));
    mesh->colors = NULL;
    UploadMesh(mesh, false);
    return mesh;
}

void update_heightmap_mesh(Mesh* mesh, float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    int vCounter = 0;       // Used to count vertices float by float
    int tcCounter = 0;      // Used to count texcoords float by float
    int nCounter = 0;       // Used to count normals float by float

    //Vector3 scaleFactor = { size.x/(mapX - 1), 1.0f, size.z/(mapZ - 1) };
    Vector3 scaleFactor = { 1.0f, 1.0f, 1.0f};

    Vector3 vA = { 0 };
    Vector3 vB = { 0 };
    Vector3 vC = { 0 };
    Vector3 vN = { 0 };

    for (int z = 0; z < mapZ-1; z++)
    {
        for (int x = 0; x < mapX-1; x++)
        {
            // Fill vertices array with data
            //----------------------------------------------------------

            // one triangle - 3 vertex
            mesh->vertices[vCounter] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 1] = heightMap[x + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 2] = (float)z*scaleFactor.z;

            mesh->vertices[vCounter + 3] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 4] = heightMap[x + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 5] = (float)(z + 1)*scaleFactor.z;

            mesh->vertices[vCounter + 6] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 7] = heightMap[(x + 1) + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 8] = (float)z*scaleFactor.z;

            // Another triangle - 3 vertex
            mesh->vertices[vCounter + 9] = mesh->vertices[vCounter + 6];
            mesh->vertices[vCounter + 10] = mesh->vertices[vCounter + 7];
            mesh->vertices[vCounter + 11] = mesh->vertices[vCounter + 8];

            mesh->vertices[vCounter + 12] = mesh->vertices[vCounter + 3];
            mesh->vertices[vCounter + 13] = mesh->vertices[vCounter + 4];
            mesh->vertices[vCounter + 14] = mesh->vertices[vCounter + 5];

            mesh->vertices[vCounter + 15] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 16] = heightMap[(x + 1) + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 17] = (float)(z + 1)*scaleFactor.z;
            vCounter += 18;     // 6 vertex, 18 floats

            // Fill texcoords array with data
            //--------------------------------------------------------------
            mesh->texcoords[tcCounter] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 1] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 2] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 3] = (float)(z + 1)/(mapZ - 1);

            mesh->texcoords[tcCounter + 4] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 5] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 6] = mesh->texcoords[tcCounter + 4];
            mesh->texcoords[tcCounter + 7] = mesh->texcoords[tcCounter + 5];

            mesh->texcoords[tcCounter + 8] = mesh->texcoords[tcCounter + 2];
            mesh->texcoords[tcCounter + 9] = mesh->texcoords[tcCounter + 3];

            mesh->texcoords[tcCounter + 10] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 11] = (float)(z + 1)/(mapZ - 1);
            tcCounter += 12;    // 6 texcoords, 12 floats

            // Fill normals array with data
            //--------------------------------------------------------------
            for (int i = 0; i < 18; i += 9)
            {
                vA.x = mesh->vertices[nCounter + i];
                vA.y = mesh->vertices[nCounter + i + 1];
                vA.z = mesh->vertices[nCounter + i + 2];

                vB.x = mesh->vertices[nCounter + i + 3];
                vB.y = mesh->vertices[nCounter + i + 4];
                vB.z = mesh->vertices[nCounter + i + 5];

                vC.x = mesh->vertices[nCounter + i + 6];
                vC.y = mesh->vertices[nCounter + i + 7];
                vC.z = mesh->vertices[nCounter + i + 8];

                vN = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(vB, vA), Vector3Subtract(vC, vA)));

                mesh->normals[nCounter + i] = vN.x;
                mesh->normals[nCounter + i + 1] = vN.y;
                mesh->normals[nCounter + i + 2] = vN.z;

                mesh->normals[nCounter + i + 3] = vN.x;
                mesh->normals[nCounter + i + 4] = vN.y;
                mesh->normals[nCounter + i + 5] = vN.z;

                mesh->normals[nCounter + i + 6] = vN.x;
                mesh->normals[nCounter + i + 7] = vN.y;
                mesh->normals[nCounter + i + 8] = vN.z;
            }

            nCounter += 18;     // 6 vertex, 18 floats
        }
    }

    // Upload vertex data to GPU (static mesh)
    UpdateMeshBuffer(*mesh, 0, mesh->vertices, mesh->vertexCount * 3 * sizeof(float), 0); // Update vertices
    UpdateMeshBuffer(*mesh, 2, mesh->normals, mesh->vertexCount * 3 * sizeof(float), 0); // Update normals
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    Texture2D ball;
    Camera3D camera;
    Mesh* mesh;
    Model model;
    Mesh* target_mesh;
    Model target_model;
    Texture2D texture;
    Model dozer;
    Shader shader;
    Shader target_shader;
    Texture2D shader_terrain;
    int shader_terrain_loc;
    unsigned char *shader_terrain_data;
};

Client* make_client(Terraform* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(1080, 720, "PufferLib Terraform");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetTargetFPS(60);
    Camera3D camera = { 0 };
                                                       //
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
    camera.position = (Vector3){ 3*env->size/4, env->size, 3*env->size/4};
    camera.target = (Vector3){ env->size/2, 0, env->size/2-1};
    client->camera = camera;

    client->shader = LoadShader(
        TextFormat("resources/terraform/shader_%i.vs", GLSL_VERSION),
        TextFormat("resources/terraform/shader_%i.fs", GLSL_VERSION)
    );
    client->target_shader = LoadShader(
        TextFormat("resources/terraform/shader_%i.vs", GLSL_VERSION),
        TextFormat("resources/terraform/target_shader_%i.fs", GLSL_VERSION)
    );

    Image img = GenImageColor(env->size, env->size, WHITE);
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    client->shader_terrain = LoadTextureFromImage(img);
    UnloadImage(img);

    client->shader_terrain_loc = GetShaderLocation(client->target_shader, "terrain");
    SetShaderValueTexture(client->target_shader, client->shader_terrain_loc, client->shader_terrain);

    client->shader_terrain_data = calloc(4*env->size*env->size, sizeof(unsigned char));

    int shader_width_loc = GetShaderLocation(client->target_shader, "width");
    SetShaderValue(client->target_shader, shader_width_loc, &env->size, SHADER_UNIFORM_INT);

    int shader_height_loc = GetShaderLocation(client->target_shader, "height");
    SetShaderValue(client->target_shader, shader_height_loc, &env->size, SHADER_UNIFORM_INT);
 
    //Image checked = GenImageChecked(env->size, env->size, 2, 2, PUFF_RED, PUFF_CYAN);
    img = LoadImage("resources/terraform/perlin.jpg");
    client->texture = LoadTextureFromImage(img);
    client->dozer = LoadModel("resources/terraform/dozer.glb");
    UnloadImage(img);
    client->mesh = NULL;
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client->mesh);
    free(client->shader_terrain_data);
    free(client->target_mesh);
    free(client);
    
}

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
            -(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            (current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Apply 45-degree rotation to the movement
        // For a -45 degree rotation (clockwise)
        float cos45 = -0.7071f;  // cos(-45°)
        float sin45 = 0.7071f; // sin(-45°)
        Vector2 rotated_delta = {
            delta.x * cos45 - delta.y * sin45,
            delta.x * sin45 + delta.y * cos45
        };

        // Update camera position (only X and Y)
        client->camera.position.z += rotated_delta.x;
        client->camera.position.x += rotated_delta.y;

        // Update camera target (only X and Y)
        client->camera.target.z += rotated_delta.x;
        client->camera.target.x += rotated_delta.y;

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

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void c_render(Terraform* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        env->client->mesh = create_heightmap_mesh(env->map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(env->client->mesh, env->map, (Vector3){env->size, 1, env->size});
        env->client->model = LoadModelFromMesh(*env->client->mesh);

        env->client->target_mesh = create_heightmap_mesh(env->target_map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(env->client->target_mesh, env->target_map, (Vector3){env->size, 1, env->size});
        env->client->target_model = LoadModelFromMesh(*env->client->target_mesh);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    Client* client = env->client;

    handle_camera_controls(client);
    //Camera3D* camera = &client->camera;
    //camera->position = (Vector3){ x+30, z+100.0f, y+30 };
    //camera->target = (Vector3){ x, 0, y-1};
    rlSetBlendFactorsSeparate(RL_SRC_ALPHA, RL_ONE_MINUS_SRC_ALPHA, RL_ONE, RL_ONE, RL_FUNC_ADD, RL_MAX);

    if (env->tick % 10 == 0) {
        update_heightmap_mesh(client->mesh, env->map, (Vector3){env->size, 1, env->size});
        update_heightmap_mesh(client->target_mesh, env->target_map, (Vector3){env->size, 1, env->size});
        for (int i = 0; i < env->size*env->size; i++) {
            client->shader_terrain_data[4*i] = env->map[i];
            client->shader_terrain_data[4*i+3] = 255;
        }
        UpdateTexture(client->shader_terrain, client->shader_terrain_data);
        SetShaderValueTexture(client->target_shader, env->client->shader_terrain_loc, env->client->shader_terrain);

    }
    //client->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = client->texture;
    client->model.materials[0].shader = client->shader;

    //client->target_model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = client->texture;
    client->target_model.materials[0].shader = client->target_shader;

    //update_heightmap_mesh(client->mesh, env->map, (Vector3){env->size, 1, env->size});
    //client->model = LoadModelFromMesh(*client->mesh);

    BeginDrawing();
    ClearBackground((Color){143, 86, 29, 255});
    BeginMode3D(client->camera);
    /*
    for(int i = 0; i < env->size*env->size; i++) {
        float height = env->map[i];
        int x = i%env->size;
        int z = i/env->size;
        DrawCube((Vector3){x, height, z}, 1.0f, 1.0f, 1.0f, DARKGREEN);
        DrawCubeWires((Vector3){x, height, z}, 1.0f, 1.0f, 1.0f, MAROON);
    }
    */

    BeginShaderMode(client->shader);
    DrawModel(client->model, (Vector3){0, 0, 0}, 1.0f, (Color){156, 50, 20, 255});
    EndShaderMode();
    rlDisableDepthTest();  // Add this line

    BeginBlendMode(RL_BLEND_CUSTOM_SEPARATE);  // Add this line
    BeginShaderMode(client->target_shader);
    DrawModel(client->target_model, (Vector3){0, 0, 0}, 1.0f, (Color){156, 50, 20, 255});
    EndShaderMode();
    EndBlendMode();
    rlEnableDepthTest();   // Add this line
    // for(int i = 0; i < env->size; i += 11){
    //     // draw grid lines every 11 units
    //     DrawLine3D((Vector3){i, 0, 0}, (Vector3){i, 0, env->size-1}, RED);
    //     DrawLine3D((Vector3){0, 0, i}, (Vector3){env->size-1, 0, i}, RED);
    // }
    for (int i = 0; i < env->num_agents; i++) {
        Dozer* dozer = &env->dozers[i];
        int x = (int)dozer->x;
        int z = (int)dozer->y;  
        int size = (int)env->size;
        
        // Get height from map using correct indexing
        float y = env->map[z * size + x] + 0.5f;
        float yy = y;
        rlPushMatrix();
        rlTranslatef(dozer->x, y, dozer->y);
        rlRotatef(-90.f - dozer->heading*RAD2DEG, 0, 1, 0);
        // if(i ==0 ){
        //     DrawCube((Vector3){0,50,0}, 10.0f, 10.0f, 10.0f, RED);
        // }
        DrawModel(client->dozer, (Vector3){0, 0, 0}, 0.25f, WHITE);
        rlPopMatrix();
        // DrawCube((Vector3){dozer->x, y, dozer->y}, 1.0f, 1.0f, 1.0f, PUFF_WHITE);
        if(IsKeyDown(KEY_LEFT_CONTROL) && i == 0) {
            int dialate = 1;
            int x_offset = env->dozers[i].x - dialate*VISION;
            int y_offset = env->dozers[i].y - dialate*VISION;
            for (int x = 0; x < 2*dialate*VISION + 1; x+=dialate) {
                for (int y = 0; y < 2*dialate*VISION + 1; y+=dialate) {
                    if(x_offset + x < 0 || x_offset + x >= env->size || y_offset + y < 0 || y_offset + y >= env->size) {
                        continue;
                    }
                    float obs_x = x_offset + x;
                    float obs_y = y_offset + y;
                    Color clr = PUFF_WHITE;
                    int idx = y*(2*VISION+1) + x;
                    int obs_idx = 319*i + 121 + idx;
                    if(env->observations[obs_idx] == 1.0f) {
                        clr = GREEN;
                    } else if(env->observations[obs_idx] == 0.66f) {
                        clr = PUFF_RED;
                    } else if(env->observations[obs_idx] == 0.33f) {
                        clr = YELLOW;
                    }
                    for(int j = 0; j < (2*SCOOP_SIZE + 1)*(2*SCOOP_SIZE + 1); j++) {
                        if(env->dozers[i].load_indices[j] == map_idx(env, x_offset + x, y_offset + y)){
                            clr = BLUE;
                            break;
                        }
                    }
                    DrawCube((Vector3){x_offset + x, yy, y_offset + y}, 0.5f, 0.5f, 0.5f, clr);
                }
            }
            int step = 1;
            for (int k = 0; k < env->size; k += step) {
                for (int l = 0; l < env->size; l += step) {
                int idx = k * env->size + l;
                    Color color = RED;
                    if (env->grid_indices[idx] == env->dozers[i].target_quadrant) {
                        color = GREEN;
                        DrawSphere((Vector3){l, 0, k}, 0.1f, color);

                    }
                }
            }
            for(int j = 0; j < env->num_quadrants; j++){
                Color color = PURPLE;
                if(env->quadrant_volume_deltas[j] > 0.0f) {
                    color = RED;
                } else if(env->quadrant_volume_deltas[j] < 0.0f) {
                    color = GREEN;
                }
                DrawLine3D((Vector3){env->quadrant_centroids[j*2], 0, env->quadrant_centroids[j*2+1]}, (Vector3){env->dozers[i].x, 0, env->dozers[i].y}, color);
            }
        }
        
        
    }
    EndMode3D();
    //DrawText(TextFormat("Dozer x: %f", x), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("score: %f", env->delta_progress), 10, 170, 20, PUFF_WHITE);
    DrawText(TextFormat("load: %f", env->dozers[0].load), 10, 190, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->tick), 10, 210, 20, PUFF_WHITE);
    DrawText(TextFormat("Current Quadrant: %d", env->grid_indices[map_idx(env, env->dozers[0].x, env->dozers[0].y)]), 10, 230, 20, PUFF_WHITE);
    DrawFPS(10, 10);
    EndDrawing();
}
