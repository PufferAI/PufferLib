#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "raylib.h"

// Constants for the simulation
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define NUM_COLONIES 2
#define MAX_FOOD_SOURCES 20
#define MAX_FOOD_PER_SOURCE 20
#define ANT_SPEED 5.0f
#define ANT_SIZE 4
#define FOOD_SIZE 6
#define COLONY_SIZE 20
// #define PHEROMONE_EVAPORATION_RATE 0.001f
// #define PHEROMONE_DEPOSIT_AMOUNT 1.0f
// #define MAX_PHEROMONES 5000
// #define PHEROMONE_SIZE 2
// #define ANT_VISION_RANGE 500.0f
// #define ANT_VISION_ANGLE (M_PI / 2)
#define TURN_ANGLE (M_PI / 4)
#define MIN_FOOD_COLONY_DISTANCE 50.0f
#define ANT_LIFETIME 5000

// Actions
#define ACTION_TURN_LEFT 0
#define ACTION_TURN_RIGHT 1
// #define ACTION_DROP_PHEROMONE 2
#define ACTION_MOVE_FORWARD 3

// Colors
#define COLONY1_COLOR (Color){220, 0, 0, 255}
#define COLONY2_COLOR (Color){0, 0, 220, 255}
// #define PHEROMONE1_COLOR (Color){255, 200, 200, 100}
// #define PHEROMONE2_COLOR (Color){200, 200, 255, 100}
#define FOOD_COLOR (Color){0, 200, 0, 255}
#define BACKGROUND_COLOR (Color){50, 50, 50, 255}

// Required Log struct for PufferLib
typedef struct Log Log;
struct Log {
    float perf;              // Performance metric
    float score;             // Total score
    float episode_return;    // Cumulative rewards
    float episode_length;    // Episode duration
    float reward;            // Reward for the current step
    float n;                 // Episode count - REQUIRED AS LAST FIELD
};

// Forward declarations
typedef struct Client Client;
typedef struct AntsEnv AntsEnv;

// Environment structs
typedef struct {
    float x, y;
} Vector2D;

typedef struct {
    Vector2D position;
    int amount;
} FoodSource;

// typedef struct {
//     Vector2D position;
//     float strength;
//     int colony_id;
// } Pheromone;

typedef struct {
    Vector2D position;
    float direction;
    int colony_id;
    bool has_food;
    int lifetime;            // Track ant lifetime for performance metrics

    // Tracking for reward shaping
    float prev_dist_to_objective;  // Previous distance to current objective (food or colony)
    int steps_with_food;           // Steps taken while carrying food (for efficiency bonus)
} Ant;

typedef struct {
    Vector2D position;
    int food_collected;
} Colony;

// Raylib client structure - FOLLOWING SNAKE PATTERN
struct Client {
    int cell_size;
    int width;
    int height;
};

// Main environment struct - RESTRUCTURED FOLLOWING SNAKE PATTERN
struct AntsEnv {
    // Required PufferLib fields - IDENTICAL TO SNAKE
    float* observations;        // Flattened observations for all ants
    int* actions;              // Actions for all ants
    float* rewards;            // Rewards for all ants
    unsigned char* terminals;   // Terminal flags
    Log log;                   // Main aggregated log
    Log* ant_logs;             // Individual ant logs - CRITICAL ADDITION
    
    // Environment state
    Colony colonies[NUM_COLONIES];
    Ant* ants;                 // Dynamic array of all ants
    FoodSource food_sources[MAX_FOOD_SOURCES];
    // Pheromone pheromones[MAX_PHEROMONES];
    // int num_pheromones;
    int num_food_sources;
    
    // Environment parameters
    int num_ants;              // Total number of ants
    int width;                 // Environment width
    int height;                // Environment height
    int obs_size;              // Observation size per ant
    int tick;                  // Current timestep
    
    // Reward parameters
    float reward_food;
    float reward_delivery;
    float reward_death;
    float reward_demo_match;      // Reward for matching demo action
    float reward_demo_mismatch;   // Penalty for not matching demo action

    // New reward shaping parameters
    float reward_progress;           // Reward for moving closer to objective
    float reward_time_penalty;       // Small penalty per step (encourages efficiency)
    float reward_wrong_direction;    // Penalty for moving away from objective
    float reward_efficiency_bonus;   // Bonus multiplier for fast deliveries
    
    // Rendering
    Client* client;            // Raylib client
    int cell_size;
};

/**
 * Add an ant's log to the main log when the ant's episode ends.
 * CRITICAL FUNCTION - COPIED FROM SNAKE PATTERN
 * This should only be called during termination conditions for a specific ant.
 * Accumulates the ant's stats into the main log and resets the ant's individual log.
 */
void add_log(AntsEnv* env, int ant_id) {
    env->log.perf += env->ant_logs[ant_id].perf;
    env->log.score += env->ant_logs[ant_id].score;
    env->log.episode_return += env->ant_logs[ant_id].episode_return;
    env->log.episode_length += env->ant_logs[ant_id].episode_length;
    env->log.n += 1;
    env->log.reward += env->ant_logs[ant_id].reward;
    // Reset individual ant log
    env->ant_logs[ant_id] = (Log){0};
}

// Memory management functions - FOLLOWING SNAKE PATTERN
void init_ants_env(AntsEnv* env) {
    env->ants = (Ant*)calloc(env->num_ants, sizeof(Ant));
    env->ant_logs = (Log*)calloc(env->num_ants, sizeof(Log));
    env->tick = 0;
    env->client = NULL;
    // env->num_pheromones = 0;
    
    // Initialize food sources
    env->num_food_sources = MAX_FOOD_SOURCES;
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0; // Will be set in reset
    }
    
    // Initialize colonies
    env->colonies[0].position = (Vector2D){env->width / 4, env->height / 2};
    env->colonies[1].position = (Vector2D){3 * env->width / 4, env->height / 2};
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;
}

void allocate_ants_env(AntsEnv* env) {
    env->obs_size = 8; // Fixed observation size per ant (removed 2 pheromone slots)
    env->observations = (float*)calloc(env->num_ants * env->obs_size, sizeof(float));
    env->actions = (int*)calloc(env->num_ants, sizeof(int));
    env->rewards = (float*)calloc(env->num_ants, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_ants, sizeof(unsigned char));
    init_ants_env(env);
}

void c_close(AntsEnv* env) {
    if (env->ants) {
        free(env->ants);
        env->ants = NULL;
    }
    if (env->ant_logs) {
        free(env->ant_logs);
        env->ant_logs = NULL;
    }
}

void free_ants_env(AntsEnv* env) {
    c_close(env);
    if (env->observations) {
        free(env->observations);
        env->observations = NULL;
    }
    if (env->actions) {
        free(env->actions);
        env->actions = NULL;
    }
    if (env->rewards) {
        free(env->rewards);
        env->rewards = NULL;
    }
    if (env->terminals) {
        free(env->terminals);
        env->terminals = NULL;
    }
}

// Helper function implementations
static inline float random_float(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

static inline float wrap_angle(float angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

static inline float distance_squared(Vector2D a, Vector2D b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return (dx * dx) + (dy * dy);
}

static inline float get_angle(Vector2D a, Vector2D b) {
    return atan2(b.y - a.y, b.x - a.x);
}

static inline bool is_in_vision(Vector2D ant_pos, Vector2D target) {
    // Vision range removed - ants can see everything
    return true;
}

// Get the target position for an ant (colony if carrying food, nearest food otherwise)
static inline Vector2D get_ant_target(AntsEnv* env, Ant* ant) {
    if (ant->has_food) {
        // Target is the colony when carrying food
        return env->colonies[ant->colony_id].position;
    } else {
        // Target is the nearest food source when not carrying food
        float closest_food_dist_sq = env->width * env->width + env->height * env->height;
        Vector2D closest_food_pos = ant->position; // Default to current position if no food found

        for (int i = 0; i < env->num_food_sources; i++) {
            if (env->food_sources[i].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[i].position);
                if (dist_sq < closest_food_dist_sq) {
                    closest_food_dist_sq = dist_sq;
                    closest_food_pos = env->food_sources[i].position;
                }
            }
        }

        return closest_food_pos;
    }
}



// static inline void add_pheromone(AntsEnv* env, Vector2D position, int colony_id) {
//     if (env->num_pheromones >= MAX_PHEROMONES) {
//         // Replace oldest pheromone
//         for (int i = 0; i < env->num_pheromones - 1; i++) {
//             env->pheromones[i] = env->pheromones[i + 1];
//         }
//         env->num_pheromones--;
//     }
//
//     env->pheromones[env->num_pheromones].position = position;
//     env->pheromones[env->num_pheromones].strength = PHEROMONE_DEPOSIT_AMOUNT;
//     env->pheromones[env->num_pheromones].colony_id = colony_id;
//     env->num_pheromones++;
// }

void get_observation_for_ant(AntsEnv* env, int ant_idx, float* obs) {
    Ant* ant = &env->ants[ant_idx];
    Colony* colony = &env->colonies[ant->colony_id];

    // Observation structure (8 elements):
    // [0-1]: ant position (normalized)
    // [2]: ant direction (normalized between 0 and 1)
    // [3]: has_food (0 or 1)
    // [4]: direction to colony (normalized between 0 and 1)
    // [5]: distance to colony (normalized)
    // [6]: direction to closest food (normalized between 0 and 1)
    // [7]: closest food distance (normalized)

    obs[0] = ant->position.x / env->width;
    obs[1] = ant->position.y / env->height;

    // Normalize direction to 0-1 range (0 = right, 0.25 = up, 0.5 = left, 0.75 = down)
    obs[2] = (ant->direction + M_PI) / (2 * M_PI);

    obs[3] = ant->has_food ? 1.0f : 0.0f;

    // Get direction to colony (normalized between 0 and 1)
    float angle_to_colony = wrap_angle(get_angle(ant->position, colony->position));
    obs[4] = (angle_to_colony + M_PI) / (2 * M_PI);

    obs[5] = distance_squared(ant->position, colony->position) / (env->width * env->width + env->height * env->height);

    // Find closest visible food
    float closest_food_dist_sq = env->width * env->width;
    Vector2D closest_food_pos = {0, 0};
    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount > 0) {
            float dist_sq = distance_squared(ant->position, env->food_sources[i].position);
            if (
                dist_sq < closest_food_dist_sq
                && is_in_vision(ant->position, env->food_sources[i].position)
            ) {
                closest_food_dist_sq = dist_sq;
                closest_food_pos.x = env->food_sources[i].position.x;
                closest_food_pos.y = env->food_sources[i].position.y;
            }
        }
    }

    if(closest_food_pos.x == 0 && closest_food_pos.y == 0) {
        obs[6] = -1.0f;
        obs[7] = -1.0f;
    }
    else {
        // Get direction to closest food (normalized between 0 and 1)
        float angle_to_food = wrap_angle(get_angle(ant->position, closest_food_pos));
        obs[6] = (angle_to_food + M_PI) / (2 * M_PI);
        obs[7] = closest_food_dist_sq / ((env->width * env->width) + (env->height * env->height));
    }
}

void compute_observations(AntsEnv* env) {
    for (int i = 0; i < env->num_ants; i++) {
        get_observation_for_ant(env, i, &env->observations[i * env->obs_size]);
    }
}

void spawn_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    Colony* colony = &env->colonies[ant->colony_id];

    ant->position = colony->position;
    ant->direction = wrap_angle((rand() % 8) * (M_PI / 4)); // Randomly choose between 8 directions (0, 45, 90, 135, 180, 225, 270, 315 degrees)
    ant->has_food = false;
    ant->lifetime = random_float(0, ANT_LIFETIME);

    // Initialize reward shaping tracking
    ant->prev_dist_to_objective = -1.0f;  // -1 indicates uninitialized
    ant->steps_with_food = 0;

    // Reset individual ant log
    env->ant_logs[ant_id] = (Log){0};
}

void spawn_food(AntsEnv* env) {
    int idx;
    bool valid_position;
    int attempts = 0;
    
    do {
        float x = random_float(50, env->width - 50);
        float y = random_float(50, env->height - 50);
        
        valid_position = true;
        for (int j = 0; j < NUM_COLONIES; j++) {
            float dist_sq = distance_squared((Vector2D){x, y}, env->colonies[j].position);
            if (dist_sq < MIN_FOOD_COLONY_DISTANCE * MIN_FOOD_COLONY_DISTANCE) {
                valid_position = false;
                break;
            }
        }
        
        if (valid_position) {
            // Find an empty food source slot
            for (idx = 0; idx < env->num_food_sources; idx++) {
                if (env->food_sources[idx].amount == 0) {
                    env->food_sources[idx].position.x = x;
                    env->food_sources[idx].position.y = y;
                    env->food_sources[idx].amount = MAX_FOOD_PER_SOURCE;
                    return;
                }
            }
        }
        attempts++;
    } while (!valid_position && attempts < 100);
}

void c_reset(AntsEnv* env) {
    env->tick = 0;
    env->log = (Log){0};
    // env->num_pheromones = 0;
    
    // Reset colonies
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;
    
    // Initialize all ants
    int ant_idx = 0;
    for (int i = 0; i < NUM_COLONIES; i++) {
        for (int j = 0; j < env->num_ants / NUM_COLONIES; j++) {
            env->ants[ant_idx].colony_id = i;
            spawn_ant(env, ant_idx);
            ant_idx++;
        }
    }
    
    // Clear food sources and spawn new ones
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
    
    for (int i = 0; i < env->num_food_sources; i++) {
        spawn_food(env);
    }
    
    // Clear buffers
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));
    
    // Generate initial observations
    compute_observations(env);
}

// Compute the hardcoded demo action for an ant
// This replicates the logic from demo() lines 176-226
int get_demo_action(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    // Threshold is half of turn angle to avoid oscillation with 45-degree turns
    const float turn_threshold = TURN_ANGLE / 2.0f; // ~22.5 degrees

    if (ant->has_food) {
        // If ant has food, return to colony
        Colony* colony = &env->colonies[ant->colony_id];
        float angle_to_colony = get_angle(ant->position, colony->position);
        float angle_diff = wrap_angle(angle_to_colony - ant->direction);

        // Turn towards colony if angle difference is significant
        if (angle_diff > turn_threshold) {
            return ACTION_TURN_RIGHT;
        } else if (angle_diff < -turn_threshold) {
            return ACTION_TURN_LEFT;
        } else {
            return ACTION_MOVE_FORWARD;
        }
    } else {
        // If ant doesn't have food, seek nearest food source
        float closest_food_dist_sq = env->width * env->width;
        Vector2D closest_food_pos = {0, 0};
        bool found_food = false;

        for (int j = 0; j < env->num_food_sources; j++) {
            if (env->food_sources[j].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[j].position);
                if (dist_sq < closest_food_dist_sq && is_in_vision(ant->position, env->food_sources[j].position)) {
                    closest_food_dist_sq = dist_sq;
                    closest_food_pos = env->food_sources[j].position;
                    found_food = true;
                }
            }
        }

        if (found_food) {
            // Turn towards food if angle difference is significant
            float angle_to_food = get_angle(ant->position, closest_food_pos);
            float angle_diff = wrap_angle(angle_to_food - ant->direction);

            if (angle_diff > turn_threshold) {
                return ACTION_TURN_RIGHT;
            } else if (angle_diff < -turn_threshold) {
                return ACTION_TURN_LEFT;
            } else {
                return ACTION_MOVE_FORWARD;
            }
        } else {
            // If no food in sight, move forward (we'll use this as the "default" demo action)
            // Note: The random turning behavior is not deterministic, so we default to forward
            return ACTION_MOVE_FORWARD;
        }
    }
}

void step_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    env->ant_logs[ant_id].episode_length += 1;
    ant->lifetime++;

    int action = env->actions[ant_id];

    // Compute demo action and compare with agent's action
    int demo_action = get_demo_action(env, ant_id);
    if (action == demo_action) {
        // Reward for matching the demo action
        env->rewards[ant_id] += env->reward_demo_match;
        env->ant_logs[ant_id].episode_return += env->reward_demo_match;
        env->ant_logs[ant_id].reward += env->reward_demo_match;
    } else {
        // Punish for not matching the demo action
        env->rewards[ant_id] += env->reward_demo_mismatch;
        env->ant_logs[ant_id].episode_return += env->reward_demo_mismatch;
        env->ant_logs[ant_id].reward += env->reward_demo_mismatch;
    }

    // Execute action
    switch (action) {
        case ACTION_TURN_LEFT:
            ant->direction -= TURN_ANGLE;
            ant->direction = wrap_angle(ant->direction);
            break;
        case ACTION_TURN_RIGHT:
            ant->direction += TURN_ANGLE;
            ant->direction = wrap_angle(ant->direction);
            break;
        // case ACTION_DROP_PHEROMONE:
        //     // Only drop pheromones when carrying food
        //     if (ant->has_food) {
        //         add_pheromone(env, ant->position, ant->colony_id);
        //     }
        //     break;
        case ACTION_MOVE_FORWARD:
            // Move forward only when this action is selected
            ant->position.x += ANT_SPEED * cos(ant->direction);
            ant->position.y += ANT_SPEED * sin(ant->direction);
            break;
    }

    // Wrap around edges
    if (ant->position.x < 0) ant->position.x = env->width;
    if (ant->position.x > env->width) ant->position.x = 0;
    if (ant->position.y < 0) ant->position.y = env->height;
    if (ant->position.y > env->height) ant->position.y = 0;

    // REWARD SHAPING: Progress-based rewards
    // Give rewards for moving toward objective, penalty for moving away
    Vector2D objective_pos;
    if (ant->has_food) {
        // Objective is home colony
        objective_pos = env->colonies[ant->colony_id].position;
        ant->steps_with_food++;
    } else {
        // Objective is nearest food source
        float closest_food_dist_sq = env->width * env->width + env->height * env->height;
        for (int j = 0; j < env->num_food_sources; j++) {
            if (env->food_sources[j].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[j].position);
                if (dist_sq < closest_food_dist_sq) {
                    closest_food_dist_sq = dist_sq;
                    objective_pos = env->food_sources[j].position;
                }
            }
        }
    }

    // Calculate current distance to objective
    float current_dist = sqrtf(distance_squared(ant->position, objective_pos));

    // On first step or after picking up food, initialize previous distance
    if (ant->prev_dist_to_objective < 0) {
        ant->prev_dist_to_objective = current_dist;
    }

    // Calculate progress (positive if moving closer, negative if moving away)
    float progress = ant->prev_dist_to_objective - current_dist;

    // Only give progress rewards if ant actually moved (action was MOVE_FORWARD)
    if (action == ACTION_MOVE_FORWARD) {
        if (progress > 0) {
            // Moving closer to objective
            float progress_reward = env->reward_progress * progress;
            env->rewards[ant_id] += progress_reward;
            env->ant_logs[ant_id].episode_return += progress_reward;
            env->ant_logs[ant_id].reward += progress_reward;
        } else if (progress < 0) {
            // Moving away from objective (penalty)
            float wrong_dir_penalty = env->reward_wrong_direction * progress; // progress is negative
            env->rewards[ant_id] += wrong_dir_penalty;
            env->ant_logs[ant_id].episode_return += wrong_dir_penalty;
            env->ant_logs[ant_id].reward += wrong_dir_penalty;
        }
    }

    // Update previous distance for next step
    ant->prev_dist_to_objective = current_dist;

    // Time penalty (encourages efficiency)
    env->rewards[ant_id] += env->reward_time_penalty;
    env->ant_logs[ant_id].episode_return += env->reward_time_penalty;
    env->ant_logs[ant_id].reward += env->reward_time_penalty;

    // Check for food collection
    if (!ant->has_food) {
        for (int j = 0; j < env->num_food_sources; j++) {
            if (env->food_sources[j].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[j].position);
                if (dist_sq < (ANT_SIZE + FOOD_SIZE) * (ANT_SIZE + FOOD_SIZE)) {
                    ant->has_food = true;
                    env->food_sources[j].amount--;

                    // If food source is exhausted, respawn it
                    if (env->food_sources[j].amount <= 0) {
                        spawn_food(env);
                    }

                    env->rewards[ant_id] += env->reward_food;
                    env->ant_logs[ant_id].episode_return += env->reward_food;
                    env->ant_logs[ant_id].reward += env->reward_food;

                    // Reset tracking for new objective (now need to return to colony)
                    ant->prev_dist_to_objective = -1.0f;
                    ant->steps_with_food = 0;
                    break;
                }
            }
        }
    }

    // Check for food delivery
    if (ant->has_food) {
        Colony* colony = &env->colonies[ant->colony_id];
        float dist_sq = distance_squared(ant->position, colony->position);
        if (dist_sq < (ANT_SIZE + COLONY_SIZE) * (ANT_SIZE + COLONY_SIZE)) {
            ant->has_food = false;
            colony->food_collected++;

            // Base delivery reward
            float delivery_reward = env->reward_delivery;

            // Efficiency bonus: reward faster deliveries
            // Normalize by expected optimal steps (width/2 / ANT_SPEED = ~128 steps average)
            // Bonus decreases as steps_with_food increases
            if (env->reward_efficiency_bonus > 0 && ant->steps_with_food > 0) {
                float expected_steps = env->width / (2.0f * ANT_SPEED);
                float efficiency_ratio = expected_steps / (float)ant->steps_with_food;
                // Only give bonus if delivery was faster than expected
                if (efficiency_ratio > 1.0f) {
                    float efficiency_bonus = env->reward_efficiency_bonus * (efficiency_ratio - 1.0f);
                    delivery_reward += efficiency_bonus;
                }
            }

            env->rewards[ant_id] += delivery_reward;
            env->ant_logs[ant_id].episode_return += delivery_reward;
            env->ant_logs[ant_id].score += 1;
            env->ant_logs[ant_id].reward += delivery_reward;

            // Reset tracking for new foraging trip
            ant->prev_dist_to_objective = -1.0f;
            ant->steps_with_food = 0;
        }
    }

    // MULTIPLE TERMINAL CONDITIONS FOR FREQUENT LOG GENERATION
    bool should_terminate = false;
    
    // Terminal Condition 1: Shorter lifetime limit (similar to snake death frequency)
    // if (ant->lifetime > ANT_LIFETIME) {
    //     should_terminate = true;
    // }
    
    // // Terminal Condition 2: Random death chance (0.1% per step after lifetime)
    if (ant->lifetime > ANT_LIFETIME && (rand() % 1000) < 1) {
        should_terminate = true;
    }
    
    // // Terminal Condition 3: Performance-based termination after food delivery
    // if (env->ant_logs[ant_id].score > 0 && (rand() % 100) < 5) {
    //     should_terminate = true;
    // }
    
    // Execute termination and log aggregation
    if (should_terminate) {
        env->ant_logs[ant_id].perf = env->ant_logs[ant_id].episode_length > 0 ?
                                     env->ant_logs[ant_id].score / env->ant_logs[ant_id].episode_length : 0;
        add_log(env, ant_id);
        spawn_ant(env, ant_id); //Respawn the ant
        env->terminals[ant_id] = 1;
    }
}

void c_step(AntsEnv* env) {
    env->tick++;
    
    // Clear rewards and terminals
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));
    
    // Step all ants
    for (int i = 0; i < env->num_ants; i++) {
        step_ant(env, i);
    }

    // Update pheromones
    // for (int i = 0; i < env->num_pheromones; i++) {
    //     env->pheromones[i].strength -= PHEROMONE_EVAPORATION_RATE;
    //     if (env->pheromones[i].strength <= 0) {
    //         // Remove evaporated pheromone
    //         env->pheromones[i] = env->pheromones[env->num_pheromones - 1];
    //         env->num_pheromones--;
    //         i--;
    //     }
    // }

    // Generate new observations
    compute_observations(env);
}

// Raylib client functions - FOLLOWING SNAKE PATTERN
Client* make_client(int cell_size, int width, int height) {
    Client* client = (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ant Colony");
    SetTargetFPS(60);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(AntsEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    if (env->client == NULL) {
        env->client = make_client(1, env->width, env->height);
    }
    
    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);
    
    // Draw colonies
    for (int i = 0; i < NUM_COLONIES; i++) {
        Color colony_color = (i == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        DrawCircle(env->colonies[i].position.x, env->colonies[i].position.y, COLONY_SIZE, colony_color);
    }
    
    // Draw food sources
    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount > 0) {
            DrawCircle(env->food_sources[i].position.x, env->food_sources[i].position.y, 
                      FOOD_SIZE, FOOD_COLOR);
            DrawText(TextFormat("%d", env->food_sources[i].amount), 
                     env->food_sources[i].position.x, env->food_sources[i].position.y, 10, RAYWHITE);
        }
    }
    
    // Draw pheromones
    // for (int i = 0; i < env->num_pheromones; i++) {
    //     Color pheromone_color = (env->pheromones[i].colony_id == 0) ? PHEROMONE1_COLOR : PHEROMONE2_COLOR;
    //     pheromone_color.a = (unsigned char)(100 * env->pheromones[i].strength);
    //     DrawCircle(env->pheromones[i].position.x, env->pheromones[i].position.y,
    //               PHEROMONE_SIZE, pheromone_color);
    // }
    
    // Draw ants
    for (int i = 0; i < env->num_ants; i++) {
        Ant* ant = &env->ants[i];
        Color ant_color = (ant->colony_id == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        DrawCircle(ant->position.x, ant->position.y, ANT_SIZE, ant->has_food ? FOOD_COLOR : ant_color);
        
        // Draw direction indicator
        float dir_x = ant->position.x + (ANT_SIZE * 1.5f) * cos(ant->direction);
        float dir_y = ant->position.y + (ANT_SIZE * 1.5f) * sin(ant->direction);
        DrawLine(ant->position.x, ant->position.y, dir_x, dir_y, RAYWHITE);
    }
    
    // Draw UI
    DrawText(TextFormat("Colony 1 Food: %d", env->colonies[0].food_collected), 20, 20, 20, COLONY1_COLOR);
    DrawText(TextFormat("Colony 2 Food: %d", env->colonies[1].food_collected), 20, 50, 20, COLONY2_COLOR);
    DrawText(TextFormat("Tick: %d", env->tick), env->width - 120, 20, 20, RAYWHITE);
    
    EndDrawing();
}
