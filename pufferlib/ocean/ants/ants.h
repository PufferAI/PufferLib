/* Ants: A multiagent foraging environment inspired by ant colonies.
 * Two colonies compete to collect food from the environment.
 * Follows the Target env pattern for simplicity and clarity.
 */

#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Environment constants
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define NUM_COLONIES 2
#define MAX_FOOD_SOURCES 20
#define MAX_FOOD_PER_SOURCE 20
#define ANT_SPEED 5.0f
#define ANT_SIZE 4
#define FOOD_SIZE 6
#define COLONY_SIZE 20
#define TURN_ANGLE (M_PI / 4)
#define MIN_FOOD_COLONY_DISTANCE 50.0f
#define ANT_RESET_INTERVAL 2048  // Reset ant every N steps (like target.c)

// Actions
#define ACTION_TURN_LEFT 0
#define ACTION_TURN_RIGHT 1
#define ACTION_MOVE_FORWARD 2
#define ACTION_NOOP 3

// Colors
#define COLONY1_COLOR (Color){220, 0, 0, 255}
#define COLONY2_COLOR (Color){0, 0, 220, 255}
#define FOOD_COLOR (Color){0, 200, 0, 255}
#define BACKGROUND_COLOR (Color){50, 50, 50, 255}

// Required Log struct for PufferLib
typedef struct {
    float perf;              // Performance metric (score/length)
    float score;             // Total score (food deliveries)
    float episode_return;    // Cumulative rewards
    float episode_length;    // Episode duration
    float n;                 // Episode count - REQUIRED AS LAST FIELD
} Log;

// Forward declarations
typedef struct Client Client;
typedef struct AntsEnv AntsEnv;

// Simple 2D vector
typedef struct {
    float x, y;
} Vector2D;

// Food source in the environment
typedef struct {
    Vector2D position;
    int amount;
} FoodSource;

// Individual ant agent
typedef struct {
    Vector2D position;
    float direction;
    int colony_id;
    bool has_food;
    int steps_alive;  // Track steps for periodic reset
} Ant;

// Colony home base
typedef struct {
    Vector2D position;
    int food_collected;
} Colony;

// Raylib rendering client
struct Client {
    int cell_size;
    int width;
    int height;
};

// Main environment struct - FOLLOWING TARGET PATTERN
struct AntsEnv {
    Log log;                   // Required: aggregated log for all agents
    Client* client;            // Rendering client
    Ant* ants;                 // Dynamic array of ants
    Colony colonies[NUM_COLONIES];
    FoodSource food_sources[MAX_FOOD_SOURCES];

    // Required PufferLib fields
    float* observations;       // Flattened observations
    int* actions;             // Actions for all ants
    float* rewards;           // Rewards for all ants
    unsigned char* terminals; // Terminal flags

    // Environment parameters
    int num_ants;             // Total number of ants
    int width;                // Environment width
    int height;               // Environment height
    int num_food_sources;     // Active food sources
    int tick;                 // Current timestep

    // Simple reward parameters (like target.c)
    float reward_food_pickup;  // Reward for picking up food
    float reward_delivery;     // Reward for delivering food to colony
};

// Helper functions
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

static inline float clip(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Spawn a new ant at its colony
void spawn_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    Colony* colony = &env->colonies[ant->colony_id];

    ant->position = colony->position;
    ant->direction = wrap_angle((rand() % 8) * (M_PI / 4));
    ant->has_food = false;
    ant->steps_alive = 0;
}

// Spawn food at a valid location
void spawn_food(AntsEnv* env) {
    int attempts = 0;

    while (attempts < 100) {
        float x = random_float(50, env->width - 50);
        float y = random_float(50, env->height - 50);

        // Check distance from colonies
        bool valid = true;
        for (int j = 0; j < NUM_COLONIES; j++) {
            float dist_sq = distance_squared((Vector2D){x, y}, env->colonies[j].position);
            if (dist_sq < MIN_FOOD_COLONY_DISTANCE * MIN_FOOD_COLONY_DISTANCE) {
                valid = false;
                break;
            }
        }

        if (valid) {
            // Find empty slot
            for (int i = 0; i < MAX_FOOD_SOURCES; i++) {
                if (env->food_sources[i].amount == 0) {
                    env->food_sources[i].position.x = x;
                    env->food_sources[i].position.y = y;
                    env->food_sources[i].amount = MAX_FOOD_PER_SOURCE;
                    return;
                }
            }
        }
        attempts++;
    }
}

// Initialize environment memory
void init(AntsEnv* env) {
    env->ants = (Ant*)calloc(env->num_ants, sizeof(Ant));
    env->tick = 0;
    env->client = NULL;

    // Initialize colonies
    env->colonies[0].position = (Vector2D){env->width / 4, env->height / 2};
    env->colonies[1].position = (Vector2D){3 * env->width / 4, env->height / 2};
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;

    // Initialize food sources
    env->num_food_sources = MAX_FOOD_SOURCES;
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
}

// Compute observations for all ants - FOLLOWING TARGET PATTERN
void compute_observations(AntsEnv* env) {
    int obs_idx = 0;

    for (int a = 0; a < env->num_ants; a++) {
        Ant* ant = &env->ants[a];
        Colony* colony = &env->colonies[ant->colony_id];

        // Find closest food source
        float closest_food_dist_sq = env->width * env->width + env->height * env->height;
        Vector2D closest_food_pos = {0, 0};
        bool found_food = false;

        for (int i = 0; i < env->num_food_sources; i++) {
            if (env->food_sources[i].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[i].position);
                if (dist_sq < closest_food_dist_sq) {
                    closest_food_dist_sq = dist_sq;
                    closest_food_pos = env->food_sources[i].position;
                    found_food = true;
                }
            }
        }

        // Observation: [colony_dx, colony_dy, food_dx, food_dy, has_food, heading]
        // Normalized to roughly -1 to 1 range
        env->observations[obs_idx++] = (colony->position.x - ant->position.x) / env->width;
        env->observations[obs_idx++] = (colony->position.y - ant->position.y) / env->height;

        if (found_food) {
            env->observations[obs_idx++] = (closest_food_pos.x - ant->position.x) / env->width;
            env->observations[obs_idx++] = (closest_food_pos.y - ant->position.y) / env->height;
        } else {
            env->observations[obs_idx++] = 0.0f;
            env->observations[obs_idx++] = 0.0f;
        }

        env->observations[obs_idx++] = ant->has_food ? 1.0f : 0.0f;
        env->observations[obs_idx++] = ant->direction / (2 * M_PI);
    }
}

// Required function: reset environment
void c_reset(AntsEnv* env) {
    env->tick = 0;
    env->log = (Log){0};

    // Reset colonies
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;

    // Initialize ants
    int ants_per_colony = env->num_ants / NUM_COLONIES;
    for (int i = 0; i < env->num_ants; i++) {
        env->ants[i].colony_id = i / ants_per_colony;
        if (env->ants[i].colony_id >= NUM_COLONIES) {
            env->ants[i].colony_id = NUM_COLONIES - 1;
        }
        spawn_ant(env, i);
    }

    // Clear and respawn food
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
    for (int i = 0; i < env->num_food_sources; i++) {
        spawn_food(env);
    }

    // Clear buffers
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));

    compute_observations(env);
}

// Update food collection and delivery - FOLLOWING TARGET update_goals PATTERN
void update_food_interactions(AntsEnv* env) {
    for (int a = 0; a < env->num_ants; a++) {
        Ant* ant = &env->ants[a];

        // Check for food pickup
        if (!ant->has_food) {
            for (int f = 0; f < env->num_food_sources; f++) {
                if (env->food_sources[f].amount > 0) {
                    float dist_sq = distance_squared(ant->position, env->food_sources[f].position);
                    if (dist_sq < (ANT_SIZE + FOOD_SIZE) * (ANT_SIZE + FOOD_SIZE)) {
                        ant->has_food = true;
                        env->food_sources[f].amount--;

                        // Respawn food if depleted
                        if (env->food_sources[f].amount <= 0) {
                            spawn_food(env);
                        }

                        // Simple reward
                        env->rewards[a] += env->reward_food_pickup;
                        env->log.episode_return += env->reward_food_pickup;
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

                // Reward and log update - LIKE TARGET
                env->rewards[a] += env->reward_delivery;
                env->log.perf += 1.0f;  // Performance metric (food delivered)
                env->log.score += 1.0f;  // Score (food delivered)
                env->log.episode_return += env->reward_delivery;
                env->log.episode_length += ant->steps_alive;
                env->log.n += 1;  // Episode count (number of deliveries)

                // Reset ant after delivery
                ant->steps_alive = 0;
            }
        }
    }
}

// Required function: step environment
void c_step(AntsEnv* env) {
    env->tick++;

    // Clear rewards and terminals
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));

    // Update all ants - SIMPLIFIED LIKE TARGET
    for (int i = 0; i < env->num_ants; i++) {
        Ant* ant = &env->ants[i];
        ant->steps_alive++;

        // Execute action
        int action = env->actions[i];
        switch (action) {
            case ACTION_TURN_LEFT:
                ant->direction -= TURN_ANGLE;
                ant->direction = wrap_angle(ant->direction);
                break;
            case ACTION_TURN_RIGHT:
                ant->direction += TURN_ANGLE;
                ant->direction = wrap_angle(ant->direction);
                break;
            case ACTION_MOVE_FORWARD:
                ant->position.x += ANT_SPEED * cos(ant->direction);
                ant->position.y += ANT_SPEED * sin(ant->direction);
                break;
            case ACTION_NOOP:
                // Do nothing
                break;
        }

        // Wrap around edges
        if (ant->position.x < 0) ant->position.x = env->width;
        if (ant->position.x > env->width) ant->position.x = 0;
        if (ant->position.y < 0) ant->position.y = env->height;
        if (ant->position.y > env->height) ant->position.y = 0;

        // Periodic reset like target.c (lines 158-161)
        if (ant->steps_alive % ANT_RESET_INTERVAL == 0) {
            spawn_ant(env, i);
            env->terminals[i] = 1;
        }
    }

    // Update food interactions
    update_food_interactions(env);

    // Compute new observations
    compute_observations(env);
}

// Required function: render (with lazy client initialization)
void c_render(AntsEnv* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Ants");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = 1;
        env->client->width = env->width;
        env->client->height = env->height;
    }

    // Standard exit key
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);

    // Draw colonies
    for (int i = 0; i < NUM_COLONIES; i++) {
        Color color = (i == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        DrawCircle(env->colonies[i].position.x, env->colonies[i].position.y,
                   COLONY_SIZE, color);
    }

    // Draw food
    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount > 0) {
            DrawCircle(env->food_sources[i].position.x, env->food_sources[i].position.y,
                      FOOD_SIZE, FOOD_COLOR);
            DrawText(TextFormat("%d", env->food_sources[i].amount),
                    env->food_sources[i].position.x - 5,
                    env->food_sources[i].position.y - 5, 10, RAYWHITE);
        }
    }

    // Draw ants
    for (int i = 0; i < env->num_ants; i++) {
        Ant* ant = &env->ants[i];
        Color color = (ant->colony_id == 0) ? COLONY1_COLOR : COLONY2_COLOR;

        // Change color if carrying food
        if (ant->has_food) {
            color = FOOD_COLOR;
        }

        DrawCircle(ant->position.x, ant->position.y, ANT_SIZE, color);

        // Direction indicator
        float dir_x = ant->position.x + (ANT_SIZE * 1.5f) * cos(ant->direction);
        float dir_y = ant->position.y + (ANT_SIZE * 1.5f) * sin(ant->direction);
        DrawLine(ant->position.x, ant->position.y, dir_x, dir_y, RAYWHITE);
    }

    // Draw UI
    DrawText(TextFormat("Colony 1: %d", env->colonies[0].food_collected),
             20, 20, 20, COLONY1_COLOR);
    DrawText(TextFormat("Colony 2: %d", env->colonies[1].food_collected),
             20, 50, 20, COLONY2_COLOR);
    DrawText(TextFormat("Tick: %d", env->tick), env->width - 120, 20, 20, RAYWHITE);

    EndDrawing();
}

// Required function: cleanup
void c_close(AntsEnv* env) {
    if (env->ants) {
        free(env->ants);
        env->ants = NULL;
    }
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}
