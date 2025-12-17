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
#define TURN_ANGLE (M_PI / 12)
#define MIN_FOOD_COLONY_DISTANCE 50.0f
#define ANT_RESET_INTERVAL 2048  // Reset ant every N steps (like target.c)

// Pheromone system constants
#define MAX_PHEROMONES 5000
#define PHEROMONE_DEPOSIT_AMOUNT 1.0f
#define PHEROMONE_EVAPORATION_RATE 0.001f
#define PHEROMONE_SIZE 2
#define PHEROMONE_DROP_INTERVAL 5  // Drop pheromone every N steps while carrying food

// Vision system constants
#define ANT_VISION_RANGE 100.0f
#define ANT_VISION_ANGLE (M_PI / 6.0f)  // 30 degrees (π/6)

// Actions
#define ACTION_TURN_LEFT 0
#define ACTION_TURN_RIGHT 1
#define ACTION_MOVE_FORWARD 2
#define ACTION_NOOP 3

// Colors
#define COLONY1_COLOR (Color){220, 0, 0, 255}
#define COLONY2_COLOR (Color){0, 0, 220, 255}
#define PHEROMONE1_COLOR (Color){255, 200, 200, 100}
#define PHEROMONE2_COLOR (Color){200, 200, 255, 100}
#define FOOD_COLOR (Color){0, 200, 0, 255}
#define BACKGROUND_COLOR (Color){50, 50, 50, 255}

// Required Log struct for PufferLib
typedef struct {
    float perf;                     // Average steps per delivery (efficiency - lower is better)
    float score;                    // Food deliveries per 1000 steps (throughput)
    float episode_return;           // Cumulative rewards
    float episode_length;           // Total steps across all ants
    float avg_delivery_steps;       // Average steps taken per successful delivery
    float colony1_food;             // Food collected by colony 1
    float colony2_food;             // Food collected by colony 2
    float total_deliveries;         // Total successful food deliveries
    float successful_trips;         // Number of ants that successfully found food
    float total_resets;             // Total ant resets (successful + unsuccessful)
    float n;                        // Episode count - REQUIRED AS LAST FIELD
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

// Pheromone trail marker
typedef struct {
    Vector2D position;
    float strength;
    int colony_id;
} Pheromone;

// Individual ant agent
typedef struct {
    Vector2D position;
    float direction;
    int colony_id;
    bool has_food;
    int steps_alive;           // Track steps for periodic reset
    int steps_since_pheromone; // Track when to drop next pheromone
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
    bool show_vision_cones;  // Toggle for vision cone visualization
};

// Main environment struct - FOLLOWING TARGET PATTERN
struct AntsEnv {
    Log log;                   // Required: aggregated log for all agents
    Client* client;            // Rendering client
    Ant* ants;                 // Dynamic array of ants
    Colony colonies[NUM_COLONIES];
    FoodSource food_sources[MAX_FOOD_SOURCES];
    Pheromone pheromones[MAX_PHEROMONES];

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
    int num_pheromones;       // Active pheromones
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

// Check if target is within ant's vision cone
static inline bool is_in_vision(Vector2D ant_pos, float ant_dir, Vector2D target) {
    float dx = target.x - ant_pos.x;
    float dy = target.y - ant_pos.y;
    float dist_sq = dx * dx + dy * dy;

    // Check range
    if (dist_sq > ANT_VISION_RANGE * ANT_VISION_RANGE) {
        return false;
    }

    // Check angle
    float angle_to_target = atan2(dy, dx);
    float angle_diff = wrap_angle(angle_to_target - ant_dir);

    return fabs(angle_diff) <= ANT_VISION_ANGLE / 2.0f;
}

// Add pheromone to the environment
static inline void add_pheromone(AntsEnv* env, Vector2D position, int colony_id) {
    if (env->num_pheromones >= MAX_PHEROMONES) {
        // Replace oldest pheromone (circular buffer)
        for (int i = 0; i < env->num_pheromones - 1; i++) {
            env->pheromones[i] = env->pheromones[i + 1];
        }
        env->num_pheromones--;
    }

    env->pheromones[env->num_pheromones].position = position;
    env->pheromones[env->num_pheromones].strength = PHEROMONE_DEPOSIT_AMOUNT;
    env->pheromones[env->num_pheromones].colony_id = colony_id;
    env->num_pheromones++;
}

// Spawn a new ant at its colony
void spawn_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    Colony* colony = &env->colonies[ant->colony_id];

    ant->position = colony->position;
    ant->direction = wrap_angle((rand() % 8) * (M_PI / 4));
    ant->has_food = false;
    ant->steps_alive = 0;
    ant->steps_since_pheromone = 0;
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
    env->num_pheromones = 0;

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

// Compute observations for all ants - WITH VISION AND PHEROMONES
void compute_observations(AntsEnv* env) {
    int obs_idx = 0;

    for (int a = 0; a < env->num_ants; a++) {
        Ant* ant = &env->ants[a];
        Colony* colony = &env->colonies[ant->colony_id];

        // Find closest visible food source (with vision constraints)
        float closest_food_dist_sq = env->width * env->width + env->height * env->height;
        Vector2D closest_food_pos = {0, 0};
        bool found_food = false;

        for (int i = 0; i < env->num_food_sources; i++) {
            if (env->food_sources[i].amount > 0) {
                Vector2D food_pos = env->food_sources[i].position;
                if (is_in_vision(ant->position, ant->direction, food_pos)) {
                    float dist_sq = distance_squared(ant->position, food_pos);
                    if (dist_sq < closest_food_dist_sq) {
                        closest_food_dist_sq = dist_sq;
                        closest_food_pos = food_pos;
                        found_food = true;
                    }
                }
            }
        }

        // Find closest visible pheromone from own colony
        float closest_pheromone_dist_sq = env->width * env->width + env->height * env->height;
        Vector2D closest_pheromone_pos = {0, 0};
        bool found_pheromone = false;

        for (int i = 0; i < env->num_pheromones; i++) {
            if (env->pheromones[i].colony_id == ant->colony_id) {
                Vector2D pheromone_pos = env->pheromones[i].position;
                if (is_in_vision(ant->position, ant->direction, pheromone_pos)) {
                    float dist_sq = distance_squared(ant->position, pheromone_pos);
                    if (dist_sq < closest_pheromone_dist_sq) {
                        closest_pheromone_dist_sq = dist_sq;
                        closest_pheromone_pos = pheromone_pos;
                        found_pheromone = true;
                    }
                }
            }
        }

        // Observation: [colony_dx, colony_dy, food_dx, food_dy, pheromone_dx, pheromone_dy, has_food, heading]
        // 8 values total - normalized to roughly -1 to 1 range
        env->observations[obs_idx++] = (colony->position.x - ant->position.x) / env->width;
        env->observations[obs_idx++] = (colony->position.y - ant->position.y) / env->height;

        if (found_food) {
            env->observations[obs_idx++] = (closest_food_pos.x - ant->position.x) / env->width;
            env->observations[obs_idx++] = (closest_food_pos.y - ant->position.y) / env->height;
        } else {
            env->observations[obs_idx++] = 0.0f;
            env->observations[obs_idx++] = 0.0f;
        }

        if (found_pheromone) {
            env->observations[obs_idx++] = (closest_pheromone_pos.x - ant->position.x) / env->width;
            env->observations[obs_idx++] = (closest_pheromone_pos.y - ant->position.y) / env->height;
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
    env->num_pheromones = 0;

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

                        // Track successful trip (ant found food)
                        env->log.successful_trips += 1.0f;
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

                // Reward and log update - WITH EFFICIENCY METRICS
                env->rewards[a] += env->reward_delivery;
                env->log.episode_return += env->reward_delivery;
                env->log.episode_length += ant->steps_alive;
                env->log.total_deliveries += 1.0f;

                // Track per-colony performance
                if (ant->colony_id == 0) {
                    env->log.colony1_food += 1.0f;
                } else {
                    env->log.colony2_food += 1.0f;
                }

                // Update derived efficiency metrics
                env->log.avg_delivery_steps = env->log.episode_length / env->log.total_deliveries;

                // Performance: Average steps per delivery (lower is better)
                env->log.perf = env->log.avg_delivery_steps;

                // Score: Food deliveries per 1000 steps (higher is better)
                env->log.score = (env->log.total_deliveries * 1000.0f) / env->log.episode_length;

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

        // Automatic pheromone dropping when carrying food
        if (ant->has_food) {
            ant->steps_since_pheromone++;
            if (ant->steps_since_pheromone >= PHEROMONE_DROP_INTERVAL) {
                add_pheromone(env, ant->position, ant->colony_id);
                ant->steps_since_pheromone = 0;
            }
        }

        // Periodic reset like target.c (lines 158-161)
        if (ant->steps_alive % ANT_RESET_INTERVAL == 0) {
            spawn_ant(env, i);
            env->terminals[i] = 1;
            env->log.total_resets += 1.0f;
        }
    }

    // Update pheromone evaporation
    for (int i = 0; i < env->num_pheromones; i++) {
        env->pheromones[i].strength -= PHEROMONE_EVAPORATION_RATE;
        if (env->pheromones[i].strength <= 0) {
            // Remove evaporated pheromone (swap with last and shrink)
            env->pheromones[i] = env->pheromones[env->num_pheromones - 1];
            env->num_pheromones--;
            i--;  // Check this slot again
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
        env->client->show_vision_cones = true;  // Start with vision cones on
    }

    // Standard exit key
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    // Toggle vision cones with 'V' key
    if (IsKeyPressed(KEY_V)) {
        env->client->show_vision_cones = !env->client->show_vision_cones;
    }

    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);

    // Draw colonies
    for (int i = 0; i < NUM_COLONIES; i++) {
        Color color = (i == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        DrawCircle(env->colonies[i].position.x, env->colonies[i].position.y,
                   COLONY_SIZE, color);
    }

    // Draw pheromones (before other objects for layering)
    for (int i = 0; i < env->num_pheromones; i++) {
        Color pheromone_color = (env->pheromones[i].colony_id == 0) ? PHEROMONE1_COLOR : PHEROMONE2_COLOR;
        pheromone_color.a = (unsigned char)(100 * env->pheromones[i].strength);
        DrawCircle(env->pheromones[i].position.x, env->pheromones[i].position.y,
                   PHEROMONE_SIZE, pheromone_color);
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

    // Draw ants with optional vision cones
    for (int i = 0; i < env->num_ants; i++) {
        Ant* ant = &env->ants[i];
        Color ant_color = (ant->colony_id == 0) ? COLONY1_COLOR : COLONY2_COLOR;

        // Draw vision cone if enabled (semi-transparent)
        if (env->client->show_vision_cones) {
            Color vision_color = ant_color;
            vision_color.a = 30;  // Very transparent

            // Calculate vision cone arc
            float start_angle = (ant->direction - ANT_VISION_ANGLE / 2.0f) * 180.0f / M_PI;
            float end_angle = (ant->direction + ANT_VISION_ANGLE / 2.0f) * 180.0f / M_PI;

            DrawCircleSector(
                (Vector2){ant->position.x, ant->position.y},
                ANT_VISION_RANGE,
                start_angle,
                end_angle,
                32,  // segments for smooth arc
                vision_color
            );
        }

        // Change color if carrying food
        if (ant->has_food) {
            ant_color = FOOD_COLOR;
        }

        DrawCircle(ant->position.x, ant->position.y, ANT_SIZE, ant_color);

        // Direction indicator (pointing forward)
        float dir_x = ant->position.x + (ANT_SIZE * 1.5f) * cos(ant->direction);
        float dir_y = ant->position.y + (ANT_SIZE * 1.5f) * sin(ant->direction);
        DrawLine(ant->position.x, ant->position.y, dir_x, dir_y, RAYWHITE);
    }

    // Draw UI - Colony scores
    DrawText(TextFormat("Colony 1: %d (%.1f%%)",
                       env->colonies[0].food_collected,
                       env->log.total_deliveries > 0 ? (env->log.colony1_food / env->log.total_deliveries * 100.0f) : 0.0f),
             20, 20, 20, COLONY1_COLOR);
    DrawText(TextFormat("Colony 2: %d (%.1f%%)",
                       env->colonies[1].food_collected,
                       env->log.total_deliveries > 0 ? (env->log.colony2_food / env->log.total_deliveries * 100.0f) : 0.0f),
             20, 50, 20, COLONY2_COLOR);

    // Efficiency metrics
    DrawText(TextFormat("Efficiency: %.1f steps/food", env->log.avg_delivery_steps),
             20, 80, 18, YELLOW);
    DrawText(TextFormat("Throughput: %.2f food/1000 steps", env->log.score),
             20, 105, 18, YELLOW);

    // Success rate
    float success_rate = env->log.total_resets > 0
        ? (env->log.successful_trips / env->log.total_resets * 100.0f)
        : 0.0f;
    DrawText(TextFormat("Success Rate: %.1f%%", success_rate),
             20, 130, 18, GREEN);

    // Right side - System info
    DrawText(TextFormat("Tick: %d", env->tick), env->width - 120, 20, 20, RAYWHITE);
    DrawText(TextFormat("Pheromones: %d", env->num_pheromones), env->width - 180, 50, 20, RAYWHITE);
    DrawText(TextFormat("Deliveries: %.0f", env->log.total_deliveries), env->width - 180, 75, 18, RAYWHITE);

    // Controls help
    const char* vision_status = env->client->show_vision_cones ? "ON" : "OFF";
    DrawText(TextFormat("[V] Vision Cones: %s", vision_status), 20, env->height - 30, 16, RAYWHITE);
    DrawText("[ESC] Exit", 20, env->height - 50, 16, GRAY);

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
