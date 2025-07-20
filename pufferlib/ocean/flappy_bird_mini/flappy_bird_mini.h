/* FlappyBirdMini: a simple 2-block tall grid environment.
 * Agent can move up or down. Observes an 8x2 grid with moving obstacles.
 * +1 reward per obstacle passed.
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// Actions
const unsigned char NOOP = 0;
const unsigned char UP = 1;
const unsigned char DOWN = 2;

// Observations
const unsigned char EMPTY = 0;
const unsigned char WALL = 1;

// Game constants
#define MAX_STEPS 1000
#define MAX_OBSTACLES 8
#define OBSTACLE_SPEED 1
#define OBSTACLE_SPAWN_INTERVAL 15
#define GRID_WIDTH 8
#define GRID_HEIGHT 2
#define AGENT_X 2


typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int x;
    int y;
    int active;
} Obstacle;


typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    int agent_pos; // 0 for floor, 1 for ceiling
    int wall_pos; // 0 for floor, 1 for ceiling
    Obstacle obstacles[MAX_OBSTACLES];
    int tick;
    int score;
    int spawn_timer;
} FlappyBirdMini;

void add_log(FlappyBirdMini* env);
void c_reset(FlappyBirdMini* env);
void c_step(FlappyBirdMini* env);
void c_render(FlappyBirdMini* env);
void c_close(FlappyBirdMini* env);
void spawn_obstacle(FlappyBirdMini* env);
void move_obstacles(FlappyBirdMini* env);
void update_observations(FlappyBirdMini* env);


void add_log(FlappyBirdMini* env) {
    float max_possible_walls = 1000.0f / OBSTACLE_SPAWN_INTERVAL;
    env->log.perf = (float)env->score / max_possible_walls; 
    env->log.score = env->score;
    env->log.episode_return = env->score;
    env->log.episode_length = env->tick;
    env->log.n++;
}

void spawn_obstacle(FlappyBirdMini* env) {
    // Find an inactive obstacle slot
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (!env->obstacles[i].active) {
            env->obstacles[i].x = GRID_WIDTH - 1; // Spawn at right edge
            env->obstacles[i].y = rand() % GRID_HEIGHT; // Random position (floor or ceiling)
            env->obstacles[i].active = 1;
            break;
        }
    }
}

void move_obstacles(FlappyBirdMini* env) {
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (env->obstacles[i].active) {
            env->obstacles[i].x -= OBSTACLE_SPEED;
            
            // Check if obstacle has passed the agent
            if (env->obstacles[i].x == AGENT_X - 1) {
                env->score++;
                env->rewards[0] += 1.0f;
            }
            
            // Deactivate obstacle if it goes off screen
            if (env->obstacles[i].x < 0) {
                env->obstacles[i].active = 0;
            }
        }
    }
}

void update_observations(FlappyBirdMini* env) {
    // Clear the entire 8x2 grid (16 observations)
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        env->observations[i] = EMPTY;
    }
    
    // Add all active obstacles to observations
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (env->obstacles[i].active && env->obstacles[i].x >= 0 && env->obstacles[i].x < GRID_WIDTH) {
            int obs_idx = env->obstacles[i].y * GRID_WIDTH + env->obstacles[i].x;
            env->observations[obs_idx] = WALL;
        }
    }
}

void c_reset(FlappyBirdMini* env) {
    env->agent_pos = 0; // Start on the floor
    env->tick = 0;
    env->score = 0;
    env->spawn_timer = 0;
    env->rewards[0] = 0.0f;
    
    // Clear obstacles
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        env->obstacles[i].active = 0;
        env->obstacles[i].x = 0;
        env->obstacles[i].y = 0;
    }

    update_observations(env);
}

void c_step(FlappyBirdMini* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;

    if (action == UP) {
        env->agent_pos = 1;
    } else if (action == DOWN) {
        env->agent_pos = 0;
    }

    move_obstacles(env);

    // Check for collision with obstacles at agent position
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (env->obstacles[i].active && env->obstacles[i].x == AGENT_X && env->obstacles[i].y == env->agent_pos) {
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            return;
        }
    }

    // Spawn new obstacles
    env->spawn_timer++;
    if (env->spawn_timer >= OBSTACLE_SPAWN_INTERVAL) {
        spawn_obstacle(env);
        env->spawn_timer = 0;
    }

    update_observations(env);

    if (env->tick >= MAX_STEPS) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
}

void c_render(FlappyBirdMini* env) {
    if (!IsWindowReady()) {
        InitWindow(64*GRID_WIDTH, 64*GRID_HEIGHT, "PufferLib Flappy Bird Mini");
        SetTargetFPS(10);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int cell_size = 64;

    // Draw background grid
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            DrawRectangleLines(x * cell_size, y * cell_size, cell_size, cell_size, 
                              (Color){50, 50, 50, 255});
        }
    }

    // Draw agent
    Color agent_color = (Color){0, 187, 187, 255};
    DrawRectangle(AGENT_X * cell_size + 8, env->agent_pos * cell_size + 8, 
                  cell_size - 16, cell_size - 16, agent_color);

    // Draw obstacles
    Color obstacle_color = (Color){187, 0, 0, 255};
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        if (env->obstacles[i].active) {
            DrawRectangle(env->obstacles[i].x * cell_size + 4, 
                         env->obstacles[i].y * cell_size + 4,
                         cell_size - 8, cell_size - 8, obstacle_color);
        }
    }

    // Draw score
    char score_text[64];
    sprintf(score_text, "Score: %d", env->score);
    DrawText(score_text, 10, 10, 20, WHITE);

    EndDrawing();
}

void c_close(FlappyBirdMini* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
