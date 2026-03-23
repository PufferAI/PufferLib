#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

const unsigned char FORWARD = 0;
const unsigned char RIGHT = 1;
const unsigned char LEFT = 2;

const unsigned char EMPTY = 1;
const unsigned char WALL = 0; 

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int size; // length of the corridor
    int tick;

    unsigned char state; // Internal current position in the maze
    unsigned char starting_state; // Starting state (2 or 3)

    Texture2D puffer; 

} TMaze;

TMaze* allocate_TMaze(TMaze *env) {
    env->observations = calloc(4, sizeof(unsigned char));
    env->actions = calloc(1, sizeof(int));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(unsigned char));
    return env;
}

void free_allocated(TMaze* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

void add_log(TMaze* env) {
    env->log.perf += (env->rewards[0]+1)/2; // Normalized to 0-1
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(TMaze* env) {
    memset(env->observations, WALL, 4*sizeof(unsigned char));
    env->starting_state = rand() % 2 + 2; // 2 or 3 
    // [current, front, left, right]
    env->observations[0] = env->starting_state;
    env->observations[1] = EMPTY;
    env->tick = 0;
    env->state = 0;
}

void compute_observations(TMaze* env) {
    // if at the end of the maze
    if (env->state == env->size -1) {
        env->observations[0] = EMPTY;
        env->observations[1] = WALL;
        env->observations[2] = EMPTY;
        env->observations[3] = EMPTY;
        return;
    }
    // We don't have noops so the agent can not go back to the start
    env->observations[0] = EMPTY;
    env->observations[1] = EMPTY;
    env->observations[2] = WALL;
    env->observations[3] = WALL;
}

// Required function
void c_step(TMaze* env) {
    env->tick += 1;

    // Clear previous buffers
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    int action = env->actions[0];

    if (env->state == env->size -1) {
        const int left_reward = (env->starting_state == 2) ? 1 : -1;
        const int right_reward = (env->starting_state == 3) ? 1 : -1;

        if (action == LEFT || action == RIGHT) {
            env->rewards[0] = (action == LEFT) ? left_reward : right_reward;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
        }

    } else {
        if (action == FORWARD) {
            env->state += 1;
            compute_observations(env);
        } 
    }
}

// Required function. Should handle creating the client on first call
void c_render(TMaze* env) {
    int px = MAX(8, 1024.0/env->size);

    if (!IsWindowReady()) {
        InitWindow(px*env->size, px*5, "PufferLib TMaze MDP");
        SetTargetFPS(4);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int agent_pos = env->state;
    for (int i = 0; i < env->size; i++) {
        Color color = 
            (i == agent_pos) ? (Color){0, 255, 255, 255} :
            (i == 0 && env->starting_state == 2) ? (Color){255, 0, 0, 255} : 
            (i == 0 && env->starting_state == 3) ? (Color){0, 255, 0, 255} : 
            (Color){224, 224, 224, 255};

        if (i == agent_pos) {
            int starting_sprite_x = 0;
            float rotation = env->actions[0];
            if (rotation == -1) {
            starting_sprite_x = 128;
            rotation = 0;
            }
            Rectangle source_rect = (Rectangle){starting_sprite_x, 0, 128, 128};
            Rectangle dest_rect = (Rectangle){i*px, 2*px, px, px};        
            DrawTexturePro(env->puffer, source_rect, dest_rect,
                        (Vector2){0, 0}, 0, color);
        } else {
            DrawRectangle(i*px, 2*px, px, px, color);
        }

    }
    // Draw last terminal states
    DrawRectangle((env->size-1)*px, 1*px, px, px, (Color){255, 0, 0, 255});
    DrawRectangle((env->size-1)*px, 3*px, px, px, (Color){0, 255, 0, 255});

    char score_text[32];
    snprintf(score_text, sizeof(score_text), "Score: %f", env->rewards[0]);
    DrawText(score_text, env->size * px - 180, 10, 32, (Color){255, 255, 255, 255});
    

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(TMaze* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
