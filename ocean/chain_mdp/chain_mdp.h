#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

const unsigned char LEFT = 0;
const unsigned char RIGHT = 1;

const float STATE1_REWARD = 0.001;
const float STATEN_REWARD = 1.0;

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
    int size;
    int tick;
    unsigned char state; 

    Texture2D puffer; 

} Chain;

Chain* allocate_chain(Chain *env) {
    env->observations = calloc(1, sizeof(unsigned char));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(unsigned char));
    return env;
}

void free_allocated(Chain* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

void add_log(Chain* env) {
    env->log.perf += (env->rewards[0] == STATEN_REWARD) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Chain* env) {
    env->observations[0] = 1;
    env->state = 1;
    env->tick = 0;
}

// Required function
void c_step(Chain* env) {
    env->tick += 1;

    // Clear previous buffers
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    int action = env->actions[0];
    action = action*2 -1; // Map 0,1 to -1,1
    env->state = MIN(MAX(env->state + action, 0), env->size -1);
    env->observations[0] = env->state;

    // States 0 and N are absorbing and rewarding
    if (env->state == 0) {
        env->rewards[0] = STATE1_REWARD;
    } else if (env->state == env->size -1) {
        env->rewards[0] = STATEN_REWARD;
    }

    // Episode is over at N+9 steps
    if (env->tick == env->size + 9) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
}

// Required function. Should handle creating the client on first call
void c_render(Chain* env) {
    int px = MAX(8, 1024.0/env->size);

    if (!IsWindowReady()) {
        InitWindow(px*env->size, px*5, "PufferLib Chain MDP");
        SetTargetFPS(4);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int agent_pos = env->observations[0];
    for (int i = 0; i < env->size; i++) {
        Color color = 
            (i == agent_pos) ? (Color){0, 255, 255, 255} :
            (i == 0) ? (Color){204, 204, 0, 255} : 
            (i == env->size-1) ? (Color){255, 255, 51, 255} :
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

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Chain* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
