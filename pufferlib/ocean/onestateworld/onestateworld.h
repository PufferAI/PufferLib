#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "raylib.h"

// Marsaglia polar method from https://en.wikipedia.org/wiki/Marsaglia_polar_method
double gaussian_sample(double mean, double variance) {
    static int hasSpare = 0;
    static double spare;
    
    if (hasSpare) {
        hasSpare = 0;
        return mean + sqrt(variance) * spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + sqrt(variance) * (u * s);
}

const unsigned char LEFT = 0;
const unsigned char RIGHT = 1;

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
    int tick;

    float var_right;
    float mean_right;
    float mean_left;

    Texture2D puffer; 

} World;

World* allocate_World(World *env) {
    env->observations = calloc(1, sizeof(unsigned char));
    env->actions = calloc(1, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(unsigned char));
    return env;
}

void free_allocated(World* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

void add_log(World* env) {
    env->log.perf += env->rewards[0];
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(World* env) {
    env->observations[0] = 0;
    env->tick = 0;
    srand(time(NULL)); 
}

// Required function
void c_step(World* env) {
    env->tick += 1;

    // Clear previous buffers
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    int action = env->actions[0];

    // Tanh here because of pufferlib clamping 
    if (action == LEFT) {
        env->rewards[0] = tanh(gaussian_sample(env->mean_left, 0));
    } else {
        env->rewards[0] = tanh(gaussian_sample(env->mean_right, env->var_right));
    }

    if (env->tick >= 1000) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

// Required function. Should handle creating the client on first call
void c_render(World* env) {
    int px = 64;

    if (!IsWindowReady()) {
        InitWindow(px*5, px*5, "PufferLib OneStateWorld");
        SetTargetFPS(1);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw the puffer for fun 
    Color color = (Color){255, 255, 255, 255};
    Rectangle source_rect = (Rectangle){0, 0, 128, 128};
    Rectangle dest_rect = (Rectangle){2*px, 2*px, px, px};        
    DrawTexturePro(env->puffer, source_rect, dest_rect,
                (Vector2){0, 0}, 0, color);

    // Print the action taken either on left or right, with the reward received
    if (env->actions[0] == LEFT) {
            char score_text[32];
            snprintf(score_text, sizeof(score_text), "R: %.4f", env->rewards[0]);
            DrawText(score_text, 0, 2.5*px, 28, (Color){255, 255, 255, 255});
    } else {
            char score_text[32];
            snprintf(score_text, sizeof(score_text), "R: %.4f", env->rewards[0]);
            DrawText(score_text, 3*px, 2.5*px, 28, (Color){255, 255, 255, 255});
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(World* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
