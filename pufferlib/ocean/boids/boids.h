#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "raylib.h"

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define VELOCITY_CAP 3
#define MARGIN_TURN_FACTOR 0.2
#define VISUAL_RANGE 20
#define VISUAL_RANGE_SQUARED VISUAL_RANGE * VISUAL_RANGE
#define PROTECTED_RANGE 2
#define PROTECTED_RANGE_SQUARED PROTECTED_RANGE * PROTECTED_RANGE
#define CENTERING_FACTOR 0.0005
#define AVOID_FACTOR 0.05
#define MATCHING_FACTOR 0.05
#define MAX_AVOID_DISTANCE_SQAURED PROTECTED_RANGE_SQUARED * AVOID_FACTOR
#define MAX_AVG_POSITION_SQAURED VISUAL_RANGE_SQUARED * CENTERING_FACTOR
#define MAX_AVG_VELOCITY_SQUARED VELOCITY_CAP * 4 * MATCHING_FACTOR
#define WIDTH 800
#define HEIGHT 600
#define BOID_WIDTH 32
#define BOID_HEIGHT 32

typedef struct {
    float x;
    float y;
} Velocity;

typedef struct {
    float x;
    float y;
    Velocity velocity;
} Boid;

typedef struct {
    // an array of shape (num_boids, 4) with the 4 values correspoinding to (x, y, velocity x, velocity y)
    float* observations;
    // an array of shape (num_boids, 2) with the 2 values correspoinding to (velocity x, velocity y)
    float* actions;
    // an array of shape (num_boids, 1) with the reward for each boid
    float* rewards;
    unsigned char* terminals;
    Boid* boids;
    unsigned int num_boids;
    int max_reward;
    int min_reward;
} Boids;

typedef struct {
    float boid_width;
    float boid_height;
    Texture2D boid_texture;
} Client;

float flmax(float num1, float num2) {
    return (num1 > num2) ? num1 : num2;
}

float flmin(float num1, float num2) {
    return (num1 > num2) ? num2 : num1;
}

float flclip(float num, float min, float max) {
    return flmin(max, flmax(min, num));
}

float random_float(float min, float max) {
    return min + (float)(rand() % (int)(max - min + 1));
}

void c_init(Boids* env) {
    // Dynamic allocs
    env->observations = (float*)calloc(env->num_boids * env->num_boids, sizeof(float));
    env->actions = (float*)calloc(2, sizeof(float));
    env->rewards = (float*)calloc(env->num_boids, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_boids, sizeof(unsigned char));
    env->boids = (Boid*)calloc(env->num_boids, sizeof(Boid));

    // Initialization
    for (unsigned int indx = 0; indx < env->num_boids; indx++) {
        env->boids[indx].x = random_float(LEFT_MARGIN, WIDTH - RIGHT_MARGIN);
        env->boids[indx].y = random_float(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN);
        env->boids[indx].velocity.x = 0;
        env->boids[indx].velocity.y = 0;
    }

    // Claculate max and min rewards
    env->max_reward = 0;
    env->min_reward = -1
        * flmax(MAX_AVOID_DISTANCE_SQAURED * env->num_boids, MAX_AVG_POSITION_SQAURED)
        - 2*MARGIN_TURN_FACTOR;
}

void c_free(Boids* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->boids);
}

void c_compute_observations(Boids* env) {
    unsigned int current_start_indx = 0;
    for (unsigned int current_boid_indx = 0; current_boid_indx < env->num_boids; current_boid_indx++) {
        current_start_indx = current_boid_indx * env->num_boids;
        for (unsigned int observed_indx = 0; observed_indx < env->num_boids; observed_indx++) {
            env->observations[current_start_indx + observed_indx] = env->boids[observed_indx].x;
            env->observations[current_start_indx + observed_indx + 1] = env->boids[observed_indx].y;
            env->observations[current_start_indx + observed_indx + 2] = env->boids[observed_indx].velocity.x;
            env->observations[current_start_indx + observed_indx + 3] = env->boids[observed_indx].velocity.y;
        }
    }
}

void c_reset(Boids* env) {
    c_compute_observations(env);
}

void c_step(Boids* env) {
    Boid* current_boid;
    Boid observed_boid;
    float diff_x;
    float diff_y;
    float distance_squared;
    float reward;
    unsigned int visual_boids_num;
    Boid visual_avg_boid;

    printf("actions: %p\n", env->actions);
    printf("action: %f, %f\n", env->actions[0], env->actions[1]);
    return;
    // for (unsigned int indx = 0; indx < env->num_boids; indx++) {
    //     // Apply action
    //     current_boid = &env->boids[indx];
    //     current_boid->velocity.x += flclip(action->x, -VELOCITY_CAP, VELOCITY_CAP);
    //     current_boid->velocity.y += flclip(action->y, -VELOCITY_CAP, VELOCITY_CAP);
    //     current_boid->x = flclip(current_boid->x + current_boid->velocity.x, 0, WIDTH - BOID_WIDTH);
    //     current_boid->y = flclip(current_boid->y + current_boid->velocity.y, 0, HEIGHT - BOID_HEIGHT);
    //
    //     // Calculate rewards
    //     reward = 0, visual_boids_num = 0;
    //     visual_avg_boid.x = 0, visual_avg_boid.y = 0;
    //     visual_avg_boid.velocity.x = 0, visual_avg_boid.velocity.y = 0;
    //     for (unsigned int observed_indx = 0; observed_indx < env->num_boids; observed_indx++) {
    //         observed_boid = env->observations[observed_indx];
    //         diff_x = current_boid->x - observed_boid.x;
    //         diff_y = current_boid->y - observed_boid.y;
    //         distance_squared = diff_x*diff_x + diff_y*diff_y;
    //         if (distance_squared < PROTECTED_RANGE_SQUARED) {
    //             // seperation/avoidance reward
    //             reward -= (PROTECTED_RANGE_SQUARED - distance_squared) * AVOID_FACTOR;
    //         } else if (distance_squared < VISUAL_RANGE_SQUARED) {
    //             visual_avg_boid.x += observed_boid.x;
    //             visual_avg_boid.y += observed_boid.y;
    //             visual_avg_boid.velocity.x += observed_boid.velocity.x;
    //             visual_avg_boid.velocity.y += observed_boid.velocity.y;
    //             visual_boids_num++;
    //         }
    //     }
    //
    //     if (visual_boids_num > 0) {
    //         visual_avg_boid.x /= visual_boids_num;
    //         visual_avg_boid.y /= visual_boids_num;
    //         visual_avg_boid.velocity.x /= visual_boids_num;
    //         visual_avg_boid.velocity.y /= visual_boids_num;
    //
    //         // alignement and cohesion rewards
    //         reward -= (visual_avg_boid.velocity.x - current_boid->velocity.x)*MATCHING_FACTOR;
    //         reward -= (visual_avg_boid.velocity.y - current_boid->velocity.y)*MATCHING_FACTOR;
    //         reward -= (visual_avg_boid.x - current_boid->x)*CENTERING_FACTOR;
    //         reward -= (visual_avg_boid.y - current_boid->y)*CENTERING_FACTOR;
    //     }
    //
    //     // Margin rewards
    //     if (current_boid->y < TOP_MARGIN) {
    //         reward -= MARGIN_TURN_FACTOR;
    //     } else if (current_boid->y > HEIGHT - BOTTOM_MARGIN) {
    //         reward -= MARGIN_TURN_FACTOR;
    //     }
    //
    //     if (current_boid->x < LEFT_MARGIN) {
    //         reward -= MARGIN_TURN_FACTOR;
    //     } else if (current_boid->x > WIDTH - RIGHT_MARGIN) {
    //         reward -= MARGIN_TURN_FACTOR;
    //     }
    //
    //     // min-max normalizing reward
    //     env->rewards[indx] = 2
    //         * ((reward-env->min_reward) / (env->max_reward - env->min_reward))
    //         - 1;
    // }
}

Client* c_make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);
    client->boid_texture = LoadTexture("resources/puffers_128.png");
    client->boid_width = BOID_WIDTH;
    client->boid_height = BOID_HEIGHT;

    return client;
}

void c_close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Client* client, Boids* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (unsigned int indx = 0; indx < env->num_boids; indx++) {
        DrawTexturePro(
            client->boid_texture,
            (Rectangle){
                (env->boids[indx].velocity.x > 0) ? 0 : 128,
                0,
                128,
                128,
            },
            (Rectangle){
                env->boids[indx].x,
                env->boids[indx].y,
                client->boid_width,
                client->boid_height
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    EndDrawing();
}
