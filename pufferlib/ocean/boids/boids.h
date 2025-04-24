#include <stdlib.h>
#include <stdbool.h>
#include "raylib.h"

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define TURN_FACTOR 0.2
#define VISUAL_RANGE 20
#define PROTECTED_RANGE 2
#define CENTERING_FACTOR 0.0005
#define AVOID_FACTOR 0.05
#define MATCHING_FACTOR 0.05
#define MAX_SPEED 3
#define MIN_SPEED 2
#define WIDTH 800
#define HEIGHT 600
#define NUM_BOIDS 1

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
    Boid observations[NUM_BOIDS][NUM_BOIDS];
    Velocity actions[NUM_BOIDS];
    float rewards[NUM_BOIDS];
    unsigned char* terminals;
    Boid boids[NUM_BOIDS];
} Boids;

typedef struct {
    float boid_width;
    float boid_height;
    Texture2D boid_texture;
} Client;

void c_init(Boids* env) {
    env->boids[0].x = WIDTH / 2;
    env->boids[0].y = HEIGHT / 2;
    env->boids[0].velocity.x = 0;
    env->boids[0].velocity.y = 0;
}

void c_compute_observations(Boids* env) {
    for (unsigned int current_indx = 0; current_indx < NUM_BOIDS; current_indx++) {
        for (unsigned int watched_indx = 0; watched_indx < current_indx; watched_indx++) {
            env->observations[current_indx][watched_indx] = env->boids[watched_indx];
        }
        for (unsigned int watched_indx = current_indx + 1; watched_indx < NUM_BOIDS; watched_indx++) {
            env->observations[current_indx][watched_indx] = env->boids[watched_indx];
        }

    }
}

void c_reset(Boids* env) {
    c_compute_observations(env);
}

void c_step(Boids* env, Velocity* action) {
    Boid* current_boid = &env->boids[0];
    current_boid->velocity.x = action->x;
    current_boid->velocity.y = action->y;
    current_boid->x += current_boid->velocity.x;
    current_boid->y += current_boid->velocity.y;
}

Client* c_make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);
    client->boid_texture = LoadTexture("resources/puffers_128.png");
    client->boid_width = 32;
    client->boid_height = 32;

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

    DrawTexturePro(
        client->boid_texture,
        (Rectangle){
            (env->boids[0].velocity.x > 0) ? 0 : 128,
            0,
            128,
            128,
        },
        (Rectangle){
            env->boids[0].x,
            env->boids[0].y,
            client->boid_width,
            client->boid_height
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    EndDrawing();
}
