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
    Boid* observations;
    Velocity* actions;
    float* rewards;
    unsigned char* terminals;
    Boid* boids;
    unsigned int num_boids;
} Boids;

typedef struct {
    float boid_width;
    float boid_height;
    Texture2D boid_texture;
} Client;

void c_init(Boids* env, unsigned int num_boids) {
    printf("STARTING INIT\n");
    env->observations = (Boid*)calloc(num_boids * num_boids, sizeof(Boid));
    env->actions = (Velocity*)calloc(num_boids, sizeof(Velocity));
    env->rewards = (float*)calloc(num_boids, sizeof(float));
    env->terminals = (unsigned char*)calloc(num_boids, sizeof(unsigned char));
    env->boids = (Boid*)calloc(num_boids, sizeof(Boid));
    env->num_boids = num_boids;
    for (unsigned int indx = 0; indx < num_boids; indx++) {
        env->boids[indx].x = WIDTH / 2;
        env->boids[indx].y = HEIGHT / 2;
        env->boids[indx].velocity.x = 0;
        env->boids[indx].velocity.y = 0;
    }
    printf("ENDING INIT\n");
}

void c_compute_observations(Boids* env) {
    unsigned int current_observations_start_indx = 0;
    for (unsigned int current_boid_indx = 0; current_boid_indx < env->num_boids; current_boid_indx++) {
        current_observations_start_indx = current_boid_indx * env->num_boids;
        for (unsigned int observed_boid_indx = 0; observed_boid_indx < current_boid_indx; observed_boid_indx++) {
            env->observations[current_observations_start_indx + observed_boid_indx] = env->boids[observed_boid_indx];
        }
        for (unsigned int observed_boid_indx = current_boid_indx + 1; observed_boid_indx < env->num_boids; observed_boid_indx++) {
            env->observations[current_observations_start_indx + observed_boid_indx] = env->boids[observed_boid_indx];
        }
    }
}

void c_reset(Boids* env) {
    c_compute_observations(env);
}

void c_step(Boids* env, Velocity* action) {
    for (unsigned int indx = 0; indx < env->num_boids; indx++) {
        Boid* current_boid = &env->boids[indx]; // TODO: remove pointer, try using the struct directly
        current_boid->velocity.x = action->x;
        current_boid->velocity.y = action->y;
        current_boid->x += current_boid->velocity.x;
        current_boid->y += current_boid->velocity.y;
    }
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
