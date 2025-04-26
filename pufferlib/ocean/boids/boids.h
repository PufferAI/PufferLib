// TODO: create a c_free function to free allocations in the c_init, fixing memory leaks
#include <stdlib.h>
#include <stdbool.h>
#include "raylib.h"

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define MARGIN_TURN_FACTOR 0.2
#define VISUAL_RANGE 20
#define VISUAL_RANGE_SQUARED VISUAL_RANGE * VISUAL_RANGE
#define PROTECTED_RANGE 2
#define PROTECTED_RANGE_SQUARED PROTECTED_RANGE * PROTECTED_RANGE
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

float get_random_float(float min, float max) {
    return min + (float)(rand() % (int)(max - min + 1));
}

// TODO: remove num_boids initialization out from the c_init and into the struct initialization of Boids
void c_init(Boids* env, unsigned int num_boids) {
    env->observations = (Boid*)calloc(num_boids * num_boids, sizeof(Boid));
    env->actions = (Velocity*)calloc(num_boids, sizeof(Velocity));
    env->rewards = (float*)calloc(num_boids, sizeof(float));
    env->terminals = (unsigned char*)calloc(num_boids, sizeof(unsigned char));
    env->boids = (Boid*)calloc(num_boids, sizeof(Boid));
    env->num_boids = num_boids;
    for (unsigned int indx = 0; indx < num_boids; indx++) {
        env->boids[indx].x = get_random_float(LEFT_MARGIN, WIDTH - RIGHT_MARGIN);
        env->boids[indx].y = get_random_float(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN);
        env->boids[indx].velocity.x = 0;
        env->boids[indx].velocity.y = 0;
    }
}

void c_compute_observations(Boids* env) {
    unsigned int current_observations_start_indx = 0;
    for (unsigned int current_boid_indx = 0; current_boid_indx < env->num_boids; current_boid_indx++) {
        current_observations_start_indx = current_boid_indx * env->num_boids;
        for (unsigned int observed_indx = 0; observed_indx < current_boid_indx; observed_indx++) {
            env->observations[current_observations_start_indx + observed_indx] = env->boids[observed_indx];
        }
        for (unsigned int observed_indx = current_boid_indx + 1; observed_indx < env->num_boids; observed_indx++) {
            env->observations[current_observations_start_indx + observed_indx] = env->boids[observed_indx];
        }
    }
}

void c_reset(Boids* env) {
    c_compute_observations(env);
}

void c_step(Boids* env, Velocity* action) {
    Boid* current_boid;
    Boid observed_boid;
    float diff_x;
    float diff_y;
    float squared_distance;
    float reward;
    unsigned int visual_boids_num;
    Boid visual_avg_boid;

    for (unsigned int indx = 0; indx < env->num_boids; indx++) {
        // Apply action
        current_boid = &env->boids[indx];
        // TODO: change = to +=
        current_boid->velocity.x = action->x;
        current_boid->velocity.y = action->y;
        current_boid->x += current_boid->velocity.x;
        current_boid->y += current_boid->velocity.y;

        // Calculate rewards
        reward = 0, visual_boids_num = 0;
        visual_avg_boid.x = 0, visual_avg_boid.y = 0;
        visual_avg_boid.velocity.x = 0, visual_avg_boid.velocity.y = 0;
        for (unsigned int observed_indx = 0; observed_indx < env->num_boids; observed_indx++) {
            observed_boid = env->observations[observed_indx];
            diff_x = current_boid->x - observed_boid.x;
            diff_y = current_boid->y - observed_boid.y;
            squared_distance = diff_x*diff_x + diff_y*diff_y;
            if (squared_distance < PROTECTED_RANGE_SQUARED) {
                reward -= (PROTECTED_RANGE_SQUARED - squared_distance) * AVOID_FACTOR;
            } else if (squared_distance < VISUAL_RANGE_SQUARED) {
                visual_avg_boid.x += observed_boid.x;
                visual_avg_boid.y += observed_boid.y;
                visual_avg_boid.velocity.x += observed_boid.velocity.x;
                visual_avg_boid.velocity.y += observed_boid.velocity.y;
                visual_boids_num++;
            }
        }

        if (visual_boids_num > 0) {
            visual_avg_boid.x /= visual_boids_num;
            visual_avg_boid.y /= visual_boids_num;
            visual_avg_boid.velocity.x /= visual_boids_num;
            visual_avg_boid.velocity.y /= visual_boids_num;
            reward -= (visual_avg_boid.velocity.x - current_boid->velocity.x)*MATCHING_FACTOR;
            reward -= (visual_avg_boid.velocity.y - current_boid->velocity.y)*MATCHING_FACTOR;
            reward -= (visual_avg_boid.x - current_boid->x)*CENTERING_FACTOR;
            reward -= (visual_avg_boid.y - current_boid->y)*CENTERING_FACTOR;
        }

        if (current_boid->y < TOP_MARGIN) {
            reward -= MARGIN_TURN_FACTOR;
        } else if (current_boid->y > HEIGHT - BOTTOM_MARGIN) {
            reward -= MARGIN_TURN_FACTOR;
        }

        if (current_boid->x < LEFT_MARGIN) {
            reward -= MARGIN_TURN_FACTOR;
        } else if (current_boid->x > WIDTH - RIGHT_MARGIN) {
            reward -= MARGIN_TURN_FACTOR;
        }

        env->rewards[indx] = reward;
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
