/* Dinosaur: a single-agent env that mimics google's offline dinosaur game */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define PLAYER_HEIGHT 48
#define PLAYER_WIDTH 32
#define PLAYER_JUMP 10.0f
#define GRAVITY 0.5f

#define CACTUS_HEIGHT 24
#define CACTUS_WIDTH 24

#define BIRD_HEIGHT 24
#define BIRD_WIDTH 48
#define BIRD_Y 44

const unsigned char NOOP = 0;
const unsigned char JUMP = 1;
const unsigned char CROUCH = 2;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Texture2D dinosaur_up;
    Texture2D dinosaur_down;
    Texture2D cactus;
    Texture2D bird;
} Client;

typedef struct {
    float x;
    float y;
    float y_velocity;
    float jump_strength;
    int ticks;
    float width;
    float height;
    float x_offset;
} Agent;

enum ObstacleType {
    CACTUS,
    BIRD
};

typedef struct {
    float x;
    float y;
    float width;
    float height;
    enum ObstacleType type;
} Obstacle;

typedef struct {
    /* Mandatory */
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    /* Not customizable */
    Client* client;
    Agent* agent;
    Obstacle* obstacles;
    int num_obstacles;
    float floor_y;
    int speed;
    int spawn_rate;
    float gravity;
    int spawn_ticks;
    /* Customizable */
    int width;
    int height;
    int speed_init;
    int speed_max;
    int spawn_rate_min;
    int spawn_rate_max;
    int rate_increment_rate;
    int max_obstacles;
} Dinosaur;

Client* make_client(Dinosaur* env){
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(env->width, env->height, "Pufferlib Dinosaur");
    SetTargetFPS(60);

    client->cactus = LoadTexture("resources/dinosaur/cactus.png");
    client->bird = LoadTexture("resources/dinosaur/bird.png");
    client->dinosaur_up = LoadTexture("resources/dinosaur/dino.png");
    client->dinosaur_down = LoadTexture("resources/dinosaur/dino_down.png");
    return client;
}

void init(Dinosaur* env) {
    env->gravity = GRAVITY;
    env->floor_y = env->height/2.0f;
    env->spawn_rate = 1;
    env->spawn_ticks = 0;

    env->agent = calloc(1, sizeof(Agent));
    env->agent->x = 0.0f + 2.0f * PLAYER_WIDTH;
    env->agent->y = 0.0f;
    env->agent->jump_strength = PLAYER_JUMP;
    env->agent->width = PLAYER_WIDTH;
    env->agent->height = PLAYER_HEIGHT;
}

void compute_observations(Dinosaur* env) {
    int obs_idx = 0;
    env->observations[obs_idx++] = env->agent->y / (pow(env->agent->jump_strength, 2) / (2 * env->gravity));
    env->observations[obs_idx++] = env->agent->x/env->width;
    env->observations[obs_idx++] = (float) env->speed / 10.0f;
    env->observations[obs_idx++] = env->agent->ticks/100.0f;

    for(int o = 0; o < env->max_obstacles; o++){
        if (o < env->num_obstacles) {
            Obstacle* obstacle = &env->obstacles[o];
            env->observations[obs_idx++] = obstacle->x/env->width;
            env->observations[obs_idx++] = obstacle->y/(env->width / 2);
            env->observations[obs_idx++] = obstacle->type == CACTUS ? 0.2f : 0.8f;
        } else {
            env->observations[obs_idx++] = 1.0f;
            env->observations[obs_idx++] = -1.0f;
            env->observations[obs_idx++] = 0.0f;
        }
    }
}

void c_reset(Dinosaur* env){
    env->speed = env->speed_init;
    env->spawn_rate = 1;
    env->spawn_ticks = 0;

    env->agent->ticks = 0;
    env->agent->y_velocity = 0.0f;
    env->agent->y = 0.0f;

    env->num_obstacles = 0;
    if (env->obstacles != NULL) {
          free(env->obstacles);
          env->obstacles = NULL;
    }

    compute_observations(env);
}

void c_step(Dinosaur* env){
    env->agent->ticks += 1;
    env->spawn_ticks += 1;
    *env->rewards = 0.01f;
    *env->terminals = 0;

    // handle user input
    switch(env->actions[0]){
        case NOOP:
            env->agent->y_velocity = -env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT;
            env->agent->width = PLAYER_WIDTH;
            env->agent->x_offset = 0.0f;
            break;
        case CROUCH:
            env->agent->y_velocity = -env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT / 2.f;
            env->agent->width = PLAYER_WIDTH * 2.0f;
            env->agent->x_offset = PLAYER_WIDTH;
            break;
        case JUMP:
            if(env->agent->y == 0.0f) env->agent->y_velocity = env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT;
            env->agent->width = PLAYER_WIDTH;
            env->agent->x_offset = 0.0f;
            break;
    }

    // gravity
    env->agent->y_velocity -= env->gravity;
    env->agent->y += env->agent->y_velocity;
    if(env->agent->y <= 0){
        env->agent->y = 0;
        env->agent->y_velocity = 0;
    }

    float agent_x_max = env->agent->x + env->agent->x_offset;
    float agent_x_min = agent_x_max - env->agent->width;
    float agent_y_min = env->agent->y;
    float agent_y_max = agent_y_min + env->agent->height;
    for(int o = 0; o < env->num_obstacles; o++){
        // move obstacles
        Obstacle* obstacle = &env->obstacles[o];
        obstacle->x -= env->speed;

        // handle collisions
        float obstacle_x_max = obstacle->x;
        float obstacle_x_min = obstacle_x_max - obstacle->width;
        float obstacle_y_min = obstacle->y;
        float obstacle_y_max = obstacle_y_min + env->agent->height;
        if(
            ((agent_x_max <= obstacle_x_max && agent_x_max >= obstacle_x_min) ||
            (agent_x_min <= obstacle_x_max && agent_x_min >= obstacle_x_min)) &&
            ((agent_y_max <= obstacle_y_max && agent_y_max >= obstacle_y_min) ||
            (agent_y_min <= obstacle_y_max && agent_y_min >= obstacle_y_min))
        ){
            *env->rewards = -1.0f;
            *env->terminals = 1;
            env->log.episode_return += env->agent->ticks / 100.0f - 1.0f;
            env->log.episode_length += env->agent->ticks;
            env->log.score += env->agent->ticks / 100.0f - 1.0f;
            env->log.perf += env->agent->ticks / 100.0f - 1.0f;
            env->log.n += 1;
            c_reset(env);
            return;
        }

        // despawn obstacles
        if(obstacle->x < 0 - 10){
            for(int j = o; j < env->num_obstacles - 1; j++){
                env->obstacles[j] = env->obstacles[j+1];
            }
            env->num_obstacles--;
            env->obstacles = realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
            o--;
        }
    }

    // spawn new obstacles
    if(env->spawn_ticks % env->spawn_rate == 0){
        int spawn_num = rand() % 4 + 1;
        if(spawn_num < 4){
            while(spawn_num + env->num_obstacles >= env->max_obstacles) spawn_num = rand() % 3;
            for(int i  = 0; i < spawn_num; i++){
                env->num_obstacles++;
                env->obstacles = realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
                env->obstacles[env->num_obstacles-1] = (Obstacle) {
                    .x = env->width + i * (CACTUS_WIDTH + 10.0f),
                    .y = 0,
                    .width = CACTUS_WIDTH,
                    .height = CACTUS_HEIGHT,
                    .type = CACTUS
                };
            }
        } else if (env->num_obstacles <= env->max_obstacles){
            env->num_obstacles++;
            env->obstacles = realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
            env->obstacles[env->num_obstacles-1] = (Obstacle) {
                .x = env->width + BIRD_WIDTH + 10.0f,
                .y = BIRD_Y,
                .width = BIRD_WIDTH,
                .height = BIRD_HEIGHT,
                .type = BIRD
            };
        }
        env->spawn_rate = rand() % (env->spawn_rate_max - env->spawn_rate_min) + env->spawn_rate_min;
        env->spawn_rate = env->spawn_rate / ((float)env->speed / (float)env->speed_init);
        env->spawn_ticks = 0;
   }

    // increase speed
    if(env->agent->ticks > 0 && env->agent->ticks % env->rate_increment_rate == 0){
        if(env->speed <= env->speed_max) env->speed+=1;
    }

    compute_observations(env);
}

void c_render(Dinosaur* env){
    if(env->client == NULL) {
        env->client = make_client(env);
    }

    if(IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();

    ClearBackground((Color){255, 255, 255, 255});
    DrawRectangle(0, env->height/2.0, env->width, env->height, (Color){95, 87, 79, 255});

    for(int o = 0; o < env->num_obstacles; o++){
        Obstacle* obstacle = &env->obstacles[o];
        Texture2D tex;
        tex = obstacle->type == CACTUS ? env->client->cactus : env->client->bird;
        DrawTexturePro(
            tex,
            (Rectangle){0, 0, obstacle->width, obstacle->height},
            (Rectangle){
                obstacle->x - obstacle->width,
                env->floor_y - obstacle->height - obstacle->y,
                obstacle->width,
                obstacle->height,
            },
            (Vector2){0, 0},
            0.0,
            WHITE
        );
    }

    Texture2D tex;
    switch(env->actions[0]){
        case NOOP:
        case JUMP:
            tex = env->client->dinosaur_up;
            break;
        case CROUCH:
            tex = env->client->dinosaur_down;
            break;
    }
    DrawTexturePro(
        tex,
        (Rectangle){0, 0, env->agent->width, env->agent->height},
        (Rectangle){
            env->agent->x - env->agent->width + env->agent->x_offset,
            env->floor_y - env->agent->height - env->agent->y,
            env->agent->width,
            env->agent->height
        },
        (Vector2){0, 0},
        0.0f,
        WHITE
    );

    EndDrawing();
}

void c_close(Dinosaur* env){
    free(env->agent);
    free(env->obstacles);
    if(env->client != NULL){
        UnloadTexture(env->client->cactus);
        UnloadTexture(env->client->dinosaur_up);
        UnloadTexture(env->client->dinosaur_down);
        CloseWindow();
        free(env->client);
    }
}
