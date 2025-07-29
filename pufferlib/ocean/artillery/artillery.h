#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <time.h>

#define FIRE 0
#define ADDPOWDER 1
#define RMPOWDER 2
#define AIMUP 3
#define AIMDOWN 4

#define PI2 PI * 2

typedef struct Client {
    float width;   // 640
    float height;  // 480
    int render;
    int debug;
} Client;

typedef struct Artillery {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    float score;
    int i;

    float powder;
    float angle;
    float tx;
    float ty;

    float target_min_x;
    float target_max_x;
    float target_min_y;
    float target_max_y;

    float min_aim_angle;
    float max_aim_angle;
    float max_reward;
    float max_reward_dist;

    int debug;
    unsigned int rng;
    int render_many;
    int method;
} Artillery;

void free_allocated(WhiskerRacer* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(WhiskerRacer* env) {
    env->log.episode_length += env->tick;
    if (env->log.episode_length > 0.01f) {
    }
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
}

void compute_observations(WhiskerRacer* env) {
    env->observations[0] = env->powder;
    env->observations[1] = env->angle;
    env->observations[2] = env->tx;
    env->observations[3] = env->ty;
    env->observations[4] = env->score / 100.0f;
}

Client* make_client(WhiskerRacer* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Artillery");
    if (env->render_many) SetTargetFPS(10 / env->frameskip);
    else SetTargetFPS(60 / env->frameskip);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void get_random_start(WhiskerRacer* env) {
    int env->tx = rand() % env->width;
    int env->ty = rand() % env->height;
    if (env->tx < env->target_min_x) env->tx = env->target_min_x;
    if (env->tx > env->target_max_x) env->tx = env->target_max_x;
    if (env->tx < env->target_min_y) env->ty = env->target_min_y;
    if (env->tx > env->target_max_y) env->ty = env->target_max_y;

    env->angle = rand();
    env->powder = rand();
}

void reset_round(WhiskerRacer* env) {
    get_random_start(env);
}

void c_reset(WhiskerRacer* env) {
    env->score = 0;
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
}

void c_render(WhiskerRacer* env) {

    int height = env->height;

    env->render = 1;
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    if (env->render_many)
    {
        env->method = rand() % 3;
        get_random_start(env);
    }

    BeginDrawing();
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground(LIGHTBLUE);

    float car_width = 24.0f;
    float car_height = 12.0f;
    float car_x = env->tx;
    float car_y = height - env->ty;
    Vector2 origin = {car_width / 2.0f, car_height / 2.0f};
    DrawRectanglePro(
        (Rectangle){car_x, car_y, car_width, car_height},
        origin,
        0 * 180.0f / PI,
        (Color){255, 0, 255, 255}
    );

    EndDrawing();
}

void init(WhiskerRacer* env) {
    env->tick = 0;

    env->debug = 0;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_pi2 = 1.0f / PI2;

    srand(env->rng + env->i);

    get_random_start(env);
}

void allocate(WhiskerRacer* env) {
    init(env);
    env->observations = (float*)calloc(5, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void step_frame(WhiskerRacer* env, float action) {
    float act = 0.0;

    /*#define FIRE 0
    #define ADDPOWDER 1
    #define RMPOWDER 2
    #define AIMUP 3
    #define AIMDOWN 4*/

    if (action == FIRE) {
        act = -1.0;
        // pew env->ang += PI / env->turn_pi_frac;
    } else if (action == ADDPOWDER) {
        act = -0.5;
        if (env->powder < 0.95) env->powder += 0.05;
    } else if (action == RMPOWDER) {
        act = 0.0;
        if (env->powder > 0.05) env->powder -= 0.05;
    } else if (action == AIMUP) {
        act = 0.5;
        if (env->angle < env->max_aim_angle - 0.05) env->angle += 0.05;
    } else if (action == AIMDOWN) {
        act = 1.0;
        if (env->angle > env->min_aim_angle + 0.05) env->angle -= 0.05;
    }
    if (env->continuous){
        act = action;
    }

/*
    env->vx = env->v * cosf(env->ang);
    env->vy = env->v * sinf(env->ang);
    env->px = env->px + env->vx;
    env->py = env->py + env->vy;
    if (env->px < 0) env->px = 0;
    else if (env->px > env->width) env->px = env->width;
    if (env->py < 0) env->py = 0;
    else if (env->py > env->height) env->py = env->height;

    calc_whisker_lengths(env);

    update_radial_progress(env);
*/
}

void c_step(WhiskerRacer* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }
    compute_observations(env);
}
