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

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

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
    int i;

    int width;
    int height;
    float score;
    int tick;

    float powder;
    float angle;
    float tx;
    float ty;

    float px;
    float py;
    float vx;
    float vy;
    float g;
    int projectile_active;

    float target_min_x;
    float target_max_x;
    float target_min_y;
    float target_max_y;

    float min_aim_angle;
    float max_aim_angle;
    float max_reward;
    float max_reward_dist;
    float max_score;

    float ftmp1;
    float ftmp2;
    float ftmp3;
    float ftmp4;

    int frameskip;
    int render;
    int continuous;

    int debug;
    unsigned int rng;
    int render_many;
    int method;

    // Math
    float inv_width;
    float inv_height;
    float inv_pi2;
} Artillery;

void c_close(Artillery* env) {
    //unload_track();
}

void free_allocated(Artillery* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Artillery* env) {
    env->log.episode_length += env->tick;
    if (env->log.episode_length > 0.01f) {
    }
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
}

float calculate_score(Artillery* env, float hit_x, float hit_y) {
    float dx = hit_x - env->tx;
    float dy = hit_y - env->ty;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist <= 10.0f) return 1.0f; // Direct hit
    if (dist >= env->max_reward_dist) return 0.0f;

    return 1.0f - (dist / env->max_reward_dist);
}

void compute_observations(Artillery* env) {
    env->observations[0] = env->powder;
    env->observations[1] = env->angle;
    env->observations[2] = env->tx;
    env->observations[3] = env->ty;
    env->observations[4] = env->score / 100.0f;
}

Client* make_client(Artillery* env) {
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

void get_random_start(Artillery* env) {
    env->tx = rand() % env->width;
    env->ty = rand() % env->height;
    if (env->tx < env->target_min_x) env->tx = env->target_min_x;
    if (env->tx > env->target_max_x) env->tx = env->target_max_x;
    if (env->ty < env->target_min_y) env->ty = env->target_min_y;
    if (env->ty > env->target_max_y) env->ty = env->target_max_y;

    env->angle = ((float)rand() / RAND_MAX) * (env->max_aim_angle - env->min_aim_angle) + env->min_aim_angle;
    env->powder = (float)rand() / RAND_MAX;
}

void reset_round(Artillery* env) {
    get_random_start(env);
    env->projectile_active = 0;
}

void c_reset(Artillery* env) {
    env->score = 0;
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
}

void c_render(Artillery* env) {
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

    if (env->render_many) {
        env->method = rand() % 3;
        get_random_start(env);
    }

    BeginDrawing();
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground((Color){135, 206, 235, 255});

    DrawCircle(env->tx, height - env->ty, 15.0f, RED);
    DrawCircleLines(env->tx, height - env->ty, 15.0f, BLUE);

    float barrel_length = 40.0f;
    float barrel_width = 8.0f;
    float barrel_x = 30.0f;
    float barrel_y = height - 30.0f;

    Vector2 barrel_start = {barrel_x, barrel_y};
    Vector2 barrel_end = {
        barrel_x + barrel_length * cosf(env->angle),
        barrel_y - barrel_length * sinf(env->angle)
    };

    DrawLineEx(barrel_start, barrel_end, barrel_width, DARKGRAY);
    DrawCircle(barrel_x, barrel_y, 12.0f, GRAY);

    if (env->projectile_active) {
        DrawCircle(env->px, height - env->py, 4.0f, BLACK);
    }

    EndDrawing();
}

void init(Artillery* env) {
    env->tick = 0;
    env->debug = 0;
    env->g = 9.8f;
    env->projectile_active = 0;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_pi2 = 1.0f / PI2;

    srand(env->rng + env->i);

    get_random_start(env);
}

void allocate(Artillery* env) {
    init(env);
    env->observations = (float*)calloc(5, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void fire_projectile(Artillery* env) {
    if (env->projectile_active) return;

    float velocity = env->powder * 300.0f + 50.0f; // Convert powder to velocity
    env->vx = velocity * cosf(env->angle);
    env->vy = velocity * sinf(env->angle);
    env->px = 30.0f;
    env->py = 30.0f;
    env->projectile_active = 1;
}

void step_frame(Artillery* env, float action) {
    float act = 0.0;

    if (action == FIRE) {
        act = -1.0;
        fire_projectile(env);
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
    if (env->continuous) {
        act = action;
    }

    if (env->projectile_active) {
        env->px = env->px + env->vx * 0.016f; // Assuming ~60fps
        env->py = env->py + env->vy * 0.016f;
        env->vy = env->vy - env->g * 0.016f;

        if (env->px < 0 || env->px > env->width || env->py < 0) {
            env->projectile_active = 0;
            env->rewards[0] = 0.0f;
            env->terminals[0] = 1;
        }

        float dx = env->px - env->tx;
        float dy = env->py - env->ty;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist <= 15.0f) {
            float score = calculate_score(env, env->px, env->py);
            env->score += score * 100.0f;
            env->rewards[0] = score;
            env->projectile_active = 0;
            env->terminals[0] = 1;
        }
    }

    if (env->px > env->tx + 20) { // Missed
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(Artillery* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }
    compute_observations(env);
}
