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

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float dist;
    float max_reward_distn;
    float turn_penaltyn;
    float acc1000;
    float n;
} Log;

typedef struct Client {
    float width;
    float height;
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
    float dist;

    int moving_target;
    int timed_shell;

    float powder;
    float powder0;
    float angle;
    float angle0;
    float tx;
    float ty;

    float px;
    float py;
    float vx;
    float vy;
    float g;
    int projectile_active;
    float projectile_time;

    float v0;
    float vx0;
    float vy0;
    float x0;
    float y0;

    float target_min_x;
    float target_max_x;
    float target_min_y;
    float target_max_y;
    float target_size;
    float target_vx;
    float target_vy;

    float min_aim_angle;
    float max_aim_angle;
    float max_reward;
    float max_reward_dist;
    float max_reward_distn;
    float max_dist0;
    float dist_fade;
    float turn_penalty;
    float turn_penaltyn;
    int turn_penalty_delay;
    float turn_penalty_ramp;
    float miss_penalty;
    float max_score;
    int fired;
    float vm;
    float out_bounds_penalty;

    float ftmp1;
    float ftmp2;
    float ftmp3;
    float ftmp4;

    int frameskip;
    int render;
    int continuous;

    int debug;
    unsigned int rng;
    int method;
    int same_runs;
    int runs;

    // Math
    float inv_width;
    float inv_height;
    float inv_max_angle;
    float inv_angle_range;
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
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.dist += env->dist;
    env->log.max_reward_distn += env->max_reward_distn;
    env->log.turn_penaltyn += env->turn_penaltyn;
    env->log.n += 1;
    env->log.acc1000 += 750.0f - env->dist;
}

void calculate_parabola_closest_distance(Artillery* env) {
    env->v0 = env->powder * env->vm;
    env->vx0 = env->v0 * cosf(env->angle);
    env->vy0 = env->v0 * sinf(env->angle);
    env->x0 = 30.0f;
    env->y0 = 30.0f;

    float tx = env->tx;
    float ty = env->ty;
    float tvx = env->target_vx;
    float tvy = env->target_vy;
    float txn = tx;
    float tyn = ty;

    float min_dist2 = 99999999.0f;

    for (float t = 0; t < 60.0f; t += 0.25f) {
        float x = env->x0 + env->vx0 * t;
        float y = env->y0 + env->vy0 * t - 0.5f * env->g * t * t;

        if (y < 0 || x < 0 || x > env->width) break;

        txn = tx + tvx * t;
        tyn = ty + tvy * t;

        float dx = x - txn;
        float dy = y - tyn;
        float dist2 = dx * dx + dy * dy;

        if (dist2 < min_dist2) {
            min_dist2 = dist2;
        }
    }
    env->dist = sqrt(min_dist2);
}

void fire_projectile(Artillery* env) {
    if (env->debug > 0) printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!FIRE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    calculate_parabola_closest_distance(env);
    float score;

    if (env->dist >= env->max_reward_distn) {
        score = env->miss_penalty;
    } else {
        score = 1.0f - (env->dist / env->max_reward_distn);
    }

    if (env->debug > 0) printf("    env%d tick%d closest_dist = %.3f score=%.3f\n", env->i, env->tick, env->dist, score);
    env->score += score;
    env->rewards[0] += score;

    if (env->render) {
        env->projectile_active = 1;
        env->projectile_time = 0.0f;
        env->px = env->x0;
        env->py = env->y0;
        env->vx = env->vx0;
        env->vy = env->vy0;
    }
}

void compute_observations(Artillery* env) {
    if (env->debug > 0) printf("  Compute Observations\n");
    env->observations[0] = env->powder;
    if (env->debug > 0) printf("    powder = %.3f\n", env->observations[0]);
    env->observations[1] = env->angle - env->min_aim_angle;
    if (env->debug > 0) printf("    angle = %.3f\n", env->observations[1]);
    env->observations[2] = env->tx * env->inv_width;
    if (env->debug > 0) printf("    tx = %.3f\n", env->observations[2]);
    env->observations[3] = env->ty * env->inv_height;
    if (env->debug > 0) printf("    ty = %.3f\n", env->observations[3]);
    env->observations[4] = env->score;
    if (env->debug > 0) printf("    score = %.6f\n", env->observations[4]);
    env->observations[5] = env->tick * 0.01;
    if (env->debug > 0) printf("    tick = %.3f\n", env->observations[5]);

    if (env->moving_target == 1) {
        env->observations[6] = env->target_vx * 0.01;
        if (env->debug > 0) printf("    target_vx = %.3f\n", env->observations[6]);
        env->observations[7] = env->target_vy * 0.01;
        if (env->debug > 0) printf("    target_vy = %.3f\n", env->observations[7]);
    }
}

Client* make_client(Artillery* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Artillery");
    SetTargetFPS(30);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void get_random_start(Artillery* env) {
    if (env->debug > 0) printf("get_random_start\n");
    env->tx = rand() % env->width;
    env->ty = rand() % env->height;
    if (env->tx < env->target_min_x) env->tx = env->target_min_x;
    if (env->tx > env->target_max_x) env->tx = env->target_max_x;
    if (env->ty < env->target_min_y) env->ty = env->target_min_y;
    if (env->ty > env->target_max_y) env->ty = env->target_max_y;
    env->angle = ((float)rand() / (float)RAND_MAX) + env->min_aim_angle;
    env->angle0 = env->angle;
    env->powder = (float)rand() / (float)RAND_MAX;
    env->powder0 = env->powder;
    env->target_vx = -rand() % 20 - 10;
    env->target_vy = -rand() % 10 - 5;
}

void reset_round(Artillery* env) {
    if (env->runs % (int)env->same_runs == 0) {
        get_random_start(env);
    }
    else {
        env->angle = env->angle0;
        env->powder = env->powder0;
    }
    env->tick = 0;
    env->runs += 1;
    env->score = 0;
    env->fired = 0;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->max_reward_distn = env->max_dist0 - (int)(env->runs * env->dist_fade);
    if (env->max_reward_distn < env->max_reward_dist) env->max_reward_distn = env->max_reward_dist;
}

void c_reset(Artillery* env) {
    compute_observations(env);
    reset_round(env);
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

    BeginDrawing();
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    ClearBackground((Color){135, 206, 235, 255});

    DrawCircle(env->tx, height - env->ty, env->target_size, RED);

    float barrel_length = 40.0f;
    float barrel_width = 8.0f;
    float barrel_x = 30.0f;
    float barrel_y = height - 30.0f;

    Vector2 barrel_start = {barrel_x, barrel_y};
    Vector2 barrel_end = {
        barrel_x + barrel_length * cosf(env->angle),
        barrel_y - barrel_length * sinf(env->angle)
    };

    float v0 = env->powder * env->vm;
    float vx0 = v0 * cosf(env->angle);
    float vy0 = v0 * sinf(env->angle);
    float x0 = 30.0f;
    float y0 = 30.0f;

    Vector2 prev_point = {x0, height - y0};
    int j = 0;
    for (float t = 0.25f; t < 60.0f; t += 0.5f) {
        float x = x0 + vx0 * t;
        float y = y0 + vy0 * t - 0.5f * env->g * t * t;

        if (y < 0 || x < 0 || x > env->width) break;

        Vector2 current_point = {x, height - y};
        if (j % 2 == 0) DrawLineV(prev_point, current_point, WHITE);
        if (j % 2 == 1) DrawLineV(prev_point, current_point, BLACK);
        prev_point = current_point;
        j += 1;
    }

    DrawLineEx(barrel_start, barrel_end, barrel_width, DARKGRAY);
    DrawCircle(barrel_x, barrel_y, 12.0f, GRAY);

    if (env->projectile_active) {
        DrawCircle(env->px, height - env->py, 4.0f, BLACK);
    }

    DrawText(TextFormat("%.3f", env->score), 10, 10, 20, BLACK);

    EndDrawing();
}

void init(Artillery* env) {
    env->runs = 0;
    env->tick = 0;
    if (env->same_runs < 1) env->same_runs = 1;
    env->g = 9.8f;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->dist = env->width;

    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_max_angle = 1.0f / env->max_aim_angle;
    env->inv_angle_range = 1.0f / (env->max_aim_angle - env->min_aim_angle);

    srand(env->rng + env->i);

    get_random_start(env);
}

void allocate(Artillery* env) {
    init(env);
    int obs_size = (env->moving_target == 1) ? 8 : 6;
    env->observations = (float*)calloc(obs_size, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

float get_turn_penalty(Artillery* env) {
    int start_tick = env->turn_penalty_delay;
    int end_tick = start_tick + env->turn_penalty_ramp;

    if (env->tick <= start_tick) {
        return 0.0f;
    } else if (env->tick < end_tick) {
        float progress = (env->tick - start_tick) * env->turn_penalty_ramp;
        return env->turn_penalty * progress;
    } else {
        return env->turn_penalty;
    }
}

void step_frame(Artillery* env, float action) {
    if (env->debug > 0) printf("STEP env%d tick%d=========================\n", env->i, env->tick);

    if (!env->projectile_active) {
        if (action == FIRE) {
            env->fired = 1;
            if (env->projectile_active == 0) {
                fire_projectile(env);
            }
        } else if (action == ADDPOWDER) {
            if (env->powder < 0.95) {
                env->powder += 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == RMPOWDER) {
            if (env->powder > 0.05) {
                env->powder -= 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == AIMUP) {
            if (env->angle < env->max_aim_angle - 0.05) {
                env->angle += 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == AIMDOWN) {
            if (env->angle > env->min_aim_angle + 0.05) {
                env->angle -= 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        }

        if (action != FIRE) {
            env->turn_penaltyn = get_turn_penalty(env);
            env->score += env->turn_penaltyn;
            env->rewards[0] += env->turn_penaltyn;
        }
    }
    else { // Projectile Active
        env->projectile_time += 0.25f;
        env->px = env->x0 + env->vx0 * env->projectile_time;
        if (env->debug > 1) printf("env->px = %.3f, env->vx0 = %.3f, ptime = %.3f\n", env->px, env->vx0, env->projectile_time);
        if (env->debug > 1) printf("env->tx = %.3f\n", env->tx);
        env->py = env->y0 + env->vy0 * env->projectile_time - 0.5f * env->g * env->projectile_time * env->projectile_time;
        if (env->debug > 1) printf("env->py = %.3f, env->vy0 = %.3f, ptime = %.3f\n", env->py, env->vy0, env->projectile_time);
    }

    if (env->moving_target == 1) {
        env->tx += env->target_vx * 0.25f;
        env->ty += env->target_vy * 0.25f;
    }

    if (env->debug > 1) printf("  env->px = %.1f env->tx = %.1f env->render=%d\n", env->px, env->tx, env->render);
    if ((env->fired == 1 && (!env->render || env->px > env->tx + env->target_size || env->py < 0.0f)) || (env->score < -1.0f)) {
        if (env->debug > 0) printf("==================terminate=================\n\n\n\n\n\n\n\n\n\n");
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
