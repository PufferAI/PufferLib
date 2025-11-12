#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <time.h>

#define WIDTH 1280
#define INVWIDTH 1.0f / WIDTH
#define HEIGHT 720
#define INVHEIGHT 1.0f / HEIGHT

#define FIRE 0
#define ADDPOWDER 1
#define RMPOWDER 2
#define AIMUP 3
#define AIMDOWN 4

#define MINX1 600
#define MAXX1 1230
#define MINY1 300
#define MAXY1 670
#define MINAIMANGLE 0.56f
#define MAXAIMANGLE 1.56f

#define VCOEFF 150.0f

#define MAX_PROJECTILE_TIME 60.0f
#define TIMESTEP 0.25f

typedef struct Log {
    float episode_return;
    float score;
    float scoreL;
    float scoreR;
    float distL;
    float distR;
    float episode_length;
    float max_reward_distn;
    float turn_penaltyn;
    float acc1000;
    float n;
} Log;

typedef struct Client {
    int render;
    int debug;
} Client;

typedef struct Gun {
    float powder;
    float angle;
    float px;
    float py;
    int projectile_active;
    float projectile_time;
    float v0;
    float vx0;
    float vy0;
    float x0;
    float y0;
    float tx;
    float ty;
    float dist;
    float score;
    int fired;
    float turn_penaltyn;
} Gun;

typedef struct ArtyMulti {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int i;

    float score;
    float scoreL;
    float scoreR;
    int tick;

    Gun gun[2];

    float g;

    float max_reward_dist;
    float max_reward_distn;
    float max_dist0;
    float dist_fade;
    float turn_penalty;
    float turn_penalty_delay;
    float turn_penalty_ramp;
    float miss_penalty;
    float out_bounds_penalty;

    int frameskip;
    int render;

    int debug;
    unsigned int rng;
    int runs;
} ArtyMulti;

void c_close(ArtyMulti* env) {
}

void add_log(ArtyMulti* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.scoreL += env->gun[0].score;
    env->log.scoreR += env->gun[1].score;
    env->log.distL += env->gun[0].dist;
    env->log.distR += env->gun[1].dist;
    env->log.max_reward_distn += env->max_reward_distn;
    env->log.turn_penaltyn += env->gun[0].turn_penaltyn + env->gun[1].turn_penaltyn;
    env->log.n += 1;
    env->log.acc1000 += env->gun[1].score * (1500.0f - (env->gun[0].dist + env->gun[1].dist));
}

void compute_observations(ArtyMulti* env) {
    if (env->debug > 0) printf("  Compute Observations\n");

    env->observations[0] = env->gun[0].powder;
    env->observations[1] = env->gun[0].angle - MINAIMANGLE;
    env->observations[2] = env->gun[0].tx * INVWIDTH;
    env->observations[3] = env->gun[0].ty * INVHEIGHT;

    env->observations[4] = env->gun[1].powder;
    env->observations[5] = env->gun[1].angle - MINAIMANGLE;
    env->observations[6] = env->gun[1].tx * INVWIDTH;
    env->observations[7] = env->gun[1].ty * INVHEIGHT;

    env->observations[8] = env->score;
    env->observations[9] = env->tick * 0.01;
}

void get_random_start(ArtyMulti* env) {
    if (env->debug > 0) printf("get_random_start\n");

    env->gun[0].tx = (rand() % (WIDTH/2 - 100)) + WIDTH/2 + 50;
    env->gun[0].ty = rand() % HEIGHT;
    if (env->gun[0].ty < MINY1) env->gun[0].ty = MINY1;
    if (env->gun[0].ty > MAXY1) env->gun[0].ty = MAXY1;
    env->gun[0].angle = 0.5f;
    env->gun[0].powder = 0.95f;
    env->gun[0].x0 = 30.0f;
    env->gun[0].y0 = 30.0f;

    env->gun[1].tx = (rand() % (WIDTH/2 - 100)) + 50;
    env->gun[1].ty = rand() % HEIGHT;
    if (env->gun[1].ty < MINY1) env->gun[1].ty = MINY1;
    if (env->gun[1].ty > MAXY1) env->gun[1].ty = MAXY1;
    env->gun[1].angle = 0.95f;
    env->gun[1].powder = 0.75f;
    env->gun[1].x0 = WIDTH - 30.0f;
    env->gun[1].y0 = 30.0f;
}

void reset_round(ArtyMulti* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0;
    get_random_start(env);
    env->tick = 0;
    env->runs += 1;
    env->score = 0;

    for (int i = 0; i < 2; i++) {
        env->gun[i].fired = 0;
        env->gun[i].projectile_active = 0;
        env->gun[i].projectile_time = 0.0f;
        env->gun[i].score = 0;
        env->gun[i].turn_penaltyn = 0;
    }

    env->max_reward_distn = env->max_dist0 - (int)(env->runs * env->dist_fade);
    if (env->max_reward_distn < env->max_reward_dist) env->max_reward_distn = env->max_reward_dist;
}

void c_reset(ArtyMulti* env) {
    compute_observations(env);
    reset_round(env);
}

void init(ArtyMulti* env) {
    env->runs = 0;
    env->tick = 0;
    env->g = 9.81f;

    for (int i = 0; i < 2; i++) {
        env->gun[i].projectile_active = 0;
        env->gun[i].projectile_time = 0.0f;
        env->gun[i].dist = WIDTH;
        env->gun[i].fired = 0;
    }

    srand(env->rng + env->i);
    get_random_start(env);
}

float get_turn_penalty(ArtyMulti* env) {
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

float calculate_parabola_closest_distance(ArtyMulti* env, int gun_idx) {
    Gun* gun = &env->gun[gun_idx];
    gun->v0 = gun->powder * VCOEFF;

    float angle_multiplier = (gun_idx == 0) ? 1.0f : -1.0f;
    gun->vx0 = gun->v0 * cosf(gun->angle) * angle_multiplier;
    gun->vy0 = gun->v0 * sinf(gun->angle);

    float tx = gun->tx;
    float ty = gun->ty;

    float min_dist2 = 99999999.0f;

    float mid_x = (gun->x0 + tx) * 0.5f;
    float mid_y = 0.0f;
    int found_mid = 0;

    for (float t = 0; t < MAX_PROJECTILE_TIME; t += TIMESTEP) {
        float x = gun->x0 + gun->vx0 * t;
        float y = gun->y0 + gun->vy0 * t - 0.5f * env->g * t * t;

        if (y < 0 || x < 0 || x > WIDTH) break;

        if (!found_mid && ((gun_idx == 0 && x >= mid_x) || (gun_idx == 1 && x <= mid_x))) {
            mid_y = y;
            found_mid = 1;
        }

        float dx = x - tx;
        float dy = y - ty;
        float dist2 = dx * dx + dy * dy;

        if (dist2 < min_dist2) {
            min_dist2 = dist2;
        }
    }
    gun->dist = sqrt(min_dist2);

    float traj_score = 0.0f;
    if (found_mid) {
        if (gun_idx == 0) {
            traj_score = (mid_y < ty) ? 1.0f : 0.5f;
        } else {
            traj_score = (mid_y > ty * 2.0) ? 1.0f : 0.25f;
        }
    }

    return traj_score;
}

void fire_projectile(ArtyMulti* env, int gun_idx) {
    Gun* gun = &env->gun[gun_idx];
    if (env->debug > 0) printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!FIRE GUN %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", gun_idx);
    float traj_score = calculate_parabola_closest_distance(env, gun_idx);
    float score;

    if (gun->dist >= env->max_reward_distn) {
        score = env->miss_penalty;
    } else {
        score = 1.0f - (gun->dist / env->max_reward_distn);
    }

    if (score > 0.0f) {
        score = score * traj_score;
    }

    if (env->debug > 0) printf("    env%d tick%d gun%d closest_dist = %.3f score=%.3f\n", env->i, env->tick, gun_idx, gun->dist, score);
    gun->score += score;
    env->score += score * 0.5f;
    if (gun_idx == 0) env->scoreL += score * 0.5f;
    if (gun_idx == 1) env->scoreR += score * 0.5f;
    env->rewards[0] += score * 0.5f;

    if (env->render) {
        gun->projectile_active = 1;
        gun->projectile_time = 0.0f;
        gun->px = gun->x0;
        gun->py = gun->y0;
    }
}

void step_frame(ArtyMulti* env, float action0, float action1) {
    if (env->debug > 0) printf("STEP env%d tick%d=========================\n", env->i, env->tick);
    if (env->debug > 0) printf("  Gun0 Action%.3f    Gun1 Action%.3f\n", action0, action1);

    float actions[2] = {action0, action1};

    for (int gun_idx = 0; gun_idx < 2; gun_idx++) {
        Gun* gun = &env->gun[gun_idx];
        float action = actions[gun_idx];

        if (!gun->projectile_active) {
            if (action == FIRE) {
                gun->fired = 1;
                if (gun->projectile_active == 0) {
                    fire_projectile(env, gun_idx);
                }
            } else if (action == ADDPOWDER) {
                if (gun->powder < 0.95) {
                    gun->powder += 0.05;
                } else {
                    env->score += env->out_bounds_penalty;
                }
            } else if (action == RMPOWDER) {
                if (gun->powder > 0.05) {
                    gun->powder -= 0.05;
                } else {
                    env->score += env->out_bounds_penalty;
                }
            } else if (action == AIMUP) {
                if (gun->angle < MAXAIMANGLE - 0.05) {
                    gun->angle += 0.05;
                } else {
                    env->score += env->out_bounds_penalty;
                }
            } else if (action == AIMDOWN) {
                if (gun->angle > MINAIMANGLE + 0.05) {
                    gun->angle -= 0.05;
                } else {
                    env->score += env->out_bounds_penalty;
                }
            }

            if (action != FIRE) {
                float turn_pen = get_turn_penalty(env);
                gun->turn_penaltyn = turn_pen;
                gun->score += turn_pen;
                env->score += turn_pen * 0.5f;
                if (gun_idx == 0) env->scoreL += turn_pen * 0.5f;
                if (gun_idx == 1) env->scoreR += turn_pen * 0.5f;
                env->rewards[0] += gun->turn_penaltyn * 0.5f;
            }
        } else { // Projectile Active
            gun->projectile_time += TIMESTEP;
            gun->px = gun->x0 + gun->vx0 * gun->projectile_time;
            gun->py = gun->y0 + gun->vy0 * gun->projectile_time - 0.5f * env->g * gun->projectile_time * gun->projectile_time;
        }
    }

    int both_fired = env->gun[0].fired && env->gun[1].fired;
    int projectiles_done = 1;

    if (env->render) {
        for (int i = 0; i < 2; i++) {
            Gun* gun = &env->gun[i];
            if (gun->projectile_active && gun->px > 0 && gun->px < WIDTH && gun->py > 0) {
                projectiles_done = 0;
            }
        }
    }

    if ((both_fired && (projectiles_done || !env->render)) || (env->score < -2.0f)) {
        if (env->debug > 0) printf("==================terminate=================\n\n\n\n\n\n\n\n\n\n");
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(ArtyMulti* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action0 = env->actions[0];
    float action1 = env->actions[1];

    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action0, action1);
    }
    compute_observations(env);
}

Client* make_client(ArtyMulti* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(WIDTH, HEIGHT, "PufferLib ArtyMulti Dual");
    SetTargetFPS(30);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(ArtyMulti* env) {
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

    DrawCircle(env->gun[0].tx, HEIGHT - env->gun[0].ty, 15, RED);
    DrawCircle(env->gun[1].tx, HEIGHT - env->gun[1].ty, 15, BLUE);

    for (int gun_idx = 0; gun_idx < 2; gun_idx++) {
        Gun* gun = &env->gun[gun_idx];

        float barrel_length = 40.0f;
        float barrel_width = 8.0f;
        float barrel_x = gun->x0;
        float barrel_y = HEIGHT - gun->y0;

        float angle_multiplier = (gun_idx == 0) ? 1.0f : -1.0f;

        Vector2 barrel_start = {barrel_x, barrel_y};
        Vector2 barrel_end = {
            barrel_x + barrel_length * cosf(gun->angle) * angle_multiplier,
            barrel_y - barrel_length * sinf(gun->angle)
        };

        Vector2 prev_point = {gun->x0, HEIGHT - gun->y0};
        int j = 0;
        for (float t = TIMESTEP; t < MAX_PROJECTILE_TIME; t += 0.5f) {


            float v0 = gun->powder * VCOEFF;
            float vx0 = v0 * cosf(gun->angle) * angle_multiplier;
            float vy0 = v0 * sinf(gun->angle);

            float x = gun->x0 + vx0 * t;
            float y = gun->y0 + vy0 * t - 0.5f * env->g * t * t;

            if (y < 0 || x < 0 || x > WIDTH) break;

            Vector2 current_point = {x, HEIGHT - y};
            Color line_color = (gun_idx == 0) ? ORANGE : SKYBLUE;
            if (j % 2 == 0) DrawLineV(prev_point, current_point, BLACK);
            prev_point = current_point;
            j += 1;
        }

        Color gun_color = (gun_idx == 0) ? DARKGRAY : DARKBLUE;
        DrawLineEx(barrel_start, barrel_end, barrel_width, gun_color);
        DrawCircle(barrel_x, barrel_y, 12.0f, gun_color);

        if (gun->projectile_active) {
            Color proj_color = (gun_idx == 0) ? ORANGE : BLUE;
            DrawCircle(gun->px, HEIGHT - gun->py, 4.0f, proj_color);
        }
    }

    DrawText(TextFormat("Score: %.3f", env->score), 10, 10, 20, BLACK);
    DrawText(TextFormat("Gun0: %.3f Gun1: %.3f", env->gun[0].score, env->gun[1].score), 10, 35, 20, BLACK);

    EndDrawing();
}
