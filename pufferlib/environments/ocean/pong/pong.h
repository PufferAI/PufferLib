#include <stdlib.h>
#include <math.h>
#include "raylib.h"

typedef struct CPong CPong;
struct CPong {
    float* observations;
    unsigned int* actions;
    float* rewards;
    unsigned char* terminals;
    float* paddle_yl_yr;
    float* ball_x_y;
    float* ball_vx_vy;
    unsigned int* score_l_r;
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float paddle_speed;
    float ball_initial_speed_x;
    float ball_initial_speed_y;
    float ball_max_speed_y;
    float ball_speed_y_increment;
    unsigned int max_score;
    float min_paddle_y;
    float max_paddle_y;
    float paddle_dir;
    unsigned int* misc_logging;
    int tick;
    int n_bounces;
    int win;
    int frameskip;
};

void init(CPong* env) {
    // logging
    env->tick = 0;
    env->n_bounces = 0;
    env->win = 0;

    // precompute
    env->min_paddle_y = -env->paddle_height / 2;
    env->max_paddle_y = env->height - env->paddle_height/2;
    
    env->paddle_dir = 0;
}

void allocate(CPong* env) {
    init(env);
    env->observations = (float*)calloc(8, sizeof(float));
    env->actions = (unsigned int*)calloc(2, sizeof(unsigned int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->paddle_yl_yr = (float*)calloc(2, sizeof(float));
    env->ball_x_y = (float*)calloc(2, sizeof(float));
    env->ball_vx_vy = (float*)calloc(2, sizeof(float));
    env->score_l_r = (unsigned int*)calloc(2, sizeof(unsigned int));
    env->misc_logging = (unsigned int*)calloc(4, sizeof(unsigned int));
}

void free_allocated(CPong* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->paddle_yl_yr);
    free(env->ball_x_y);
    free(env->ball_vx_vy);
    free(env->score_l_r);
    free(env->misc_logging);
}

void compute_observations(CPong* env) {
    env->observations[0] = (env->paddle_yl_yr[0] - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    env->observations[1] = (env->paddle_yl_yr[1] - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    env->observations[2] = env->ball_x_y[0] / env->width;
    env->observations[3] = env->ball_x_y[1] / env->height;
    env->observations[4] = (env->ball_vx_vy[0] + env->ball_initial_speed_x) / (2 * env->ball_initial_speed_x);
    env->observations[5] = (env->ball_vx_vy[1] + env->ball_max_speed_y) / (2 * env->ball_max_speed_y);
    env->observations[6] = env->score_l_r[0] / env->max_score;
    env->observations[7] = env->score_l_r[1] / env->max_score;
}

void reset_round(CPong* env) {
    env->paddle_yl_yr[0] = env->height / 2 - env->paddle_height / 2;
    env->paddle_yl_yr[1] = env->height / 2 - env->paddle_height / 2;
    env->ball_x_y[0] = env->width / 5;
    env->ball_x_y[1] = env->height / 2 - env->ball_height / 2;
    env->ball_vx_vy[0] = env->ball_initial_speed_x;
    env->ball_vx_vy[1] = (rand() % 2 - 1) * env->ball_initial_speed_y;

    env->misc_logging[0] = 1;
    env->misc_logging[1] = env->tick;
    env->misc_logging[2] = env->n_bounces;
    env->misc_logging[3] = env->win;
    env->tick = 0;
    env->n_bounces = 0;
}

void reset(CPong* env) {
    reset_round(env);
    env->score_l_r[0] = 0;
    env->score_l_r[1] = 0;
    compute_observations(env);
}

void step(CPong* env) {
    env->misc_logging[0] = 0; // reset round is over bit
    env->tick += 1;

    env->rewards[0] = 0;
    env->terminals[0] = 0;

    // move ego paddle
    unsigned int act = env->actions[0];
    env->paddle_dir = 0;
    if (act == 0) { // still
        env->paddle_dir = 0;
    } else if (act == 1) { // up
        env->paddle_dir = 1;
    } else if (act == 2) { // down
        env->paddle_dir = -1;
    }

    for (int i = 0; i < env->frameskip; i++) {
        env->paddle_yl_yr[1] += env->paddle_speed * env->paddle_dir;
        
        // move opponent paddle
        float opp_paddle_delta = env->ball_x_y[1] - (env->paddle_yl_yr[0] + env->paddle_height / 2);
        opp_paddle_delta = fminf(fmaxf(opp_paddle_delta, -env->paddle_speed), env->paddle_speed);
        env->paddle_yl_yr[0] += opp_paddle_delta;

        // clip paddles
        env->paddle_yl_yr[1] = fminf(fmaxf(
            env->paddle_yl_yr[1], env->min_paddle_y), env->max_paddle_y);
        env->paddle_yl_yr[0] = fminf(fmaxf(
            env->paddle_yl_yr[0], env->min_paddle_y), env->max_paddle_y);

        // move ball
        env->ball_x_y[0] += env->ball_vx_vy[0];
        env->ball_x_y[1] += env->ball_vx_vy[1];

        // handle collision with top & bottom walls
        if (env->ball_x_y[1] < 0 || env->ball_x_y[1] + env->ball_height > env->height) {
            env->ball_vx_vy[1] = -env->ball_vx_vy[1];
        }

        // handle collision on left
        if (env->ball_x_y[0] < 0) {
            if (env->ball_x_y[1] + env->ball_height > env->paddle_yl_yr[0] && \
                env->ball_x_y[1] < env->paddle_yl_yr[0] + env->paddle_height) {
                // collision with paddle
                env->ball_vx_vy[0] = -env->ball_vx_vy[0];
                env->n_bounces += 1;
            } else {
                // collision with wall: WIN
                env->win = 1;
                env->score_l_r[1] += 1;
                env->rewards[0] = 10.0; // agent wins

                if (env->score_l_r[1] == env->max_score) {
                    env->terminals[0] = 1;
                    reset(env);
                    return;
                } else {
                    reset_round(env);
                    return;
                }
            }
        }

        // handle collision on right (TODO duplicated code)
        if (env->ball_x_y[0] + env->ball_width > env->width) {
            if (env->ball_x_y[1] + env->ball_height > env->paddle_yl_yr[1] && \
                env->ball_x_y[1] < env->paddle_yl_yr[1] + env->paddle_height) {
                // collision with paddle
                env->ball_vx_vy[0] = -env->ball_vx_vy[0];
                env->n_bounces += 1;
                env->rewards[0] = 1.0; // agent bounced the ball
                // ball speed change
                env->ball_vx_vy[1] += env->ball_speed_y_increment * env->paddle_dir;
                env->ball_vx_vy[1] = fminf(fmaxf(env->ball_vx_vy[1], -env->ball_max_speed_y), env->ball_max_speed_y);
                if (fabsf(env->ball_vx_vy[1]) < 0.01) { // we dont want a horizontal ball
                    env->ball_vx_vy[1] = env->ball_speed_y_increment;
                }
            } else {
                // collision with wall: LOSE
                env->win = 0;
                env->score_l_r[0] += 1;
                // env->rewards[0] = -5.0
                if (env->score_l_r[0] == env->max_score) {
                    env->terminals[0] = 1;
                    reset(env);
                    return;
                } else {
                    reset_round(env);
                    return;
                }
            }

            // clip ball
            env->ball_x_y[0] = fminf(fmaxf(env->ball_x_y[0], 0), env->width - env->ball_width);
            env->ball_x_y[1] = fminf(fmaxf(env->ball_x_y[1], 0), env->height - env->ball_height);
        }
        compute_observations(env);
    }
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float x_pad;
    Color paddle_left_color;
    Color paddle_right_color;
    Color ball_color;
};

Client* make_client(CPong* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->paddle_width = env->paddle_width;
    client->paddle_height = env->paddle_height;
    client->ball_width = env->ball_width;
    client->ball_height = env->ball_height;
    client->x_pad = 3*client->paddle_width;
    client->paddle_left_color = (Color){0, 255, 0, 255};
    client->paddle_right_color = (Color){255, 0, 0, 255};
    client->ball_color = (Color){255, 255, 255, 255};

    InitWindow(env->width + 2*client->x_pad, env->height, "PufferLib Pong");
    SetTargetFPS(15);  // 60 / frame_skip

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void render(Client* client, CPong* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw left paddle
    DrawRectangle(
        client->x_pad - client->paddle_width,
        client->height - env->paddle_yl_yr[0] - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_left_color
    );

    // Draw right paddle
    DrawRectangle(
        client->width + client->x_pad,
        client->height - env->paddle_yl_yr[1] - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_right_color
    );

    // Draw ball
    DrawRectangle(
        client->x_pad + env->ball_x_y[0],
        client->height - env->ball_x_y[1] - client->ball_height,
        client->ball_width,
        client->ball_height,
        client->ball_color
    );

    DrawFPS(10, 10);

    // Draw scores
    DrawText(
        TextFormat("%i", env->score_l_r[0]),
        client->width / 2 + client->x_pad - 50 - MeasureText(TextFormat("%i", env->score_l_r[0]), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );
    DrawText(
        TextFormat("%i", env->score_l_r[1]),
        client->width / 2 + client->x_pad + 50 - MeasureText(TextFormat("%i", env->score_l_r[1]), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );

    EndDrawing();
}
