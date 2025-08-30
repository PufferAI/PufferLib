#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "raylib.h"

#define X_THRESHOLD 2.4f
#define THETA_THRESHOLD_RADIANS (12 * 2 * M_PI / 360)
#define MAX_STEPS 200
#define WIDTH 600
#define HEIGHT 200
#define SCALE 100

typedef struct Log Log;
struct Log {
    float perf;
    float episode_length;
    float x_threshold_termination;
    float pole_angle_termination;
    float max_steps_termination;
    float n;
    float score;
};

typedef struct Client Client;
struct Client {
};

typedef struct Cartpole Cartpole;
struct Cartpole {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    Client* client;
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    int tick;
    float cart_mass;
    float pole_mass;
    float pole_length;
    float gravity;
    float force_mag;
    float tau;
    int continuous;
    float episode_return;
};

void add_log(Cartpole* env) {
    if (env->episode_return > 0) {
        env->log.perf = env->episode_return / MAX_STEPS;
    } else {
        env->log.perf = 0.0f;
    }
    env->log.episode_length += env->tick;
    env->log.score += env->tick;
    env->log.x_threshold_termination += (env->x < -X_THRESHOLD || env->x > X_THRESHOLD);
    env->log.pole_angle_termination += (env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS);
    env->log.max_steps_termination += (env->tick >= MAX_STEPS);
    env->log.n += 1;
}

void init(Cartpole* env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
}

void allocate(Cartpole* env) {
    init(env);
    env->observations = (float*)calloc(4, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Cartpole* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_close(Cartpole* env) {
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

Client* make_client(Cartpole* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(WIDTH, HEIGHT, "puffer Cartpole");
    SetTargetFPS(60);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Cartpole* env) {
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    if (IsKeyPressed(KEY_TAB))
        ToggleFullscreen();

    if (env->client == NULL) {
        env->client = make_client(env);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, HEIGHT / 1.5, WIDTH, HEIGHT / 1.5, PUFF_CYAN);
    float cart_x = WIDTH / 2 + env->x * SCALE;
    float cart_y = HEIGHT / 1.6;
    DrawRectangle((int)(cart_x - 20), (int)(cart_y - 10), 40, 20, PUFF_CYAN);
    float pole_length = 2.0f * 0.5f * SCALE;
    float pole_x2 = cart_x + sinf(env->theta) * pole_length;
    float pole_y2 = cart_y - cosf(env->theta) * pole_length;
    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, PUFF_RED);
    DrawText(TextFormat("Steps: %i", env->tick), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Cart Position: %.2f", env->x), 10, 40, 20, PUFF_WHITE);
    DrawText(TextFormat("Pole Angle: %.2f", env->theta * 180.0f / M_PI), 10, 70, 20, PUFF_WHITE);
    EndDrawing();
}

void compute_observations(Cartpole* env) {
    env->observations[0] = env->x;
    env->observations[1] = env->x_dot;
    env->observations[2] = env->theta;
    env->observations[3] = env->theta_dot;
}

void c_reset(Cartpole* env) {
    env->episode_return = 0.0f;
    env->x = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->tick = 0;
    
    compute_observations(env);
}

void c_step(Cartpole* env) {  
    // float force = 0.0;
    // if (env->continuous) {
    //     force = env->actions[0] * FORCE_MAG;
    // } else {
    //     force = (env->actions[0] > 0.5f) ? FORCE_MAG : -FORCE_MAG; 
    // }

    float a = env->actions[0];

    /* ===== runtime sanity check –– delete after debugging ===== */
    if (!isfinite(a) || a < -1.0001f || a > 1.0001f) {
        fprintf(stderr,
                "[BAD ACTION] tick=%d  raw=%.6f\n",
                env->tick, a);
        fflush(stderr);
    }
    /* ========================================================== */

    if (!isfinite(a)) {
        a = 0.0f;
    }
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->actions[0] = a;

    float force = env->continuous ? a * env->force_mag
                                  : (a > 0.5f ? env->force_mag: -env->force_mag);

    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);

    float total_mass = env->cart_mass + env->pole_mass;
    float polemass_length = total_mass + env->pole_mass;
    float temp = (force + polemass_length * env->theta_dot * env->theta_dot * sintheta) / total_mass;
    float thetaacc = (env->gravity * sintheta - costheta * temp) / 
                     (env->pole_length * (4.0f / 3.0f - total_mass * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    env->x += env->tau * env->x_dot;
    env->x_dot += env->tau * xacc;
    env->theta += env->tau * env->theta_dot;
    env->theta_dot += env->tau * thetaacc;

    env->tick += 1;
    
    bool terminated = env->x < -X_THRESHOLD || env->x > X_THRESHOLD ||
                env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS;
    bool truncated = env->tick >= MAX_STEPS;
    bool done = terminated || truncated;

    env->rewards[0] = done ? 0.0f : 1.0f;
    env->episode_return += env->rewards[0];
    env->terminals[0] = terminated ? 1 : 0;

    if (done) {
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}
