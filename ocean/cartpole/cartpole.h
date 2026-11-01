#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 4
#define NUM_ATNS 1

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

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
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
    unsigned int rng;
};
typedef Env Cartpole;

void add_log(Cartpole* env) {
    env->log.perf += env->episode_return / MAX_STEPS;
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

void puf_close(Cartpole* env) {
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

// Hold Left Shift + A/D or arrows. CPU eval is forward→step→render.
static void cartpole_human_controls(Cartpole *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = 1.0f;
    } else {
        env->agents[0].actions[0] = -1.0f;
    }
}

void puf_render(Cartpole* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    if (env->client == NULL) {
        env->client = make_client(env);
    }

    cartpole_human_controls(env);

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
    DrawText("[Shift] A/D or arrows", 10, 100, 16, PUFF_WHITE);
    EndDrawing();
    puf_web_vsync();
}

void compute_observations(Cartpole* env) {
    float* obs = env->agents[0].observations;
    obs[0] = env->x;
    obs[1] = env->x_dot;
    obs[2] = env->theta;
    obs[3] = env->theta_dot;
}

void puf_reset(Cartpole* env) {
    env->episode_return = 0.0f;
    env->x = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->tick = 0;
    
    compute_observations(env);
}

void puf_step(Cartpole* env) {
    cartpole_human_controls(env);
    float a = env->agents[0].actions[0];
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->agents[0].actions[0] = a;

    float force = env->continuous ? a * env->force_mag
        : (a > 0.5f ? env->force_mag: -env->force_mag);

    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);

    float total_mass = env->cart_mass + env->pole_mass;
    float polemass_length = env->pole_mass * env->pole_length;
    float temp = (force + polemass_length * env->theta_dot * env->theta_dot * sintheta) / total_mass;
    float thetaacc = (env->gravity * sintheta - costheta * temp) / 
                     (env->pole_length * (4.0f / 3.0f - env->pole_mass * costheta * costheta / total_mass));
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

    env->agents[0].rewards[0] = done ? 0.0f : 1.0f;
    env->episode_return += env->agents[0].rewards[0];
    env->agents[0].terminals[0] = terminated ? 1 : 0;

    if (done) {
        add_log(env);
        puf_reset(env);
    }

    compute_observations(env);
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "pole_angle_termination", log->pole_angle_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->cart_mass = dict_get(kwargs, "cart_mass");
    env->pole_mass = dict_get(kwargs, "pole_mass");
    env->pole_length = dict_get(kwargs, "pole_length");
    env->gravity = dict_get(kwargs, "gravity");
    env->force_mag = dict_get(kwargs, "force_mag");
    env->tau = dict_get(kwargs, "dt");
    env->continuous = dict_get(kwargs, "continuous");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
}

