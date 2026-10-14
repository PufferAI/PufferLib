/* MIT, Copyright (c) 2026 Marc / Rage Industries.
 * Puffer environment: ten hanging passive poles and one delayed cart force.
 */
#ifndef PUFFER_NPOLE_H
#define PUFFER_NPOLE_H

#include <stdlib.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include "npole_physics.h"

#define ACT_SIZES {1}
#define OBS_SIZE 34
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 50
#define NP_WIDTH 960
#define NP_HEIGHT 840

struct Log {
    float perf;
    float score;
    float episode_length;
    float hold_time;
    float rail_failure;
    float numeric_failure;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    unsigned int rng;
    int tag;
    int boundary_reached;
    NpEpisode episode;
    double episode_return;
};

static void np_observe(Env *env) {
    float *obs = env->agents[0].observations;
    const NpState *state = &env->episode.state;
    obs[0] = (float)state->y[0];
    obs[1] = (float)state->y[NP_DOF];
    for (int i = 0; i < NP_POLES; i++) {
        obs[2 + 3 * i] = (float)sin(state->y[1 + i]);
        obs[3 + 3 * i] = (float)cos(state->y[1 + i]);
        obs[4 + 3 * i] = (float)(state->y[NP_DOF + 1 + i] / 8.0);
    }
    obs[32] = (float)state->last_applied;
    obs[33] = (float)state->pending;
}

void puf_init(Env *env, Dict *kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    memset(&env->log, 0, sizeof(env->log));
}

void puf_reset(Env *env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    np_reset_seed(&env->episode, env->rng);
    env->episode_return = 0.0;
    np_observe(env);
}

void puf_step(Env *env) {
    Agent *agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
    double action = agent->actions[0];
    if (isfinite(action)) {
        action = np_clip(action);
        agent->actions[0] = (float)action;
    }
    int failed = np_episode_tick(&env->episode, action);
    double height = 0.0;
    for (int i = 0; i < NP_POLES; i++) {
        height += cos(env->episode.state.y[1 + i]);
    }
    /* Dense training reward is distinct from perf's strict full-episode gate. */
    double reward = failed ? -1.0 : 0.5 + height / (2.0 * NP_POLES)
        + 0.2 * np_good(env->episode.state.y)
        - 0.001 * action * action;
    if (!isfinite(reward)) {
        reward = -1.0;
    }
    agent->rewards[0] = (float)reward;
    env->episode_return += reward;
    if (failed || env->episode.state.tick >= NP_HORIZON) {
        env->log.perf += np_success(&env->episode);
        env->log.score += (float)env->episode_return;
        env->log.episode_length += env->episode.state.tick;
        env->log.hold_time += (float)(NP_DT * env->episode.hold);
        env->log.rail_failure += !env->episode.rail_ok;
        env->log.numeric_failure += !env->episode.numeric_ok;
        env->log.n += 1.0f;
        /* Puffer's terminal also resets recurrent state on time-limit reset. */
        agent->terminals[0] = 1.0f;
        puf_reset(env);
    } else {
        np_observe(env);
    }
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hold_time", log->hold_time);
    dict_set(out, "rail_failure", log->rail_failure);
    dict_set(out, "numeric_failure", log->numeric_failure);
    dict_set(out, "n", log->n);
}

#ifndef NP_HEADLESS
static void np_draw(const NpEpisode *episode, const char *status,
        unsigned int seed, const char *controls) {
    const Color background = {6, 24, 24, 255};
    const Color foreground = {224, 240, 234, 255};
    const Color muted = {137, 169, 162, 255};
    const Color accent = {50, 210, 181, 255};
    const Color colors[NP_POLES] = {
        {80, 214, 198, 255}, {92, 188, 239, 255}, {130, 158, 250, 255},
        {172, 136, 238, 255}, {217, 128, 199, 255}, {244, 139, 149, 255},
        {251, 170, 109, 255}, {236, 199, 109, 255}, {177, 211, 123, 255},
        {123, 218, 163, 255}
    };
    float scale = 60.0f;
    float cart_x = NP_WIDTH * 0.5f + (float)episode->state.y[0] * scale;
    float cart_y = 444.0f;
    BeginDrawing();
    ClearBackground(background);
    DrawText("TEN PENDULUMS", 30, 24, 32, foreground);
    DrawText("10 passive links. One cart force.", 30, 67, 18, muted);
    DrawText(status, 30, 102, 22, accent);
    DrawText(TextFormat("Time  %.2f / 20 s", episode->state.tick * NP_DT),
        680, 28, 20, foreground);
    DrawText(TextFormat("Force  %+.2f N", episode->state.last_applied * 10.0),
        680, 61, 20, foreground);
    DrawText(TextFormat("Final hold  %.2f s", episode->hold * NP_DT),
        680, 94, 20, muted);
    float left = NP_WIDTH * 0.5f - (float)NP_RAIL * scale;
    float right = NP_WIDTH * 0.5f + (float)NP_RAIL * scale;
    DrawLineEx((Vector2){left, cart_y + 15}, (Vector2){right, cart_y + 15}, 2, muted);
    DrawLine((int)left, (int)cart_y - 14, (int)left, (int)cart_y + 25, muted);
    DrawLine((int)right, (int)cart_y - 14, (int)right, (int)cart_y + 25, muted);
    DrawRectangleRounded((Rectangle){cart_x - 18, cart_y - 9, 36, 20}, 0.25f, 4, foreground);
    DrawCircle((int)cart_x - 11, (int)cart_y + 13, 4, accent);
    DrawCircle((int)cart_x + 11, (int)cart_y + 13, 4, accent);
    Vector2 point = {cart_x, cart_y};
    for (int i = 0; i < NP_POLES; i++) {
        Vector2 next = {
            point.x + (float)(0.5 * sin(episode->state.y[1 + i])) * scale,
            point.y - (float)(0.5 * cos(episode->state.y[1 + i])) * scale
        };
        DrawLineEx(point, next, 4.0f, colors[i]);
        DrawCircleV(next, 4.5f, colors[i]);
        point = next;
    }
    DrawText("50 Hz commands  /  20 ms delay  /  10 N limit", 30, 772, 18, muted);
    DrawText(TextFormat("Seed %u", seed), 30, 805, 16, muted);
    DrawText(controls, 395, 805, 16, foreground);
    EndDrawing();
}
#endif

void puf_render(Env *env) {
#ifndef NP_HEADLESS
    if (!IsWindowReady()) {
        InitWindow(NP_WIDTH, NP_HEIGHT, "Ten Pendulums");
        SetTargetFPS(60);
    }
    np_draw(&env->episode, "Puffer policy", env->rng, "");
    puf_web_vsync();
#else
    (void)env;
#endif
}

void puf_close(Env *env) {
    (void)env;
#ifndef NP_HEADLESS
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}

#endif
