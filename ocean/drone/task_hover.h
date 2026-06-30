#pragma once

#include "drone.h"

// types

#define HOVER_SCORE_DIST_SCALE 0.01f
#define HOVER_SCORE_VEL_SCALE 0.01f
#define HOVER_SCORE_OMEGA_SCALE 0.1f

typedef struct {
    float target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;
} HoverConfig;

typedef struct {
    float* prev_potential;
    float* score;
    float* perf;
    float* ema_dist;
    float* ema_vel;
    float* ema_omega;
} HoverState;

// lifecycle

static void hover_init(DroneEnv* env) {
    HoverState* state = (HoverState*)calloc(1, sizeof(HoverState));
    state->prev_potential = (float*)calloc(env->num_agents, sizeof(float));
    state->score = (float*)calloc(env->num_agents, sizeof(float));
    state->perf = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_dist = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_vel = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_omega = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

static void hover_close(DroneEnv* env) {
    HoverState* state = (HoverState*)env->task_state;
    if (state != NULL) {
        free(state->prev_potential);
        free(state->score);
        free(state->perf);
        free(state->ema_dist);
        free(state->ema_vel);
        free(state->ema_omega);
        free(state);
    }
    free(env->task_config);
}

// helpers

static inline void hover_set_target(unsigned int* rng, Drone* agent, float target_dist) {
    float u = rndf(0.0f, 1.0f, rng);
    float v = rndf(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * (float)M_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    Vec3 dir = (Vec3){r_xy * cosf(a), r_xy * sinf(a), z};

    float rad = target_dist * cbrtf(rndf(0.0f, 1.0f, rng));
    Vec3 p = add3(agent->state.pos, scalmul3(dir, rad));

    agent->target->pos = (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z),
    };
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

static inline float hover_potential(float dist, float vel, float omega, HoverConfig* cfg) {
    float d = 1.0f / (1.0f + dist / cfg->hover_dist);
    float v = 1.0f / (1.0f + vel / cfg->hover_vel);
    float w = 1.0f / (1.0f + omega / cfg->hover_omega);
    return d * (0.7f + 0.15f * v + 0.15f * w);
}

static inline float hover_score(float dist, float vel, float omega) {
    float d = dist / HOVER_SCORE_DIST_SCALE;
    float v = vel / HOVER_SCORE_VEL_SCALE;
    float w = omega / HOVER_SCORE_OMEGA_SCALE;
    float penalty = 0.7f * d + 0.15f * v + 0.15f * w;
    return 1.0f / (1.0f + 0.05f * penalty);
}

// callbacks

static void hover_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;

    agent->state.pos = random_pos(&env->rng);
    hover_set_target(&env->rng, agent, cfg->target_dist);

    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    state->score[idx] = 0.0f;
    state->perf[idx] = hover_score(dist, vel, omega);
    state->ema_dist[idx] = dist;
    state->ema_vel[idx] = vel;
    state->ema_omega[idx] = omega;
    state->prev_potential[idx] = hover_potential(dist, vel, omega, cfg);
}

static float hover_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;

    float curr = hover_potential(cache->dist, cache->vel, cache->omega, cfg);
    float reward = cfg->alpha_dist * (cache->prev_dist - cache->dist) + cfg->alpha_hover * curr +
                   cfg->alpha_shaping * (curr - state->prev_potential[idx]) -
                   cfg->alpha_omega * cache->omega;
    state->prev_potential[idx] = curr;

    float score = hover_score(cache->dist, cache->vel, cache->omega);
    state->score[idx] += score;
    state->perf[idx] = 0.98f * state->perf[idx] + 0.02f * score;
    state->ema_dist[idx] = 0.99f * state->ema_dist[idx] + 0.01f * cache->dist;
    state->ema_vel[idx] = 0.99f * state->ema_vel[idx] + 0.01f * cache->vel;
    state->ema_omega[idx] = 0.99f * state->ema_omega[idx] + 0.01f * cache->omega;
    return reward;
}

static bool hover_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    return cache->dist > (cfg->target_dist + 1.0f) || agent->episode_length >= HORIZON;
}

static void hover_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;
    log->score += state->score[idx];
    log->perf += state->perf[idx];
    log_task_add(log, 0, state->ema_dist[idx]);
    log_task_add(log, 1, state->ema_vel[idx]);
    log_task_add(log, 2, state->ema_omega[idx]);
    log_task_add(log, 3, cache->dist > (cfg->target_dist + 1.0f) ? 1.0f : 0.0f);
}

// definition

static const Task TASK_HOVER = {
    .name = "hover",
    .log_keys = {"ema_dist", "ema_vel", "ema_omega", "oob"},
    .num_log_keys = 4,
    .init = hover_init,
    .close = hover_close,
    .env_reset = NULL,
    .reset = hover_reset,
    .reward = hover_reward,
    .done = hover_done,
    .log = hover_log,
    .render = NULL,
};
