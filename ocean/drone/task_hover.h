#include "drone.h"

// types

#define HOVER_SCORE_DIST_SCALE 0.01f
#define HOVER_SCORE_VEL_SCALE 0.01f
#define HOVER_SCORE_OMEGA_SCALE 0.01f

typedef struct {
    float target_dist;
    float alpha_hover;
    float alpha_dist;
    float sphere_radius;
    int horizon;
} HoverConfig;

typedef struct {
    float* score;
    float* perf;
    float* ema_dist;
    float* ema_vel;
    float* ema_omega;
} HoverState;

// lifecycle

static void hover_init(DroneEnv* env) {
    HoverState* state = (HoverState*)calloc(1, sizeof(HoverState));
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

static inline Vec3 random_ball_offset(unsigned int* rng, float radius) {
    float u = rndf(0.0f, 1.0f, rng);
    float v = rndf(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * (float)M_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    Vec3 dir = (Vec3){r_xy * cosf(a), r_xy * sinf(a), z};
    return scalmul3(dir, radius * cbrtf(rndf(0.0f, 1.0f, rng)));
}

static inline float hover_score(float dist, float vel, float omega) {
    float d = dist / HOVER_SCORE_DIST_SCALE;
    float v = vel / HOVER_SCORE_VEL_SCALE;
    float w = omega / HOVER_SCORE_OMEGA_SCALE;
    float penalty = 0.7f * d + 0.15f * v + 0.15f * w;
    return 1.0f / (1.0f + 0.05f * penalty);
}

static void hover_reset_to(DroneEnv* env, Drone* agent, int idx, Vec3 target, float spawn_dist) {
    HoverState* state = (HoverState*)env->task_state;

    agent->target->pos = target;
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target->normal = (Vec3){0.0f, 0.0f, 0.0f};

    Vec3 p = add3(target, random_ball_offset(&env->rng, spawn_dist));
    agent->state.pos = (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z),
    };

    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);
    state->score[idx] = 0.0f;
    state->perf[idx] = hover_score(dist, vel, omega);
    state->ema_dist[idx] = dist;
    state->ema_vel[idx] = vel;
    state->ema_omega[idx] = omega;
}

static inline Vec3 sphere_slot(int idx, int num_agents, float radius) {
    float phi = (float)M_PI * (sqrtf(5.0f) - 1.0f);
    float y = 1.0f - 2.0f * ((float)idx / (float)num_agents);
    float r = sqrtf(fmaxf(0.0f, 1.0f - y * y));
    float theta = phi * (float)idx;
    return (Vec3){radius * cosf(theta) * r, radius * sinf(theta) * r, radius * y};
}

static inline float cube_axis(int i, int side, float radius) {
    if (side <= 1) return 0.0f;
    return radius * (2.0f * (float)i / (float)(side - 1) - 1.0f);
}

static inline Vec3 cube_slot(int idx, int num_agents, float radius) {
    float r = radius * 0.57735027f;
    int side = (int)ceilf(cbrtf((float)num_agents));
    int x = idx % side;
    int y = (idx / side) % side;
    int z = idx / (side * side);
    return (Vec3){cube_axis(x, side, r), cube_axis(y, side, r), cube_axis(z, side, r)};
}

static inline Vec3 flag_slot(int idx) {
    float y = (float)(idx % 8) - 3.5f;
    float z = 2.5f - 0.75f * (float)(idx / 8);
    return (Vec3){0.0f, y, z};
}

// callbacks

static void hover_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    hover_reset_to(env, agent, idx, random_pos(&env->rng), cfg->target_dist);
}

static void sphere_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    Vec3 slot = sphere_slot(idx, env->num_agents, cfg->sphere_radius);
    hover_reset_to(env, agent, idx, slot, cfg->target_dist);
}

static void cube_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    Vec3 slot = cube_slot(idx, env->num_agents, cfg->sphere_radius);
    hover_reset_to(env, agent, idx, slot, cfg->target_dist);
}

static void flag_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    hover_reset_to(env, agent, idx, flag_slot(idx), cfg->target_dist);
}

static float hover_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;

    float score = hover_score(cache->dist, cache->vel, cache->omega);
    float reward = cfg->alpha_hover * score;
    reward += cfg->alpha_dist * (cache->prev_dist - cache->dist);

    state->score[idx] += score;
    state->perf[idx] = 0.98f * state->perf[idx] + 0.02f * score;
    state->ema_dist[idx] = 0.99f * state->ema_dist[idx] + 0.01f * cache->dist;
    state->ema_vel[idx] = 0.99f * state->ema_vel[idx] + 0.01f * cache->vel;
    state->ema_omega[idx] = 0.99f * state->ema_omega[idx] + 0.01f * cache->omega;
    if (cache->dist > cfg->target_dist + 1.0f) reward -= env->oob_penalty; // sphere exit ends the episode
    return reward;
}

static bool hover_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    return cache->dist > (cfg->target_dist + 1.0f) || agent->episode_length >= cfg->horizon;
}

static void hover_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;
    TaskLog* t = &log->task[env->task];
    t->n += 1.0f;
    t->perf += state->perf[idx];
    t->score += state->score[idx];
    t->keys[0] += state->ema_dist[idx];
    t->keys[1] += state->ema_vel[idx];
    t->keys[2] += state->ema_omega[idx];
    t->keys[3] += cache->dist > (cfg->target_dist + 1.0f) ? 1.0f : 0.0f;
}
