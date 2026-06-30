#pragma once

#include "drone.h"

// types

typedef struct {
    int max_rings;
    float ring_reward;
    float collision_penalty;
    float time_penalty;
    float alpha_dist;
} RaceConfig;

typedef struct {
    Target* ring_buffer;
    int* ring_idx;
    int* rings_passed;
    float* collisions;
} RaceState;

// lifecycle

static void race_init(DroneEnv* env) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)calloc(1, sizeof(RaceState));
    state->ring_buffer = (Target*)calloc(cfg->max_rings, sizeof(Target));
    state->ring_idx = (int*)calloc(env->num_agents, sizeof(int));
    state->rings_passed = (int*)calloc(env->num_agents, sizeof(int));
    state->collisions = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

static void race_close(DroneEnv* env) {
    RaceState* state = (RaceState*)env->task_state;
    if (state != NULL) {
        free(state->ring_buffer);
        free(state->ring_idx);
        free(state->rings_passed);
        free(state->collisions);
        free(state);
    }
    free(env->task_config);
}

// helpers

static inline void reset_rings(unsigned int* rng, Target* ring_buffer, int num_rings) {
    ring_buffer[0] = rndring(rng, RING_RADIUS);
    for (int i = 1; i < num_rings; i++) {
        do {
            ring_buffer[i] = rndring(rng, RING_RADIUS);
        } while (norm3(sub3(ring_buffer[i].pos, ring_buffer[i - 1].pos)) < 2.0f * RING_RADIUS);
    }
}

static inline int check_ring(Drone* drone, Target* ring) {
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);
    float new_dot = dot3(sub3(drone->state.pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    if (valid_dir || invalid_dir) {
        Vec3 dir = sub3(drone->state.pos, drone->prev_pos);
        float denom = dot3(ring->normal, dir);
        if (fabsf(denom) < 1e-9f) return 0;

        float t = -prev_dot / denom;
        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float d = norm3(sub3(intersection, ring->pos));

        if (d < (ring->radius - 0.5f) && valid_dir) return 1;
        if (d < ring->radius + 0.5f) return -1;
    }
    return 0;
}

// callbacks

static void race_env_reset(DroneEnv* env) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    reset_rings(&env->rng, state->ring_buffer, cfg->max_rings);
}

static void race_reset(DroneEnv* env, Drone* agent, int idx) {
    RaceState* state = (RaceState*)env->task_state;

    do {
        agent->state.pos = random_pos(&env->rng);
    } while (norm3(sub3(agent->state.pos, state->ring_buffer[0].pos)) < 2.0f * RING_RADIUS);

    state->ring_idx[idx] = 0;
    state->rings_passed[idx] = 0;
    state->collisions[idx] = 0.0f;
    *agent->target = state->ring_buffer[0];
}

static float race_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;

    float reward = cfg->alpha_dist * (cache->prev_dist - cache->dist);

    int result = check_ring(agent, &state->ring_buffer[state->ring_idx[idx]]);
    if (result == 1) {
        state->rings_passed[idx]++;
        state->ring_idx[idx]++;
        if (state->ring_idx[idx] < cfg->max_rings)
            *agent->target = state->ring_buffer[state->ring_idx[idx]];
        reward += cfg->ring_reward;
    } else if (result == -1) {
        state->collisions[idx] += 1.0f;
        reward -= cfg->collision_penalty;
    }

    reward -= cfg->time_penalty;
    return reward;
}

static bool race_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    return state->rings_passed[idx] >= cfg->max_rings || agent->episode_length >= HORIZON;
}

static void race_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    float completed = state->rings_passed[idx] >= cfg->max_rings ? 1.0f : 0.0f;
    log->score += (float)state->rings_passed[idx];
    log->perf += completed;
    log_task_add(log, 0, (float)state->rings_passed[idx]);
    log_task_add(log, 1, state->collisions[idx]);
    log_task_add(log, 2, completed);
}

static void race_render(DroneEnv* env, Client* client) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    for (int i = 0; i < cfg->max_rings; i++)
        DrawRing3D(state->ring_buffer[i], 0.2f, GREEN, BLUE);
}

// definition

static const Task TASK_RACE = {
    .name = "race",
    .log_keys = {"rings_passed", "ring_collisions", "completed"},
    .num_log_keys = 3,
    .init = race_init,
    .close = race_close,
    .env_reset = race_env_reset,
    .reset = race_reset,
    .reward = race_reward,
    .done = race_done,
    .log = race_log,
    .render = race_render,
};
