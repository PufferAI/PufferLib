// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone
//
// 5c API port of Fin's multitask drone (PR #599 / FinlaySanders/4.0).

#pragma once

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "dronelib.h"
#include "physics.h"
typedef float obs_t;
#include "pufferenv.h"

// 5c API surface (pufferl / puffercpu)
#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define MAX_DRONES 256
// Train is 100 Hz. Eval is 60 fps, so puffercpu emits 5 steps per 3 frames.
#define PUF_STEPS_PER_SEC 100

typedef Env DroneEnv;

typedef enum {
    TASK_HOVER = 0,
    TASK_RACE = 1,
    TASK_SPHERE = 2,
    TASK_CUBE = 3,
    TASK_FLAG = 4,
} TaskType;

#define NUM_TASKS 5

typedef struct {
    float dist;
    float prev_dist;
    float vel;
    float omega;
} StepCache;

typedef struct {
    float n;
    float perf;
    float score;
    float keys[4];
} TaskLog;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    TaskLog task[NUM_TASKS];
    float n;
};

typedef struct Client Client;

struct Env {
    Client* client;
    Agent agents[MAX_DRONES];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;

    Drone* drones;
    // Host-only SIMD kernel. void* so Env stays CUDA-device-safe (no GCC vectors).
    void* physics;
    Log log;

    TaskType task;
    void* task_config;
    void* task_state;

    // Shared reward shaping
    float alpha_vel;
    float alpha_omega;
    float alpha_action;

    // Domain randomisation
    float dr;

    // Physics integrator (0=RK4, 1=RK2)
    int integrator;

    // Cached task configs so TAB can switch hover/race/sphere/cube/flag.
    float hover_target_dist;
    float alpha_hover;
    float hover_alpha_dist;
    float sphere_radius;
    int hover_horizon;
    int max_rings;
    float ring_reward;
    float race_alpha_dist;
    int race_horizon;
};

// Task sampling fractions (set in puf_init for puf_log episode_frac keys)
static float task_fracs[NUM_TASKS];

// Forward decls used by tasks / render before includes
void reset_agent(DroneEnv* env, int idx);
void puf_close(DroneEnv* env);
void close_client(Client* client);

#include "tasklib.h"

void init(DroneEnv* env) {
    assert(env->num_agents > 0 && env->num_agents <= MAX_DRONES);
    env->drones = (Drone*)calloc(env->num_agents, sizeof(Drone));
    for (int i = 0; i < env->num_agents; i++)
        env->drones[i].target = (Target*)calloc(1, sizeof(Target));

    env->physics = calloc(1, sizeof(Physics));
    physics_init((Physics*)env->physics, env->num_agents, env->integrator);
    env->log = (Log){0};
}

void add_log(DroneEnv* env, int idx, StepCache* cache) {
    Drone* agent = &env->drones[idx];
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.n += 1.0f;
    task_log(env, agent, idx, &env->log, cache);
}

void reset_agent(DroneEnv* env, int idx) {
    Drone* agent = &env->drones[idx];
    Target* target = agent->target;
    memset(agent, 0, sizeof(Drone));
    agent->target = target;

    init_drone(agent, &env->rng, env->dr);
    task_reset(env, agent, idx);
    physics_set_drone((Physics*)env->physics, idx, &agent->params, &agent->state);
    agent->prev_pos = agent->state.pos;
}

void compute_observations(DroneEnv* env) {
    bool is_race = (env->task == TASK_RACE);
    for (int i = 0; i < env->num_agents; i++)
        compute_drone_observations(&env->drones[i],
            env->agents[i].observations, is_race);
}

// Contiguous action buffer base (pufferl/puffercpu layout agents[i] stride NUM_ATNS)
static inline float* drone_actions_base(DroneEnv* env) {
    return env->agents[0].actions;
}

void puf_reset(DroneEnv* env) {
    task_env_reset(env);
    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, i);
        if (env->agents[i].terminals) env->agents[i].terminals[0] = 0.0f;
        if (env->agents[i].rewards) env->agents[i].rewards[0] = 0.0f;
    }
    compute_observations(env);
}

// One ACTION_DT (100 Hz) of physics + rewards. Training calls this once.
static void drone_tick(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++)
        env->drones[i].prev_pos = env->drones[i].state.pos;

    physics_step((Physics*)env->physics, drone_actions_base(env));

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->drones[i];
        float* action = env->agents[i].actions;
        agent->episode_length++;

        agent->state = physics_get_state((Physics*)env->physics, i);
        StepCache cache;
        cache.dist = norm3(sub3(agent->target->pos, agent->state.pos));
        cache.prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
        cache.vel = norm3(agent->state.vel);
        cache.omega = norm3(agent->state.omega);

        float reward = task_reward(env, agent, i, &cache);
        reward -= env->alpha_vel * cache.vel;
        reward -= env->alpha_omega * cache.omega;

        if (agent->episode_length > 1) {
            float da = 0.0f;
            for (int k = 0; k < 4; k++) {
                float d = action[k] - agent->prev_action[k];
                da += d * d;
            }
            reward -= env->alpha_action * da;
        }
        for (int k = 0; k < 4; k++) agent->prev_action[k] = action[k];

        bool done = task_done(env, agent, i, &cache);

        agent->episode_return += reward;
        env->agents[i].rewards[0] = reward;
        env->agents[i].terminals[0] = done ? 1.0f : 0.0f;

        if (done) {
            add_log(env, i, &cache);
            reset_agent(env, i);
        }
    }

    compute_observations(env);
}

void puf_step(DroneEnv* env) {
    drone_tick(env);
}

void puf_close(DroneEnv* env) {
    task_close(env);

    if (env->drones != NULL) {
        for (int i = 0; i < env->num_agents; i++)
            free(env->drones[i].target);
        free(env->drones);
        env->drones = NULL;
    }
    if (env->physics != NULL) {
        physics_close((Physics*)env->physics);
        free(env->physics);
        env->physics = NULL;
    }

    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
}

static void drone_fill_task_config(DroneEnv* env) {
    if (env->task == TASK_RACE) {
        RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
        cfg->max_rings = env->max_rings;
        cfg->ring_reward = env->ring_reward;
        cfg->alpha_dist = env->race_alpha_dist;
        cfg->horizon = env->race_horizon;
        env->task_config = cfg;
    } else {
        HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
        cfg->target_dist = env->hover_target_dist;
        cfg->alpha_hover = env->alpha_hover;
        cfg->alpha_dist = env->hover_alpha_dist;
        cfg->sphere_radius = env->sphere_radius;
        cfg->horizon = env->hover_horizon;
        env->task_config = cfg;
    }
}

static void drone_set_task(DroneEnv* env, TaskType task) {
    if (env->task_config) task_close(env);
    env->task = task;
    drone_fill_task_config(env);
    task_init(env);
    if (env->drones) puf_reset(env);
}

static TaskType drone_next_task(TaskType cur) {
    return (TaskType)(((int)cur + 1) % NUM_TASKS);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_drones");
    if (env->num_agents <= 0 || env->num_agents > MAX_DRONES) {
        fprintf(stderr, "drone: num_drones=%d out of range (1..%d)\n",
                env->num_agents, MAX_DRONES);
        exit(1);
    }

    env->alpha_vel = dict_get(kwargs, "alpha_vel");
    env->alpha_omega = dict_get(kwargs, "alpha_omega");
    env->alpha_action = dict_get(kwargs, "alpha_action");
    env->dr = dict_get(kwargs, "dr");
    env->integrator = dict_get(kwargs, "use_rk2");

    task_fracs[TASK_HOVER] = dict_get(kwargs, "hover_frac");
    task_fracs[TASK_RACE] = dict_get(kwargs, "race_frac");
    task_fracs[TASK_SPHERE] = dict_get(kwargs, "sphere_frac");
    task_fracs[TASK_CUBE] = dict_get(kwargs, "cube_frac");
    task_fracs[TASK_FLAG] = dict_get(kwargs, "flag_frac");

    float total = 0.0f;
    for (int t = 0; t < NUM_TASKS; t++) total += task_fracs[t];
    if (total <= 0.0f) total = 1.0f;

    // Deterministic per-env task assignment from rng seed index (Fin binding)
    int idx = (int)env->rng;
    float cum = 0.0f;
    env->task = TASK_HOVER;
    for (int t = 0; t < NUM_TASKS; t++) {
        cum += task_fracs[t] / total;
        if ((int)floorf((idx + 1) * cum) > (int)floorf(idx * cum)) {
            env->task = (TaskType)t;
            break;
        }
    }

    env->hover_target_dist = dict_get(kwargs, "hover_target_dist");
    env->alpha_hover = dict_get(kwargs, "alpha_hover");
    env->hover_alpha_dist = dict_get(kwargs, "hover_alpha_dist");
    env->sphere_radius = dict_get(kwargs, "sphere_radius");
    env->hover_horizon = (int)dict_get(kwargs, "hover_horizon");
    env->max_rings = (int)dict_get(kwargs, "max_rings");
    env->ring_reward = dict_get(kwargs, "ring_reward");
    env->race_alpha_dist = dict_get(kwargs, "race_alpha_dist");
    env->race_horizon = (int)dict_get(kwargs, "race_horizon");

    drone_fill_task_config(env);

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }

    task_init(env);
    init(env);
}

static inline float task_avg(float sum, float n) { return n > 0.0f ? sum / n : 0.0f; }

void puf_log(Log* log, Dict* out) {
    static int first = 1;

    float perf = 0.0f, score = 0.0f;
    int active = 0;
    for (int t = 0; t < NUM_TASKS; t++) {
        float n = log->task[t].n;
        if (n <= 0.0f) continue;
        perf += log->task[t].perf / n;
        score += log->task[t].score / n;
        active++;
    }
    dict_set(out, "perf", active > 0 ? perf / active : 0.0f);
    dict_set(out, "score", active > 0 ? score / active : 0.0f);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    if (log->task[TASK_HOVER].n > 0.0f || (first && task_fracs[TASK_HOVER] > 0.0f)) {
        TaskLog* h = &log->task[TASK_HOVER];
        dict_set(out, "hover/perf", task_avg(h->perf, h->n));
        dict_set(out, "hover/score", task_avg(h->score, h->n));
        dict_set(out, "hover/ema_dist", task_avg(h->keys[0], h->n));
        dict_set(out, "hover/ema_vel", task_avg(h->keys[1], h->n));
        dict_set(out, "hover/ema_omega", task_avg(h->keys[2], h->n));
        dict_set(out, "hover/oob", task_avg(h->keys[3], h->n));
        dict_set(out, "hover/episode_frac", h->n);
    }
    if (log->task[TASK_RACE].n > 0.0f || (first && task_fracs[TASK_RACE] > 0.0f)) {
        TaskLog* r = &log->task[TASK_RACE];
        dict_set(out, "race/perf", task_avg(r->perf, r->n));
        dict_set(out, "race/score", task_avg(r->score, r->n));
        dict_set(out, "race/rings_passed", task_avg(r->keys[0], r->n));
        dict_set(out, "race/ring_collisions", task_avg(r->keys[1], r->n));
        dict_set(out, "race/completed", task_avg(r->keys[2], r->n));
        dict_set(out, "race/oob", task_avg(r->keys[3], r->n));
        dict_set(out, "race/episode_frac", r->n);
    }
    if (log->task[TASK_SPHERE].n > 0.0f || (first && task_fracs[TASK_SPHERE] > 0.0f)) {
        TaskLog* s = &log->task[TASK_SPHERE];
        dict_set(out, "sphere/perf", task_avg(s->perf, s->n));
        dict_set(out, "sphere/score", task_avg(s->score, s->n));
        dict_set(out, "sphere/ema_dist", task_avg(s->keys[0], s->n));
        dict_set(out, "sphere/ema_vel", task_avg(s->keys[1], s->n));
        dict_set(out, "sphere/ema_omega", task_avg(s->keys[2], s->n));
        dict_set(out, "sphere/oob", task_avg(s->keys[3], s->n));
        dict_set(out, "sphere/episode_frac", s->n);
    }
    if (log->task[TASK_CUBE].n > 0.0f || (first && task_fracs[TASK_CUBE] > 0.0f)) {
        TaskLog* c = &log->task[TASK_CUBE];
        dict_set(out, "cube/perf", task_avg(c->perf, c->n));
        dict_set(out, "cube/score", task_avg(c->score, c->n));
        dict_set(out, "cube/ema_dist", task_avg(c->keys[0], c->n));
        dict_set(out, "cube/ema_vel", task_avg(c->keys[1], c->n));
        dict_set(out, "cube/ema_omega", task_avg(c->keys[2], c->n));
        dict_set(out, "cube/oob", task_avg(c->keys[3], c->n));
        dict_set(out, "cube/episode_frac", c->n);
    }
    if (log->task[TASK_FLAG].n > 0.0f || (first && task_fracs[TASK_FLAG] > 0.0f)) {
        TaskLog* f = &log->task[TASK_FLAG];
        dict_set(out, "flag/perf", task_avg(f->perf, f->n));
        dict_set(out, "flag/score", task_avg(f->score, f->n));
        dict_set(out, "flag/ema_dist", task_avg(f->keys[0], f->n));
        dict_set(out, "flag/ema_vel", task_avg(f->keys[1], f->n));
        dict_set(out, "flag/ema_omega", task_avg(f->keys[2], f->n));
        dict_set(out, "flag/oob", task_avg(f->keys[3], f->n));
        dict_set(out, "flag/episode_frac", f->n);
    }

    first = 0;
    dict_set(out, "n", log->n);
}

// Rendering (defines Client, puf_render, close_client)
#include "render.h"
