#include "drone.h"

#define RACE_OOB_SCALE 2.0f

#define RACE_RING_MIN_DIST (5.0f * RING_RADIUS)
#define RACE_RING_MAX_DIST 8.0f
#define RACE_RING_SEPARATION (3.0f * RING_RADIUS)
#define RACE_MAX_PLACE_ATTEMPTS 100
#define RACE_MAX_TRACK_ATTEMPTS 16

// types

typedef struct {
    int max_rings;
    float ring_reward;
    float alpha_dist;
    int horizon;
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

static inline bool ring_overlaps(const Target* rings, int count, Vec3 pos) {
    for (int i = 0; i < count; i++)
        if (norm3(sub3(rings[i].pos, pos)) < RACE_RING_SEPARATION) return true;
    return false;
}

static inline bool in_gap_band(Vec3 a, Vec3 b) {
    float d = norm3(sub3(a, b));
    return d >= RACE_RING_MIN_DIST && d <= RACE_RING_MAX_DIST;
}

static inline Target gen_next_ring(unsigned int* rng, const Target* rings, int count,
                                   const Target* close) {
    const Target* prev = &rings[count - 1];
    Target best = rndring(rng, RING_RADIUS);
    bool have_fallback = false;
    for (int attempt = 0; attempt < RACE_MAX_PLACE_ATTEMPTS; attempt++) {
        Target ring = rndring(rng, RING_RADIUS);
        if (!in_gap_band(ring.pos, prev->pos)) continue;
        if (ring_overlaps(rings, count, ring.pos)) continue;
        if (!have_fallback) { best = ring; have_fallback = true; }
        if (close != NULL && !in_gap_band(ring.pos, close->pos)) continue;
        return ring;
    }
    return best;
}

static inline Vec3 path_normal(const Target* rings, int n, int i) {
    if (n < 2) return (Vec3){0.0f, 0.0f, 1.0f};
    Vec3 dir = sub3(rings[(i + 1) % n].pos, rings[(i - 1 + n) % n].pos);
    float len = norm3(dir);
    return len > 1e-6f ? scalmul3(dir, 1.0f / len) : (Vec3){0.0f, 0.0f, 1.0f};
}

static inline void center_rings(Target* rings, int n) {
    Vec3 lo = rings[0].pos, hi = rings[0].pos;
    for (int i = 1; i < n; i++) {
        lo.x = fminf(lo.x, rings[i].pos.x); hi.x = fmaxf(hi.x, rings[i].pos.x);
        lo.y = fminf(lo.y, rings[i].pos.y); hi.y = fmaxf(hi.y, rings[i].pos.y);
        lo.z = fminf(lo.z, rings[i].pos.z); hi.z = fmaxf(hi.z, rings[i].pos.z);
    }
    Vec3 mid = scalmul3(add3(lo, hi), 0.5f);
    for (int i = 0; i < n; i++) rings[i].pos = sub3(rings[i].pos, mid);
}

static inline int check_ring(Vec3 pos, Vec3 prev_pos, Target* ring) {
    float prev_dot = dot3(sub3(prev_pos, ring->pos), ring->normal);
    float new_dot = dot3(sub3(pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    if (valid_dir || invalid_dir) {
        Vec3 dir = sub3(pos, prev_pos);
        float denom = dot3(ring->normal, dir);
        if (fabsf(denom) < 1e-9f) return 0;

        float t = -prev_dot / denom;
        Vec3 intersection = add3(prev_pos, scalmul3(dir, t));
        float d = norm3(sub3(intersection, ring->pos));

        // margins scale with radius
        float margin = 0.1f * ring->radius;
        if (d < (ring->radius - margin) && valid_dir) return 1;
        if (d < ring->radius + margin) return -1;
    }
    return 0;
}

// callbacks

static void race_env_reset(DroneEnv* env) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;

    // regenerate the track when the closing segment (last ring back to ring 0) falls outside the gap band,
    // which the per-ring fallback allowed for about one track in nine
    for (int attempt = 0; attempt < RACE_MAX_TRACK_ATTEMPTS; attempt++) {
        state->ring_buffer[0] = rndring(&env->rng, RING_RADIUS);
        for (int i = 1; i < cfg->max_rings; i++) {
            const Target* close = (i == cfg->max_rings - 1) ? &state->ring_buffer[0] : NULL;
            state->ring_buffer[i] = gen_next_ring(&env->rng, state->ring_buffer, i, close);
        }
        if (in_gap_band(state->ring_buffer[cfg->max_rings - 1].pos, state->ring_buffer[0].pos)) break;
    }

    center_rings(state->ring_buffer, cfg->max_rings);

    for (int i = 0; i < cfg->max_rings; i++) {
        state->ring_buffer[i].normal = path_normal(state->ring_buffer, cfg->max_rings, i);
    }
}

static void race_reset(DroneEnv* env, Drone* agent, int idx) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;

    int g = (int)(rand_r(&env->rng) % cfg->max_rings);
    Target* gate = &state->ring_buffer[g];

    float back = rndf(1.0f, 3.0f, &env->rng);
    Vec3 pos = sub3(gate->pos, scalmul3(gate->normal, back));
    pos = add3(pos, (Vec3){rndf(-0.3f, 0.3f, &env->rng), rndf(-0.3f, 0.3f, &env->rng),
                           rndf(-0.3f, 0.3f, &env->rng)});
    agent->state.pos = (Vec3){
        clampf(pos.x, -MARGIN_X, MARGIN_X),
        clampf(pos.y, -MARGIN_Y, MARGIN_Y),
        clampf(pos.z, -MARGIN_Z, MARGIN_Z),
    };

    state->ring_idx[idx] = g;
    state->rings_passed[idx] = 0;
    state->collisions[idx] = 0.0f;
    *agent->target = *gate;
}

static float race_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;

    float reward = cfg->alpha_dist * (cache->prev_dist - cache->dist);

    int result = check_ring(agent->state.pos, agent->prev_pos, &state->ring_buffer[state->ring_idx[idx]]);
    if (result == 1) {
        state->rings_passed[idx]++;
        state->ring_idx[idx] = (state->ring_idx[idx] + 1) % cfg->max_rings;
        *agent->target = state->ring_buffer[state->ring_idx[idx]];
        reward += cfg->ring_reward;
    } else if (result == -1) {
        state->collisions[idx] += 1.0f;
    }
    if (out_of_bounds(agent->state.pos, RACE_OOB_SCALE)) reward -= env->oob_penalty; // the death step

    return reward;
}

static bool race_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    return out_of_bounds(agent->state.pos, RACE_OOB_SCALE) || agent->episode_length >= cfg->horizon;
}

static void race_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    TaskLog* t = &log->task[TASK_RACE];
    t->n += 1.0f;
    t->perf += fminf((float)state->rings_passed[idx] / (float)cfg->max_rings, 1.0f);
    t->score += (float)state->rings_passed[idx];
    t->keys[0] += (float)state->rings_passed[idx];
    t->keys[1] += state->collisions[idx];
    t->keys[2] += state->rings_passed[idx] >= cfg->max_rings ? 1.0f : 0.0f;
    t->keys[3] += out_of_bounds(agent->state.pos, RACE_OOB_SCALE) ? 1.0f : 0.0f;
}
