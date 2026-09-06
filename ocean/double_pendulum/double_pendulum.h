// Double pendulum swing-up and balance task with discrete cart forces.

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define DP_OBS_SIZE 10
#define DP_ACTIONS 3
#define DP_WIDTH 800
#define DP_HEIGHT 420
#define DP_SCALE 65.0f

#define ACT_SIZES {DP_ACTIONS}
#define OBS_SIZE DP_OBS_SIZE
#define NUM_ATNS 1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float x_threshold_termination;
    float max_steps_termination;
    float invalid_termination;
    float hold_time;
    float best_height;
    float upright_frac;
    float n;
};

struct Env {
    int num_agents;
    unsigned int rng;
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;

    float x;
    float x_dot;
    float theta1;
    float theta1_dot;
    float theta2;
    float theta2_dot;
    int tick;
    float episode_return;
    int upright_steps;
    int max_upright_steps;
    int upright_count;
    float best_height;
    int physics_failed;

    float cart_mass;
    float link1_mass;
    float link2_mass;
    float link1_length;
    float link2_length;
    float gravity;
    float force_mag;
    float dt;
    float x_threshold;
    int max_steps;
};
typedef Env DoublePendulum;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_YELLOW = (Color){245, 197, 66, 255};

static inline float dp_randf(DoublePendulum* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

static inline float wrap_pi(float x) {
    while (x > M_PI) x -= 2.0f * M_PI;
    while (x < -M_PI) x += 2.0f * M_PI;
    return x;
}

static inline float dp_tip_height(DoublePendulum* env) {
    float tip_y = env->link1_length * cosf(env->theta1)
        + env->link2_length * cosf(env->theta2);
    float max_y = env->link1_length + env->link2_length;
    if (max_y < 1e-6f) return 0.0f;
    return 0.5f * (tip_y / max_y + 1.0f);
}

// Point-mass energy matching the EOM. 0 = hanging rest, 1 = upright rest.
static inline float dp_energy_norm(DoublePendulum* env) {
    float m0 = env->cart_mass;
    float m1 = env->link1_mass;
    float m2 = env->link2_mass;
    float l1 = env->link1_length;
    float l2 = env->link2_length;
    float t1 = env->theta1;
    float t2 = env->theta2;
    float w1 = env->theta1_dot;
    float w2 = env->theta2_dot;
    float xd = env->x_dot;
    float g = env->gravity;
    float c1 = cosf(t1);
    float s1 = sinf(t1);
    float c2 = cosf(t2);
    float s2 = sinf(t2);

    float x1d = xd + l1 * c1 * w1;
    float y1d = -l1 * s1 * w1;
    float x2d = x1d + l2 * c2 * w2;
    float y2d = y1d - l2 * s2 * w2;
    float ke = 0.5f * m0 * xd * xd
        + 0.5f * m1 * (x1d * x1d + y1d * y1d)
        + 0.5f * m2 * (x2d * x2d + y2d * y2d);
    float y1 = l1 * c1;
    float y2 = y1 + l2 * c2;
    float pe = m1 * g * y1 + m2 * g * y2;
    float e_hang = -m1 * g * l1 - m2 * g * (l1 + l2);
    float span = 2.0f * g * (m1 * l1 + m2 * (l1 + l2));
    if (span < 1e-6f) return 0.0f;
    return (ke + pe - e_hang) / span;
}

void compute_observations(DoublePendulum* env) {
    float* obs = env->agents[0].observations;
    float xt = env->x_threshold > 1e-6f ? env->x_threshold : 5.0f;
    float height = dp_tip_height(env);
    float energy = dp_energy_norm(env);
    obs[0] = env->x / xt;
    obs[1] = env->x_dot / 10.0f;
    obs[2] = sinf(env->theta1);
    obs[3] = cosf(env->theta1);
    obs[4] = env->theta1_dot / 15.0f;
    obs[5] = sinf(env->theta2);
    obs[6] = cosf(env->theta2);
    obs[7] = env->theta2_dot / 15.0f;
    obs[8] = height;
    obs[9] = energy * 0.5f;
}

void add_log(DoublePendulum* env, bool x_done, bool timeout, bool invalid) {
    float best = fminf(fmaxf(env->best_height, 0.0f), 1.0f);
    float frac = env->tick > 0 ? (float)env->upright_count / (float)env->tick : 0.0f;
    // perf: peak tip height this episode (0 hanging, 1 fully upright).
    env->log.perf += best;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.invalid_termination += invalid ? 1.0f : 0.0f;
    env->log.hold_time += (float)env->max_upright_steps;
    env->log.best_height += best;
    env->log.upright_frac += frac;
    env->log.n += 1.0f;
}

void init(DoublePendulum* env) {
    env->num_agents = 1;
}

void puf_reset(DoublePendulum* env) {
    env->x = dp_randf(env, -0.04f, 0.04f);
    env->x_dot = dp_randf(env, -0.04f, 0.04f);
    env->theta1 = M_PI + dp_randf(env, -0.08f, 0.08f);
    env->theta1_dot = dp_randf(env, -0.04f, 0.04f);
    env->theta2 = M_PI + dp_randf(env, -0.08f, 0.08f);
    env->theta2_dot = dp_randf(env, -0.04f, 0.04f);
    env->tick = 0;
    env->episode_return = 0.0f;
    env->upright_steps = 0;
    env->max_upright_steps = 0;
    env->upright_count = 0;
    env->best_height = 0.0f;
    env->physics_failed = 0;
    compute_observations(env);
}

static int solve_3x3(float A[3][3], float b[3], float x[3]) {
    for (int i = 0; i < 3; i++) {
        int pivot = i;
        float best = fabsf(A[i][i]);
        for (int r = i + 1; r < 3; r++) {
            float v = fabsf(A[r][i]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best < 1e-8f || !isfinite(best)) {
            x[0] = x[1] = x[2] = 0.0f;
            return 0;
        }
        if (pivot != i) {
            for (int c = i; c < 3; c++) {
                float tmp = A[i][c];
                A[i][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
            float tmp = b[i];
            b[i] = b[pivot];
            b[pivot] = tmp;
        }

        float inv = 1.0f / A[i][i];
        if (!isfinite(inv)) {
            x[0] = x[1] = x[2] = 0.0f;
            return 0;
        }
        for (int c = i; c < 3; c++) A[i][c] *= inv;
        b[i] *= inv;
        for (int r = 0; r < 3; r++) {
            if (r == i) continue;
            float f = A[r][i];
            for (int c = i; c < 3; c++) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    x[0] = b[0];
    x[1] = b[1];
    x[2] = b[2];
    if (!isfinite(x[0]) || !isfinite(x[1]) || !isfinite(x[2])) {
        x[0] = x[1] = x[2] = 0.0f;
        return 0;
    }
    return 1;
}

void integrate_physics(DoublePendulum* env, float force) {
    float m0 = env->cart_mass;
    float m1 = env->link1_mass;
    float m2 = env->link2_mass;
    float l1 = env->link1_length;
    float l2 = env->link2_length;
    float t1 = env->theta1;
    float t2 = env->theta2;
    float w1 = env->theta1_dot;
    float w2 = env->theta2_dot;
    float c1 = cosf(t1);
    float c2 = cosf(t2);
    float s1 = sinf(t1);
    float s2 = sinf(t2);
    float c12 = cosf(t1 - t2);
    float s12 = sinf(t1 - t2);

    float A[3][3] = {
        {m0 + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2},
        {(m1 + m2) * l1 * c1, (m1 + m2) * l1 * l1, m2 * l1 * l2 * c12},
        {m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2 * l2},
    };
    A[0][0] += 1e-8f;
    A[1][1] += 1e-8f;
    A[2][2] += 1e-8f;
    float b[3] = {
        force + (m1 + m2) * l1 * s1 * w1 * w1 + m2 * l2 * s2 * w2 * w2,
        (m1 + m2) * env->gravity * l1 * s1 - m2 * l1 * l2 * s12 * w2 * w2,
        m2 * env->gravity * l2 * s2 + m2 * l1 * l2 * s12 * w1 * w1,
    };
    float qdd[3];
    if (!solve_3x3(A, b, qdd)) {
        env->physics_failed = 1;
        return;
    }
    for (int i = 0; i < 3; i++) {
        qdd[i] = fminf(fmaxf(qdd[i], -1.0e5f), 1.0e5f);
    }

    env->x_dot += env->dt * qdd[0];
    env->theta1_dot += env->dt * qdd[1];
    env->theta2_dot += env->dt * qdd[2];
    env->x_dot = fminf(fmaxf(env->x_dot, -20.0f), 20.0f);
    env->theta1_dot = fminf(fmaxf(env->theta1_dot, -30.0f), 30.0f);
    env->theta2_dot = fminf(fmaxf(env->theta2_dot, -30.0f), 30.0f);
    env->x += env->dt * env->x_dot;
    env->theta1 = wrap_pi(env->theta1 + env->dt * env->theta1_dot);
    env->theta2 = wrap_pi(env->theta2 + env->dt * env->theta2_dot);
}

// Non-farmable swing-up: pay only for a new episode-best tip height.
float height_record_reward(DoublePendulum* env) {
    float height = dp_tip_height(env);
    if (!isfinite(height)) height = 0.0f;
    height = fminf(fmaxf(height, 0.0f), 1.0f);

    bool near_top = height > 0.9f;
    bool stable = near_top
        && fabsf(env->theta1_dot) < 1.5f
        && fabsf(env->theta2_dot) < 1.5f
        && fabsf(env->x_dot) < 1.0f;
    if (near_top) env->upright_count += 1;
    if (stable) env->upright_steps += 1;
    else env->upright_steps = 0;
    if (env->upright_steps > env->max_upright_steps) {
        env->max_upright_steps = env->upright_steps;
    }

    float reward = fmaxf(0.0f, height - env->best_height);
    if (height > env->best_height && height > 0.9f) {
        reward += 0.05f * (height - 0.9f) / 0.1f;
    }
    if (stable) {
        reward += 0.1f;
    }
    env->best_height = fmaxf(env->best_height, height);
    return reward;
}

// Hold Left Shift + A/D or arrows.
static void double_pendulum_human_controls(DoublePendulum *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[0] = 0;
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = 2;
    } else {
        env->agents[0].actions[0] = 1;
    }
}

void puf_step(DoublePendulum* env) {
    double_pendulum_human_controls(env);
    float a = env->agents[0].actions[0];
    int action = (int)a;
    if ((unsigned)action >= DP_ACTIONS) action = 1;
    float force = 0.0f;
    if (action == 0) force = -env->force_mag;
    else if (action == 2) force = env->force_mag;

    env->physics_failed = 0;
    integrate_physics(env, force);
    env->tick += 1;

    bool invalid = env->physics_failed
        || !isfinite(env->x) || !isfinite(env->x_dot)
        || !isfinite(env->theta1) || !isfinite(env->theta1_dot)
        || !isfinite(env->theta2) || !isfinite(env->theta2_dot);
    float xt = env->x_threshold > 1e-6f ? env->x_threshold : 5.0f;
    bool x_done = env->x < -xt || env->x > xt;
    int max_steps = env->max_steps > 0 ? env->max_steps : 600;
    bool timeout = env->tick >= max_steps;
    bool done = invalid || x_done || timeout;

    float reward = 0.0f;
    if (!invalid) reward = height_record_reward(env);
    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;
    // Timeout, rail, and NaN/singular are all true terminals (no GAE bootstrap).
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f;

    if (done) {
        add_log(env, x_done && !invalid, timeout && !invalid && !x_done, invalid);
        puf_reset(env);
        return;
    }
    compute_observations(env);
}

void puf_render(DoublePendulum* env) {
    if (!IsWindowReady()) {
        InitWindow(DP_WIDTH, DP_HEIGHT, "PufferLib Double Pendulum");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    double_pendulum_human_controls(env);
    if (!isfinite(env->x) || !isfinite(env->theta1) || !isfinite(env->theta2)) return;

    float rail_y = DP_HEIGHT * 0.72f;
    float cart_x = DP_WIDTH / 2.0f + env->x * DP_SCALE;
    cart_x = fminf(fmaxf(cart_x, 32.0f), DP_WIDTH - 32.0f);
    float cart_y = rail_y - 16.0f;
    float l1 = env->link1_length * 2.0f * DP_SCALE;
    float l2 = env->link2_length * 2.0f * DP_SCALE;
    Vector2 p0 = {cart_x, cart_y};
    Vector2 p1 = {cart_x + sinf(env->theta1) * l1, cart_y - cosf(env->theta1) * l1};
    Vector2 p2 = {p1.x + sinf(env->theta2) * l2, p1.y - cosf(env->theta2) * l2};

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, (int)rail_y, DP_WIDTH, (int)rail_y, PUFF_CYAN);
    DrawRectangle((int)(cart_x - 28), (int)(cart_y - 12), 56, 24, PUFF_CYAN);
    DrawLineEx(p0, p1, 7.0f, PUFF_RED);
    DrawLineEx(p1, p2, 6.0f, PUFF_YELLOW);
    DrawCircleV(p0, 8.0f, PUFF_WHITE);
    DrawCircleV(p1, 8.0f, PUFF_WHITE);
    DrawCircleV(p2, 10.0f, PUFF_WHITE);
    DrawText(TextFormat("steps %d  return %.2f  best %.2f  hold %d",
        env->tick, env->episode_return, env->best_height, env->max_upright_steps),
        20, 20, 20, PUFF_WHITE);
    DrawText(TextFormat("x %.2f  theta1 %.1f  theta2 %.1f",
        env->x, env->theta1 * 180.0f / M_PI, env->theta2 * 180.0f / M_PI),
        20, 48, 20, PUFF_WHITE);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(DoublePendulum* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "invalid_termination", log->invalid_termination);
    dict_set(out, "hold_time", log->hold_time);
    dict_set(out, "best_height", log->best_height);
    dict_set(out, "upright_frac", log->upright_frac);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->cart_mass = dict_get(kwargs, "cart_mass");
    env->link1_mass = dict_get(kwargs, "link1_mass");
    env->link2_mass = dict_get(kwargs, "link2_mass");
    env->link1_length = dict_get(kwargs, "link1_length");
    env->link2_length = dict_get(kwargs, "link2_length");
    env->gravity = dict_get(kwargs, "gravity");
    env->force_mag = dict_get(kwargs, "force_mag");
    env->dt = dict_get(kwargs, "dt");
    env->x_threshold = dict_get(kwargs, "x_threshold");
    env->max_steps = dict_get(kwargs, "max_steps");
    if (env->dt <= 0.0f) env->dt = 0.02f;
    if (env->x_threshold <= 0.0f) env->x_threshold = 5.0f;
    if (env->max_steps <= 0) env->max_steps = 1200;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
}
