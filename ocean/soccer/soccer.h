#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#include "raylib.h"

#define WALL_COLOR (Color){30, 30, 30, 255}
#define JERSEYS 1
#define NUM_TEAMS 2
#define PLAYERS_PER_SIDE 5
#define NUM_PLAYERS (NUM_TEAMS * PLAYERS_PER_SIDE)
#define ARENA_SCALE (1.125f * sqrtf((float)PLAYERS_PER_SIDE))
#define FIELD_WIDTH (240.0f * ARENA_SCALE)
#define FIELD_HEIGHT (135.0f * ARENA_SCALE)
#define ARENA_DIAGONAL sqrtf(FIELD_WIDTH * FIELD_WIDTH + FIELD_HEIGHT * FIELD_HEIGHT)
#define GOAL_DEPTH (10.0f * ARENA_SCALE)
#define GOAL_Y_MIN (45.0f * ARENA_SCALE)
#define GOAL_Y_MAX (90.0f * ARENA_SCALE)
#define GOAL_LIMIT 2
#define PLAYER_RADIUS 18.0f
#define BALL_RADIUS 8.0f
#define PLAYER_MASS 2.4f
#define BALL_MASS 0.35f
#define MAX_SPEED 70.0f
#define MIN_SPEED -50.0f
#define MAX_BALL_SPEED (2.0f * MAX_SPEED)
#define MATCH_WIN_REWARD 1.0f
#define MATCH_LOSS_REWARD -1.0f
#define MAX_BALL_BOUNCES 100
#define BALL_FEATURES 4
#define CONTEXT_FEATURES 5
#define PLAYER_FEATURES 6
#define RELATIONAL_FEATURES 6
#define BASE_OBS_SIZE (BALL_FEATURES + CONTEXT_FEATURES + NUM_PLAYERS * PLAYER_FEATURES)
#define OBS_SIZE (BASE_OBS_SIZE + PLAYERS_PER_SIDE * RELATIONAL_FEATURES)
#define NUM_ATNS (2 * PLAYERS_PER_SIDE)
#define ACT_SIZES {3, 3, 3, 3, 3, 3, 3, 3, 3, 3} // Update with NUM_ATNS and PLAYERS_PER_SIDE

typedef float obs_t;

struct Log {
    float perf;
    float score;
    float episode_return;
    float draw_rate;
    float timeout_rate;
    float timeout_score_decision_rate;
    float timeout_tied_rate;
    float slot_0_goals;
    float slot_1_goals;
    float player_bounces;
    float slot_0_ball_bounces;
    float slot_1_ball_bounces;
    float episode_length;
    float n;
};

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float speed;
    float heading;
} Player;

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
} Ball;

enum {
    GOAL_NONE = -1,
    GOAL_SLOT_0 = 0,
    GOAL_SLOT_1 = 1
};

struct Env {
    Log log;

    Agent agents[NUM_TEAMS];

    int num_agents;
    unsigned int rng;

    int tick;
    int max_steps;
    int frameskip;
    int render;
    long global_agent_steps;
    int global_agents;

    float accel;
    float turn_rate;
    float player_friction;
    float ball_friction;
    float restitution;
    float dt;

    float reward_goal;
    float reward_ball_progress;
    float reward_timeout_ball_position;
    long timeout_ball_reward_anneal_start;
    long timeout_ball_reward_anneal_end;

    Player players[NUM_PLAYERS];
    Ball ball;
    int scores[NUM_TEAMS];
    int match_ball_bounces[NUM_TEAMS];
    float match_return;

    // Shared PufferLib self-play contract: tag 0 is live/live; tag 1 is
    // live policy in slot 0 versus the single frozen bank in slot 1.
    int tag;
    int boundary_reached;
};

static inline float clamp(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static inline float wrap_angle(float angle) {
    while (angle <= -PI) angle += 2.0f * PI;
    while (angle > PI) angle -= 2.0f * PI;
    return angle;
}

static inline float randf(Env* env, float lo, float hi) {
    return lo + (float)rand_r(&env->rng) /
        ((float)RAND_MAX + 1.0f) * (hi - lo);
}

void init(Env* env) {
    env->log = (Log){0};
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents");
    env->render = (int)dict_get(kwargs, "render");
    env->max_steps = (int)dict_get(kwargs, "max_steps");
    env->frameskip = (int)dict_get(kwargs, "frameskip");
    env->accel = (float)dict_get(kwargs, "accel");
    env->turn_rate = (float)dict_get(kwargs, "turn_rate");
    env->player_friction = (float)dict_get(kwargs, "player_friction");
    env->ball_friction = (float)dict_get(kwargs, "ball_friction");
    env->restitution = (float)dict_get(kwargs, "restitution");
    env->dt = (float)dict_get(kwargs, "dt");
    env->reward_goal = (float)dict_get(kwargs, "reward_goal");
    env->reward_ball_progress = (float)dict_get(kwargs, "reward_ball_progress");
    env->reward_timeout_ball_position = (float)dict_get(kwargs, "reward_timeout_ball_position");
    env->global_agents = (int)dict_get(kwargs, "global_agents");
    env->timeout_ball_reward_anneal_start =
        (long)dict_get(kwargs, "timeout_ball_reward_anneal_start");
    env->timeout_ball_reward_anneal_end =
        (long)dict_get(kwargs, "timeout_ball_reward_anneal_end");

    init(env);
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        env->agents[agent_idx].policy = agent_idx;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "policy_0_score", log->perf);
    dict_set(out, "policy_1_score", 1.0f - log->perf);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "timeout_score_decision_rate",
        log->timeout_score_decision_rate);
    dict_set(out, "timeout_tied_rate", log->timeout_tied_rate);
    dict_set(out, "slot_0_goals", log->slot_0_goals);
    dict_set(out, "slot_1_goals", log->slot_1_goals);
    dict_set(out, "total_goals", log->slot_0_goals + log->slot_1_goals);
    dict_set(out, "goals_per_1000_steps",
        1000.0f * (log->slot_0_goals + log->slot_1_goals) /
            log->episode_length);
    dict_set(out, "ball_bounces",
        log->slot_0_ball_bounces + log->slot_1_ball_bounces);
    dict_set(out, "slot_0_ball_bounces", log->slot_0_ball_bounces);
    dict_set(out, "slot_1_ball_bounces", log->slot_1_ball_bounces);
    dict_set(out, "player_bounces", log->player_bounces);
    dict_set(out, "n", log->n);
}

void add_log(Env* env, int outcome_slot) {
    if (outcome_slot == GOAL_SLOT_0) {
        env->log.score += 1.0f;
        env->log.perf += 1.0f;
    } else if (outcome_slot == GOAL_SLOT_1) {
        env->log.score -= 1.0f;
    } else {
        env->log.perf += 0.5f;
        env->log.draw_rate += 1.0f;
    }
    env->log.episode_return += 2.0f * env->match_return;
    env->log.episode_length += (float)env->tick;
    env->log.n += 1.0f;
}

static void add_agent_reward(Env* env, int team, float reward) {
    if (team == 0) env->match_return += reward;
    *env->agents[team].rewards += reward;
}

static inline void set_player_velocity_from_speed(Player* player) {
    player->vx = player->speed * cosf(player->heading);
    player->vy = player->speed * sinf(player->heading);
}

static bool collide(
    float* ax, float* ay, float* avx, float* avy, float am,
    float* bx, float* by, float* bvx, float* bvy, float bm,
    float ar, float br, float restitution
) {
    float dx = *bx - *ax;
    float dy = *by - *ay;
    float d2 = dx * dx + dy * dy;
    float min_dist = ar + br;
    float min2 = min_dist * min_dist;

    if (d2 >= min2) return false;

    float d = sqrtf(d2);
    if (d < 1e-5f) d = 1e-5f;
    float nx = dx / d;
    float ny = dy / d;
    float overlap = min_dist - d;

    float inv_mass_a = 1.0f / am;
    float inv_mass_b = 1.0f / bm;
    float inv_mass_sum = inv_mass_a + inv_mass_b;

    *ax -= overlap * (inv_mass_a / inv_mass_sum) * nx;
    *ay -= overlap * (inv_mass_a / inv_mass_sum) * ny;
    *bx += overlap * (inv_mass_b / inv_mass_sum) * nx;
    *by += overlap * (inv_mass_b / inv_mass_sum) * ny;

    float rvx = *bvx - *avx;
    float rvy = *bvy - *avy;
    float rv_along_normal = rvx * nx + rvy * ny;
    if (rv_along_normal >= 0.0f) return false;

    float impulse = -(1.0f + restitution) * rv_along_normal / inv_mass_sum;
    float ix = impulse * nx;
    float iy = impulse * ny;

    *avx -= ix * inv_mass_a;
    *avy -= iy * inv_mass_a;
    *bvx += ix * inv_mass_b;
    *bvy += iy * inv_mass_b;
    return true;
}

static void reset_rally_state(Env* env) {
    float half_w = FIELD_WIDTH * 0.5f;
    float spawn_x_min = PLAYER_RADIUS * 2.0f;
    float spawn_x_max = half_w - PLAYER_RADIUS * 2.0f;
    float spawn_y_min = PLAYER_RADIUS + 8.0f;
    float spawn_y_max = FIELD_HEIGHT - PLAYER_RADIUS - 8.0f;

    static const float fallback_x[PLAYERS_PER_SIDE] = {
        0.18f, 0.48f, 0.48f, 0.78f, 0.78f
    };
    static const float fallback_y[PLAYERS_PER_SIDE] = {
        0.50f, 0.22f, 0.78f, 0.32f, 0.68f
    };
    float min_spawn_separation = 2.15f * PLAYER_RADIUS;
    float min_spawn_separation2 = min_spawn_separation * min_spawn_separation;

    for (int player = 0; player < PLAYERS_PER_SIDE; player++) {
        float base_x = spawn_x_min;
        float base_y = spawn_y_min;
        bool placed = false;
        for (int attempt = 0; attempt < 64 && !placed; attempt++) {
            base_x = randf(env, spawn_x_min, spawn_x_max);
            base_y = randf(env, spawn_y_min, spawn_y_max);
            placed = true;
            for (int other = 0; other < player; other++) {
                float dx = base_x - env->players[other].x;
                float dy = base_y - env->players[other].y;
                if (dx * dx + dy * dy < min_spawn_separation2) {
                    placed = false;
                    break;
                }
            }
        }
        if (!placed) {
            base_x = spawn_x_min + fallback_x[player] * (spawn_x_max - spawn_x_min);
            base_y = spawn_y_min + fallback_y[player] * (spawn_y_max - spawn_y_min);
        }

        float base_h = randf(env, -PI, PI);
        float base_v = randf(env, 0.0f, MAX_SPEED * 0.2f);
        int team_a_player = player;
        int team_b_player = PLAYERS_PER_SIDE + player;

        env->players[team_a_player].x = base_x;
        env->players[team_a_player].y = base_y;
        env->players[team_a_player].speed = base_v;
        env->players[team_a_player].heading = base_h;
        set_player_velocity_from_speed(&env->players[team_a_player]);

        env->players[team_b_player].x = FIELD_WIDTH - base_x;
        env->players[team_b_player].y = base_y;
        env->players[team_b_player].speed = base_v;
        env->players[team_b_player].heading = wrap_angle(base_h + PI);
        set_player_velocity_from_speed(&env->players[team_b_player]);
    }

    env->ball.x = FIELD_WIDTH * 0.5f;
    env->ball.y = FIELD_HEIGHT * 0.5f;
    env->ball.vx = 0.0f;
    env->ball.vy = 0.0f;
}

static void compute_observations(Env* env) {
    float time = (float)env->tick / (float)env->max_steps;

    for (int team = 0; team < env->num_agents; team++) {
        obs_t* obs = (obs_t*)env->agents[team].observations;
        int opponent = 1 - team;
        float flip = team == 0 ? 1.0f : -1.0f;
        int idx = 0;
        obs[idx++] = flip * (2.0f * env->ball.x / FIELD_WIDTH - 1.0f);
        obs[idx++] = flip * (2.0f * env->ball.y / FIELD_HEIGHT - 1.0f);
        obs[idx++] = flip * env->ball.vx / MAX_SPEED;
        obs[idx++] = flip * env->ball.vy / MAX_SPEED;
        obs[idx++] = time;
        obs[idx++] = (float)env->scores[team] / GOAL_LIMIT;
        obs[idx++] = (float)env->scores[opponent] / GOAL_LIMIT;
        obs[idx++] = (float)env->match_ball_bounces[team] / MAX_BALL_BOUNCES;
        obs[idx++] = (float)env->match_ball_bounces[opponent] / MAX_BALL_BOUNCES;

        for (int side = 0; side < NUM_TEAMS; side++) {
            int owner = side == 0 ? team : opponent;
            for (int i = 0; i < PLAYERS_PER_SIDE; i++) {
                Player* player = &env->players[owner * PLAYERS_PER_SIDE + i];
                obs[idx++] = flip * (2.0f * player->x / FIELD_WIDTH - 1.0f);
                obs[idx++] = flip * (2.0f * player->y / FIELD_HEIGHT - 1.0f);
                obs[idx++] = flip * player->vx / MAX_SPEED;
                obs[idx++] = flip * player->vy / MAX_SPEED;
                obs[idx++] = flip * cosf(player->heading);
                obs[idx++] = flip * sinf(player->heading);
            }
        }

        float goal_x = team == 0 ? FIELD_WIDTH : 0.0f;
        float goal_y = 0.5f * (GOAL_Y_MIN + GOAL_Y_MAX);
        for (int i = 0; i < PLAYERS_PER_SIDE; i++) {
            Player* player = &env->players[team * PLAYERS_PER_SIDE + i];
            float c = cosf(player->heading);
            float s = sinf(player->heading);
            float dx = env->ball.x - player->x;
            float dy = env->ball.y - player->y;
            obs[idx++] = (c * dx + s * dy) / ARENA_DIAGONAL;
            obs[idx++] = (-s * dx + c * dy) / ARENA_DIAGONAL;
            obs[idx++] = sqrtf(dx * dx + dy * dy) / ARENA_DIAGONAL;

            dx = goal_x - player->x;
            dy = goal_y - player->y;
            obs[idx++] = (c * dx + s * dy) / ARENA_DIAGONAL;
            obs[idx++] = (-s * dx + c * dy) / ARENA_DIAGONAL;
            obs[idx++] = sqrtf(dx * dx + dy * dy) / ARENA_DIAGONAL;
        }
    }
}

void puf_reset(Env* env) {
    env->tick = 0;
    env->match_return = 0.0f;
    env->scores[0] = 0;
    env->scores[1] = 0;
    env->match_ball_bounces[0] = 0;
    env->match_ball_bounces[1] = 0;
    reset_rally_state(env);
    compute_observations(env);
}

static void end_episode(Env* env, int outcome_slot) {
    if (outcome_slot == GOAL_SLOT_0) {
        add_agent_reward(env, 0, MATCH_WIN_REWARD);
        add_agent_reward(env, 1, MATCH_LOSS_REWARD);
    } else if (outcome_slot == GOAL_SLOT_1) {
        add_agent_reward(env, 1, MATCH_WIN_REWARD);
        add_agent_reward(env, 0, MATCH_LOSS_REWARD);
    }
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        *env->agents[agent_idx].terminals = 1.0f;
    }
    if (env->tag > 0) env->boundary_reached = 1;
    add_log(env, outcome_slot);
    puf_reset(env);
}

void puf_step(Env* env) {
    int repeats = env->frameskip;
    env->global_agent_steps += env->global_agents;

    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        *env->agents[agent_idx].rewards = 0.0f;
        *env->agents[agent_idx].terminals = 0.0f;
    }

    for (int t = 0; t < repeats; t++) {
        float prev_ball_x = env->ball.x;
        env->tick += 1;

        for (int team = 0; team < env->num_agents; team++) {
            float* actions = env->agents[team].actions;
            for (int i = 0; i < PLAYERS_PER_SIDE; i++) {
                Player* player = &env->players[team * PLAYERS_PER_SIDE + i];
                int steer = (int)actions[2 * i] - 1;
                int throttle = (int)actions[2 * i + 1] - 1;
                float speed_scale = fabsf(player->speed) / MAX_SPEED;
                float turn_scale = 0.25f + 0.75f * speed_scale;
                player->heading = wrap_angle(player->heading +
                    (float)steer * env->turn_rate * turn_scale * env->dt);
                player->speed = clamp(player->speed +
                    (float)throttle * env->accel * env->dt,
                    MIN_SPEED, MAX_SPEED);
                player->speed *= 1.0f - env->player_friction * env->dt;
                set_player_velocity_from_speed(player);
                player->x += player->vx * env->dt;
                player->y += player->vy * env->dt;
            }
        }

        for (int a = 0; a < NUM_PLAYERS; a++) {
            for (int b = a + 1; b < NUM_PLAYERS; b++) {
                Player* player_a = &env->players[a];
                Player* player_b = &env->players[b];
                if (collide(
                        &player_a->x, &player_a->y, &player_a->vx, &player_a->vy, PLAYER_MASS,
                        &player_b->x, &player_b->y, &player_b->vx, &player_b->vy, PLAYER_MASS,
                        PLAYER_RADIUS, PLAYER_RADIUS, env->restitution)) {
                    player_a->speed = clamp(
                        player_a->vx * cosf(player_a->heading) +
                        player_a->vy * sinf(player_a->heading), MIN_SPEED, MAX_SPEED);
                    set_player_velocity_from_speed(player_a);
                    player_b->speed = clamp(
                        player_b->vx * cosf(player_b->heading) +
                        player_b->vy * sinf(player_b->heading), MIN_SPEED, MAX_SPEED);
                    set_player_velocity_from_speed(player_b);
                    env->log.player_bounces += 1.0f;
                }
            }
        }
        for (int player_idx = 0; player_idx < NUM_PLAYERS; player_idx++) {
            Player* player = &env->players[player_idx];
            if (!collide(
                    &player->x, &player->y, &player->vx, &player->vy, PLAYER_MASS,
                    &env->ball.x, &env->ball.y, &env->ball.vx, &env->ball.vy,
                    BALL_MASS, PLAYER_RADIUS, BALL_RADIUS,
                    env->restitution)) continue;
            player->speed = clamp(
                player->vx * cosf(player->heading) +
                player->vy * sinf(player->heading), MIN_SPEED, MAX_SPEED);
            set_player_velocity_from_speed(player);
            float ball_speed = sqrtf(
                env->ball.vx * env->ball.vx + env->ball.vy * env->ball.vy);
            if (ball_speed > MAX_BALL_SPEED) {
                env->ball.vx *= MAX_BALL_SPEED / ball_speed;
                env->ball.vy *= MAX_BALL_SPEED / ball_speed;
            }
            int team = player_idx / PLAYERS_PER_SIDE;
            if (env->match_ball_bounces[team] < MAX_BALL_BOUNCES) {
                env->match_ball_bounces[team] += 1;
            }
            if (team == 0) env->log.slot_0_ball_bounces += 1.0f;
            else env->log.slot_1_ball_bounces += 1.0f;
        }

        env->ball.vx *= 1.0f - env->ball_friction * env->dt;
        env->ball.vy *= 1.0f - env->ball_friction * env->dt;
        env->ball.x += env->ball.vx * env->dt;
        env->ball.y += env->ball.vy * env->dt;

        for (int player_idx = 0; player_idx < NUM_PLAYERS; player_idx++) {
            Player* player = &env->players[player_idx];
            player->x = clamp(player->x, PLAYER_RADIUS, FIELD_WIDTH - PLAYER_RADIUS);
            player->y = clamp(player->y, PLAYER_RADIUS, FIELD_HEIGHT - PLAYER_RADIUS);
        }

        int scorer = GOAL_NONE;
        if (env->ball.y >= GOAL_Y_MIN && env->ball.y <= GOAL_Y_MAX) {
            if (env->ball.x - BALL_RADIUS <= 0.0f) scorer = GOAL_SLOT_1;
            else if (env->ball.x + BALL_RADIUS >= FIELD_WIDTH) {
                scorer = GOAL_SLOT_0;
            }
        }
        float final_ball_position = 0.0f;
        bool goal_scored = scorer != GOAL_NONE;
        if (goal_scored) {
            env->scores[scorer] += 1;
            add_agent_reward(env, scorer, env->reward_goal);
            add_agent_reward(env, 1 - scorer, -env->reward_goal);
            if (scorer == GOAL_SLOT_0) env->log.slot_0_goals += 1.0f;
            else env->log.slot_1_goals += 1.0f;
            if (env->scores[scorer] == GOAL_LIMIT) {
                end_episode(env, scorer);
                return;
            } else {
                reset_rally_state(env);
            }
        } else {
            if (env->ball.x < BALL_RADIUS) {
                env->ball.x = BALL_RADIUS;
                if (env->ball.vx < 0.0f) {
                    env->ball.vx = -env->ball.vx * env->restitution;
                }
            } else if (env->ball.x > FIELD_WIDTH - BALL_RADIUS) {
                env->ball.x = FIELD_WIDTH - BALL_RADIUS;
                if (env->ball.vx > 0.0f) {
                    env->ball.vx = -env->ball.vx * env->restitution;
                }
            }
            if (env->ball.y < BALL_RADIUS) {
                env->ball.y = BALL_RADIUS;
                if (env->ball.vy < 0.0f) {
                    env->ball.vy = -env->ball.vy * env->restitution;
                }
            } else if (env->ball.y > FIELD_HEIGHT - BALL_RADIUS) {
                env->ball.y = FIELD_HEIGHT - BALL_RADIUS;
                if (env->ball.vy > 0.0f) {
                    env->ball.vy = -env->ball.vy * env->restitution;
                }
            }
            final_ball_position = 2.0f * env->ball.x / FIELD_WIDTH - 1.0f;
        }

        if (!goal_scored) {
            float ball_progress = (env->ball.x - prev_ball_x) /
                FIELD_WIDTH;
            float progress_reward = env->reward_ball_progress * ball_progress;
            add_agent_reward(env, 0, progress_reward);
            add_agent_reward(env, 1, -progress_reward);
        }

        if (env->tick >= env->max_steps) {
            int outcome_slot = env->scores[0] > env->scores[1] ? GOAL_SLOT_0 :
                (env->scores[1] > env->scores[0] ? GOAL_SLOT_1 : GOAL_NONE);
            env->log.timeout_rate += 1.0f;
            if (outcome_slot != GOAL_NONE) {
                env->log.timeout_score_decision_rate += 1.0f;
            } else {
                env->log.timeout_tied_rate += 1.0f;
                long start = env->timeout_ball_reward_anneal_start;
                long end = env->timeout_ball_reward_anneal_end;
                float reward_scale = clamp(
                    (float)(end - env->global_agent_steps) / (float)(end - start), 0.0f, 1.0f);
                float position_reward = env->reward_timeout_ball_position *
                    reward_scale * final_ball_position;
                add_agent_reward(env, 0, position_reward);
                add_agent_reward(env, 1, -position_reward);
            }
            end_episode(env, outcome_slot);
            return;
        }
    }
    compute_observations(env);
}

void puf_close(Env* env) {
}

void puf_render(Env* env) {
    if (!env->render) return;
    if (!IsWindowReady()) {
        InitWindow((int)FIELD_WIDTH, (int)FIELD_HEIGHT, "PufferLib Soccer");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    int width = (int)FIELD_WIDTH;
    int height = (int)FIELD_HEIGHT;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawLine(0, 0, 0, height, WALL_COLOR);
    DrawLine(width - 1, 0, width - 1, height, WALL_COLOR);
    DrawLine(0, 0, width, 0, WALL_COLOR);
    DrawLine(0, height - 1, width, height - 1, WALL_COLOR);

    DrawLine(width / 2, 0, width / 2, height, (Color){58, 58, 58, 150});

    DrawRectangleLinesEx(
        (Rectangle){0, GOAL_Y_MIN, GOAL_DEPTH,
            GOAL_Y_MAX - GOAL_Y_MIN},
        2.0f, WHITE
    );
    DrawRectangleLinesEx(
        (Rectangle){FIELD_WIDTH - GOAL_DEPTH, GOAL_Y_MIN,
            GOAL_DEPTH, GOAL_Y_MAX - GOAL_Y_MIN},
        2.0f, WHITE
    );

    for (int player_idx = 0; player_idx < NUM_PLAYERS; player_idx++) {
        Player* player = &env->players[player_idx];
        Color color = player_idx < PLAYERS_PER_SIDE ? RED : BLUE;
        DrawCircle((int)player->x, (int)player->y, PLAYER_RADIUS, color);
        if (JERSEYS && player_idx < PLAYERS_PER_SIDE) {
            float arm = 0.28f * PLAYER_RADIUS;
            float span = 0.85f * PLAYER_RADIUS;
            float offset = 0.25f * PLAYER_RADIUS;
            float degrees = player->heading * RAD2DEG;
            DrawRectanglePro(
                (Rectangle){player->x, player->y, 2.0f * span, arm},
                (Vector2){span, 0.5f * arm}, degrees, DARKBLUE);
            DrawRectanglePro(
                (Rectangle){player->x - offset * cosf(player->heading),
                    player->y - offset * sinf(player->heading), arm, 2.0f * span},
                (Vector2){0.5f * arm, span}, degrees, DARKBLUE);
        } else {
            if (JERSEYS) {
                DrawCircle((int)player->x, (int)player->y, 0.5f * PLAYER_RADIUS, WHITE);
            }
            DrawLine(
                (int)player->x, (int)player->y,
                (int)(player->x + PLAYER_RADIUS * cosf(player->heading)),
                (int)(player->y + PLAYER_RADIUS * sinf(player->heading)),
                WHITE
            );
        }
    }

    DrawCircle((int)env->ball.x, (int)env->ball.y, BALL_RADIUS, WHITE);
    DrawText(TextFormat("Score: %d - %d", env->scores[0], env->scores[1]), 16, 12, 20, WHITE);

    EndDrawing();
}
