#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define SI_NOOP 0
#define SI_LEFT 1
#define SI_RIGHT 2
#define SI_FIRE 3

#define SI_ROWS 5
#define SI_COLS 11
#define SI_NUM_INVADERS (SI_ROWS * SI_COLS)
#define SI_MAX_ENEMY_BULLETS 3

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    int width;
    int height;
} Client;

typedef struct Bullet {
    float x, y;
    int active;
} Bullet;

typedef struct SpaceInvaders {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    // config
    int width;
    int height;
    int frameskip;
    int player_speed;       // px/step (int)
    int player_bullet_speed;
    int enemy_bullet_speed;
    int formation_dx;       // px per formation step
    int formation_dy;       // px per edge drop
    int formation_start_interval; // ticks between steps, scaled by alive ratio
    int enemy_fire_interval;      // base cooldown ticks
    int invader_w;
    int invader_h;
    int invader_spacing_x;
    int invader_spacing_y;
    int formation_margin_x;
    int formation_margin_y;
    int player_w;
    int player_h;
    int player_y_offset;    // from bottom
    int bullet_w;
    int bullet_h;
    int max_lives;

    // state
    int score;
    int lives;
    int tick;
    int formation_dir;      // +1 right, -1 left
    int formation_tick;     // counter toward next step
    int fire_cooldown;
    int num_alive;
    unsigned int rng;
    float episode_return_accum;

    float player_x;
    float formation_x;      // top-left of formation bounding box
    float formation_y;
    float invaders_alive[SI_NUM_INVADERS]; // 1.0 alive, 0.0 dead
    Bullet player_bullet;
    Bullet enemy_bullets[SI_MAX_ENEMY_BULLETS];

    // cached alive-cell grid bounds (derived from invaders_alive; O(1) reads)
    int row_alive[SI_ROWS];
    int col_alive[SI_COLS];
    int min_row, max_row;
    int min_col, max_col;
} SpaceInvaders;

static inline int si_row_points(int row) {
    if (row == 0) return 30;
    if (row <= 2) return 20;
    return 10;
}

static inline float si_invader_x(SpaceInvaders* env, int col) {
    return env->formation_x + col * (env->invader_w + env->invader_spacing_x);
}
static inline float si_invader_y(SpaceInvaders* env, int row) {
    return env->formation_y + row * (env->invader_h + env->invader_spacing_y);
}

static inline int si_player_y(SpaceInvaders* env) {
    return env->height - env->player_y_offset - env->player_h;
}

void init(SpaceInvaders* env) {
    env->tick = 0;
    env->rng = 42;
}

void allocate(SpaceInvaders* env) {
    init(env);
    env->observations = (float*)calloc(9 + SI_NUM_INVADERS + 3 * SI_MAX_ENEMY_BULLETS, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_close(SpaceInvaders* env) {
    // nothing dynamic
    (void)env;
}

void free_allocated(SpaceInvaders* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

void add_log(SpaceInvaders* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return_accum;
    env->log.score += env->score;
    // perf: fraction of invaders cleared
    env->log.perf += (float)(SI_NUM_INVADERS - env->num_alive) / (float)SI_NUM_INVADERS;
    env->log.n += 1;
}

void compute_observations(SpaceInvaders* env) {
    float* o = env->observations;
    int i = 0;
    o[i++] = env->player_x / (float)env->width;
    o[i++] = (float)env->player_bullet.active;
    o[i++] = env->player_bullet.x / (float)env->width;
    o[i++] = env->player_bullet.y / (float)env->height;
    o[i++] = (env->formation_dir > 0) ? 1.0f : 0.0f;
    o[i++] = env->formation_x / (float)env->width;
    o[i++] = env->formation_y / (float)env->height;
    o[i++] = (float)env->num_alive / (float)SI_NUM_INVADERS;
    o[i++] = (float)env->lives / (float)env->max_lives;
    memcpy(o + i, env->invaders_alive, sizeof(float) * SI_NUM_INVADERS);
    i += SI_NUM_INVADERS;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        o[i++] = (float)env->enemy_bullets[b].active;
        o[i++] = env->enemy_bullets[b].x / (float)env->width;
        o[i++] = env->enemy_bullets[b].y / (float)env->height;
    }
}

void reset_formation(SpaceInvaders* env) {
    env->formation_x = env->formation_margin_x;
    env->formation_y = env->formation_margin_y;
    env->formation_dir = 1;
    env->formation_tick = 0;
    env->num_alive = SI_NUM_INVADERS;
    for (int i = 0; i < SI_NUM_INVADERS; i++) env->invaders_alive[i] = 1.0f;
    for (int r = 0; r < SI_ROWS; r++) env->row_alive[r] = SI_COLS;
    for (int c = 0; c < SI_COLS; c++) env->col_alive[c] = SI_ROWS;
    env->min_row = 0; env->max_row = SI_ROWS - 1;
    env->min_col = 0; env->max_col = SI_COLS - 1;
    env->player_bullet.active = 0;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) env->enemy_bullets[b].active = 0;
    env->fire_cooldown = env->enemy_fire_interval;
}

// Recompute grid bounds (called when a boundary row/col empties).
static inline void recompute_grid_bounds(SpaceInvaders* env) {
    if (env->num_alive == 0) {
        // use sentinels; formation_bounds_cached will return "no alive" via num_alive check
        env->min_row = SI_ROWS; env->max_row = -1;
        env->min_col = SI_COLS; env->max_col = -1;
        return;
    }
    int mr = 0; while (mr < SI_ROWS && env->row_alive[mr] == 0) mr++;
    int Mr = SI_ROWS - 1; while (Mr >= 0 && env->row_alive[Mr] == 0) Mr--;
    int mc = 0; while (mc < SI_COLS && env->col_alive[mc] == 0) mc++;
    int Mc = SI_COLS - 1; while (Mc >= 0 && env->col_alive[Mc] == 0) Mc--;
    env->min_row = mr; env->max_row = Mr;
    env->min_col = mc; env->max_col = Mc;
}

// O(1) equivalent of the old formation_bounds using cached grid bounds.
static inline int formation_bounds_cached(SpaceInvaders* env, float* out_min_x,
        float* out_max_x, float* out_max_y) {
    if (env->num_alive == 0) return 0;
    *out_min_x = si_invader_x(env, env->min_col);
    *out_max_x = si_invader_x(env, env->max_col) + env->invader_w;
    *out_max_y = si_invader_y(env, env->max_row) + env->invader_h;
    return 1;
}

void c_reset(SpaceInvaders* env) {
    env->score = 0;
    env->lives = env->max_lives;
    env->tick = 0;
    env->episode_return_accum = 0.0f;
    // Random initial x: without this, starting always in the center makes
    // "stay still and fire" the optimal local minimum and PPO never learns to move.
    int max_px = env->width - env->player_w;
    env->player_x = (float)(rand_r(&env->rng) % (max_px + 1));
    reset_formation(env);
    compute_observations(env);
}

void step_formation(SpaceInvaders* env) {
    // compute current interval scaled by alive ratio: more aggressive as fewer remain
    int interval = env->formation_start_interval * env->num_alive / SI_NUM_INVADERS;
    if (interval < 2) interval = 2;
    env->formation_tick++;
    if (env->formation_tick < interval) return;
    env->formation_tick = 0;

    float min_x, max_x, max_y;
    if (!formation_bounds_cached(env, &min_x, &max_x, &max_y)) return;

    float dx = env->formation_dir * env->formation_dx;
    float new_min = min_x + dx;
    float new_max = max_x + dx;
    if (new_min < 0 || new_max > env->width) {
        // hit wall: drop and reverse
        env->formation_dir *= -1;
        env->formation_y += env->formation_dy;
    } else {
        env->formation_x += dx;
    }
}

// fire an enemy bullet from a random alive bottom invader of a random column
void maybe_enemy_fire(SpaceInvaders* env) {
    if (env->fire_cooldown > 0) { env->fire_cooldown--; return; }
    // find a free slot
    int slot = -1;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) { slot = b; break; }
    }
    if (slot < 0) return;
    // try a few random columns
    for (int attempt = 0; attempt < 6; attempt++) {
        int col = rand_r(&env->rng) % SI_COLS;
        int shooter_row = -1;
        for (int r = SI_ROWS - 1; r >= 0; r--) {
            if (env->invaders_alive[r * SI_COLS + col] > 0.0f) {
                shooter_row = r; break;
            }
        }
        if (shooter_row < 0) continue;
        env->enemy_bullets[slot].x = si_invader_x(env, col) + env->invader_w / 2.0f - env->bullet_w / 2.0f;
        env->enemy_bullets[slot].y = si_invader_y(env, shooter_row) + env->invader_h;
        env->enemy_bullets[slot].active = 1;
        env->fire_cooldown = env->enemy_fire_interval;
        return;
    }
}

static inline int aabb(float ax, float ay, float aw, float ah,
        float bx, float by, float bw, float bh) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

void step_frame(SpaceInvaders* env, int action) {
    // player
    if (action == SI_LEFT) env->player_x -= env->player_speed;
    else if (action == SI_RIGHT) env->player_x += env->player_speed;
    if (env->player_x < 0) env->player_x = 0;
    float max_px = env->width - env->player_w;
    if (env->player_x > max_px) env->player_x = max_px;

    // player fire
    if (action == SI_FIRE && !env->player_bullet.active) {
        env->player_bullet.x = env->player_x + env->player_w / 2.0f - env->bullet_w / 2.0f;
        env->player_bullet.y = si_player_y(env) - env->bullet_h;
        env->player_bullet.active = 1;
    }

    // advance player bullet
    if (env->player_bullet.active) {
        env->player_bullet.y -= env->player_bullet_speed;
        if (env->player_bullet.y + env->bullet_h < 0) env->player_bullet.active = 0;
    }

    // advance enemy bullets
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) continue;
        env->enemy_bullets[b].y += env->enemy_bullet_speed;
        if (env->enemy_bullets[b].y > env->height) env->enemy_bullets[b].active = 0;
    }

    // formation movement
    step_formation(env);
    maybe_enemy_fire(env);

    // player bullet hits invader
    if (env->player_bullet.active && env->num_alive > 0) {
        float form_top_y = si_invader_y(env, env->min_row);
        float form_bot_y = si_invader_y(env, env->max_row) + env->invader_h;
        float pb_top = env->player_bullet.y;
        float pb_bot = pb_top + env->bullet_h;
        if (pb_bot > form_top_y && pb_top < form_bot_y) {
            // narrow column range to the (at most 2) columns the bullet x overlaps
            float pb_left = env->player_bullet.x;
            float pb_right = pb_left + env->bullet_w;
            int col_pitch = env->invader_w + env->invader_spacing_x;
            // floor for possibly-negative values; avoid libc floorf call
            float rel_lo = (pb_left - env->formation_x) / (float)col_pitch;
            float rel_hi = (pb_right - env->formation_x) / (float)col_pitch;
            int col_lo = (int)rel_lo - (rel_lo < (int)rel_lo);
            int col_hi = (int)rel_hi - (rel_hi < (int)rel_hi);
            if (col_lo < env->min_col) col_lo = env->min_col;
            if (col_hi > env->max_col) col_hi = env->max_col;
            if (col_lo <= col_hi) {
                for (int c = col_lo; c <= col_hi; c++) {
                    if (env->col_alive[c] == 0) continue;
                    float ix = si_invader_x(env, c);
                    // x-overlap is implied by col selection; only need to verify
                    // when bullet straddles a gap between columns
                    if (pb_right <= ix || pb_left >= ix + env->invader_w) continue;
                    for (int r = env->min_row; r <= env->max_row; r++) {
                        int idx = r * SI_COLS + c;
                        if (env->invaders_alive[idx] == 0.0f) continue;
                        float iy = si_invader_y(env, r);
                        // y overlap (x already checked)
                        if (pb_bot <= iy || pb_top >= iy + env->invader_h) continue;
                        env->invaders_alive[idx] = 0.0f;
                        env->num_alive--;
                        env->row_alive[r]--;
                        env->col_alive[c]--;
                        if ((r == env->min_row || r == env->max_row) && env->row_alive[r] == 0) {
                            recompute_grid_bounds(env);
                        } else if ((c == env->min_col || c == env->max_col) && env->col_alive[c] == 0) {
                            recompute_grid_bounds(env);
                        }
                        env->player_bullet.active = 0;
                        int pts = si_row_points(r);
                        env->score += pts;
                        float r_add = (float)pts * 0.1f;
                        env->rewards[0] += r_add;
                        env->episode_return_accum += r_add;
                        goto after_player_hit;
                    }
                }
            }
        }
    }
after_player_hit:;

    // enemy bullet hits player
    int player_y = si_player_y(env);
    int player_hit = 0;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) continue;
        if (aabb(env->enemy_bullets[b].x, env->enemy_bullets[b].y, env->bullet_w, env->bullet_h,
                env->player_x, (float)player_y, env->player_w, env->player_h)) {
            env->enemy_bullets[b].active = 0;
            player_hit = 1;
            break;
        }
    }

    // invaders reach player line -> instant loss (O(1) via cached bounds)
    int invaded = 0;
    if (env->num_alive > 0) {
        float fmax_y = si_invader_y(env, env->max_row) + env->invader_h;
        invaded = (fmax_y >= player_y);
    }

    if (player_hit) {
        env->lives--;
        env->rewards[0] -= 1.0f;
        env->episode_return_accum -= 1.0f;
    }

    int cleared = env->num_alive == 0;
    if (cleared) {
        env->rewards[0] += 10.0f;
        env->episode_return_accum += 10.0f;
        reset_formation(env);
    }

    if (env->lives <= 0 || invaded) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(SpaceInvaders* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    int action = (int)env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick++;
        step_frame(env, action);
        if (env->terminals[0]) break;
    }
    compute_observations(env);
}

// ---------- rendering ----------

Client* make_client(SpaceInvaders* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "PufferLib Space Invaders");
    SetTargetFPS(60 / (env->frameskip > 0 ? env->frameskip : 1));
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(SpaceInvaders* env) {
    if (env->client == NULL) env->client = make_client(env);
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){6, 12, 24, 255});

    // player
    int py = si_player_y(env);
    DrawRectangle((int)env->player_x, py, env->player_w, env->player_h, (Color){0, 255, 128, 255});

    // invaders
    Color rowc[5] = {
        (Color){255, 80, 80, 255},
        (Color){255, 170, 80, 255},
        (Color){255, 230, 80, 255},
        (Color){120, 220, 255, 255},
        (Color){200, 120, 255, 255},
    };
    for (int r = 0; r < SI_ROWS; r++) {
        for (int c = 0; c < SI_COLS; c++) {
            int idx = r * SI_COLS + c;
            if (env->invaders_alive[idx] == 0.0f) continue;
            int ix = (int)si_invader_x(env, c);
            int iy = (int)si_invader_y(env, r);
            DrawRectangle(ix, iy, env->invader_w, env->invader_h, rowc[r]);
        }
    }

    // bullets
    if (env->player_bullet.active) {
        DrawRectangle((int)env->player_bullet.x, (int)env->player_bullet.y,
            env->bullet_w, env->bullet_h, WHITE);
    }
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) continue;
        DrawRectangle((int)env->enemy_bullets[b].x, (int)env->enemy_bullets[b].y,
            env->bullet_w, env->bullet_h, (Color){255, 200, 0, 255});
    }

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Lives: %i", env->lives), env->width - 100, 10, 20, WHITE);
    EndDrawing();
}
