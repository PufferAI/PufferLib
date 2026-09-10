#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define SO_OBS_SIZE 73
#define SO_NUM_ACTIONS 6
#define NUM_ALIEN_ROWS 5
#define NUM_ALIEN_COLS 11
#define NUM_ALIENS (NUM_ALIEN_ROWS * NUM_ALIEN_COLS)
#define MAX_ALIEN_BULLETS 3
#define PLAYER_BULLET_W 4.0f
#define PLAYER_BULLET_H 8.0f
#define ALIEN_BULLET_W 4.0f
#define ALIEN_BULLET_H 8.0f
#define TICK_RATE (1.0f / 60.0f)
#define WAVE_INTERVAL_FLOOR 6
#define WAVE_SCORE 990.0f

#define ACTION_NOOP 0
#define ACTION_FIRE 1
#define ACTION_RIGHT 2
#define ACTION_LEFT 3
#define ACTION_RIGHT_FIRE 4
#define ACTION_LEFT_FIRE 5

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int active;
    float x;
    float y;
} Bullet;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int width;
    int height;
    int frameskip;
    int alien_width;
    int alien_height;
    int alien_spacing_x;
    int alien_spacing_y;
    int edge_margin;
    int alien_step_x;
    int alien_step_down;
    int base_move_interval;
    int min_move_interval;
    int player_width;
    int player_height;
    float player_speed;
    float player_bullet_speed;
    float alien_bullet_speed;
    int initial_lives;
    float alien_fire_prob;

    float player_x;
    float player_y;
    int lives;
    int score;
    int wave;
    int tick;
    int alive_count;
    int move_timer;
    int move_interval;
    int direction;
    float formation_x;
    float formation_y;
    int alien_alive[NUM_ALIENS];
    Bullet player_bullet;
    Bullet alien_bullets[MAX_ALIEN_BULLETS];
} SpaceOnslaught;

static float alien_x(SpaceOnslaught* env, int col) {
    return env->formation_x + col * (env->alien_width + env->alien_spacing_x);
}

static float alien_y(SpaceOnslaught* env, int row) {
    return env->formation_y + row * (env->alien_height + env->alien_spacing_y);
}

static int rects_overlap(float x1, float y1, float w1, float h1,
        float x2, float y2, float w2, float h2) {
    return x1 < x2 + w2 && x1 + w1 > x2 && y1 < y2 + h2 && y1 + h1 > y2;
}

static int alien_points(int row) {
    if (row == 0) {
        return 30;
    }
    if (row <= 2) {
        return 20;
    }
    return 10;
}

static void update_move_interval(SpaceOnslaught* env) {
    int wave_base = env->base_move_interval - 2 * (env->wave - 1);
    if (wave_base < WAVE_INTERVAL_FLOOR) {
        wave_base = WAVE_INTERVAL_FLOOR;
    }
    int alive = env->alive_count;
    if (alive < 0) {
        alive = 0;
    }
    int interval = 0;
    if (alive > 0) {
        interval = wave_base * alive / NUM_ALIENS;
    }
    if (interval < env->min_move_interval) {
        interval = env->min_move_interval;
    }
    if (interval < 1) {
        interval = 1;
    }
    env->move_interval = interval;
}

static void clear_bullets(SpaceOnslaught* env) {
    env->player_bullet.active = 0;
    env->player_bullet.x = 0;
    env->player_bullet.y = 0;
    for (int i = 0; i < MAX_ALIEN_BULLETS; i++) {
        env->alien_bullets[i].active = 0;
        env->alien_bullets[i].x = 0;
        env->alien_bullets[i].y = 0;
    }
}

static void clamp_player(SpaceOnslaught* env) {
    if (env->player_x < 0) {
        env->player_x = 0;
    }
    float max_x = (float)(env->width - env->player_width);
    if (max_x < 0) {
        max_x = 0;
    }
    if (env->player_x > max_x) {
        env->player_x = max_x;
    }
}

static void reset_player(SpaceOnslaught* env) {
    env->player_x = (env->width - env->player_width) * 0.5f;
    env->player_y = (float)(env->height - env->player_height - 24);
    if (env->player_y < 0) {
        env->player_y = 0;
    }
    clamp_player(env);
}

static void reset_wave(SpaceOnslaught* env) {
    int formation_w = NUM_ALIEN_COLS * env->alien_width
        + (NUM_ALIEN_COLS - 1) * env->alien_spacing_x;
    env->formation_x = (env->width - formation_w) * 0.5f;
    if (env->formation_x < 0) {
        env->formation_x = 0;
    }
    env->formation_y = 48.0f;
    env->direction = 1;
    env->move_timer = 0;
    env->alive_count = NUM_ALIENS;
    for (int i = 0; i < NUM_ALIENS; i++) {
        env->alien_alive[i] = 1;
    }
    clear_bullets(env);
    update_move_interval(env);
}

static void living_bounds(SpaceOnslaught* env, int* min_col, int* max_col, int* max_row) {
    *min_col = NUM_ALIEN_COLS;
    *max_col = -1;
    *max_row = -1;
    for (int row = 0; row < NUM_ALIEN_ROWS; row++) {
        for (int col = 0; col < NUM_ALIEN_COLS; col++) {
            if (!env->alien_alive[row * NUM_ALIEN_COLS + col]) {
                continue;
            }
            if (col < *min_col) {
                *min_col = col;
            }
            if (col > *max_col) {
                *max_col = col;
            }
            if (row > *max_row) {
                *max_row = row;
            }
        }
    }
}

static void compute_observations(SpaceOnslaught* env) {
    float* obs = env->observations;
    float w = env->width > 0 ? (float)env->width : 1.0f;
    float h = env->height > 0 ? (float)env->height : 1.0f;
    float lives0 = env->initial_lives > 0 ? (float)env->initial_lives : 1.0f;
    int interval = env->move_interval > 0 ? env->move_interval : 1;
    float timer = (float)env->move_timer / (float)interval;
    if (timer < 0.0f) {
        timer = 0.0f;
    }
    if (timer > 1.0f) {
        timer = 1.0f;
    }
    int idx = 0;

    // player, player bullet, 3 alien bullets, formation, 55 aliens XD
    obs[idx++] = env->player_x / w;
    obs[idx++] = (float)env->lives / lives0;

    if (env->player_bullet.active) {
        obs[idx++] = 1.0f;
        obs[idx++] = env->player_bullet.x / w;
        obs[idx++] = env->player_bullet.y / h;
    } else {
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
    }

    for (int i = 0; i < MAX_ALIEN_BULLETS; i++) {
        if (env->alien_bullets[i].active) {
            obs[idx++] = 1.0f;
            obs[idx++] = env->alien_bullets[i].x / w;
            obs[idx++] = env->alien_bullets[i].y / h;
        } else {
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }

    obs[idx++] = env->formation_x / w;
    obs[idx++] = env->formation_y / h;
    obs[idx++] = (env->direction > 0) ? 1.0f : 0.0f;
    obs[idx++] = timer;

    for (int i = 0; i < NUM_ALIENS; i++) {
        obs[idx++] = env->alien_alive[i] ? 1.0f : 0.0f;
    }

    assert(idx == SO_OBS_SIZE);
}

void add_log(SpaceOnslaught* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / WAVE_SCORE;
    env->log.n += 1;
}

void c_reset(SpaceOnslaught* env) {
    env->score = 0;
    env->lives = env->initial_lives;
    env->wave = 1;
    env->tick = 0;
    reset_player(env);
    reset_wave(env);
    compute_observations(env);
}

static void end_episode(SpaceOnslaught* env) {
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
}

static void fire_player(SpaceOnslaught* env) {
    if (env->player_bullet.active) {
        return;
    }
    env->player_bullet.active = 1;
    env->player_bullet.x = env->player_x + env->player_width * 0.5f - PLAYER_BULLET_W * 0.5f;
    env->player_bullet.y = env->player_y - PLAYER_BULLET_H;
}

static void spawn_alien_bullet(SpaceOnslaught* env) {
    int slot = -1;
    for (int i = 0; i < MAX_ALIEN_BULLETS; i++) {
        if (!env->alien_bullets[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return;
    }

    int eligible[NUM_ALIEN_COLS];
    int spawn_row[NUM_ALIEN_COLS];
    int n = 0;
    for (int col = 0; col < NUM_ALIEN_COLS; col++) {
        for (int row = NUM_ALIEN_ROWS - 1; row >= 0; row--) {
            if (env->alien_alive[row * NUM_ALIEN_COLS + col]) {
                eligible[n] = col;
                spawn_row[n] = row;
                n++;
                break;
            }
        }
    }
    if (n == 0) {
        return;
    }

    int pick = rand_r(&env->rng) % n;
    int col = eligible[pick];
    int row = spawn_row[pick];

    env->alien_bullets[slot].active = 1;
    env->alien_bullets[slot].x = alien_x(env, col) + env->alien_width * 0.5f - ALIEN_BULLET_W * 0.5f;
    env->alien_bullets[slot].y = alien_y(env, row) + env->alien_height;
}

static void hit_player(SpaceOnslaught* env) {
    if (env->lives <= 0) {
        return;
    }
    env->lives -= 1;
    clear_bullets(env);
    reset_player(env);
    if (env->lives <= 0) {
        end_episode(env);
    }
}

static void kill_alien(SpaceOnslaught* env, int idx) {
    if (idx < 0 || idx >= NUM_ALIENS || !env->alien_alive[idx]) {
        return;
    }
    env->alien_alive[idx] = 0;
    env->alive_count -= 1;
    if (env->alive_count < 0) {
        env->alive_count = 0;
    }
    int row = idx / NUM_ALIEN_COLS;
    int points = alien_points(row);
    env->score += points;
    env->rewards[0] += (float)points;
    env->player_bullet.active = 0;
    if (env->alive_count <= 0) {
        env->wave += 1;
        reset_wave(env);
    } else {
        update_move_interval(env);
    }
}

static void step_player_bullet(SpaceOnslaught* env) {
    if (!env->player_bullet.active) {
        return;
    }
    float old_y = env->player_bullet.y;
    env->player_bullet.y -= env->player_bullet_speed * TICK_RATE;
    float y = env->player_bullet.y;
    float h = old_y + PLAYER_BULLET_H - y;
    if (h < PLAYER_BULLET_H) {
        h = PLAYER_BULLET_H;
    }
    // bottom row first so an upward shot hits the nearest alien
    for (int row = NUM_ALIEN_ROWS - 1; row >= 0; row--) {
        for (int col = 0; col < NUM_ALIEN_COLS; col++) {
            int i = row * NUM_ALIEN_COLS + col;
            if (!env->alien_alive[i]) {
                continue;
            }
            if (rects_overlap(
                    env->player_bullet.x, y, PLAYER_BULLET_W, h,
                    alien_x(env, col), alien_y(env, row),
                    (float)env->alien_width, (float)env->alien_height)) {
                kill_alien(env, i);
                return;
            }
        }
    }
    if (env->player_bullet.y + PLAYER_BULLET_H < 0) {
        env->player_bullet.active = 0;
    }
}

static void step_alien_bullets(SpaceOnslaught* env) {
    for (int i = 0; i < MAX_ALIEN_BULLETS; i++) {
        Bullet* b = &env->alien_bullets[i];
        if (!b->active) {
            continue;
        }
        float old_y = b->y;
        b->y += env->alien_bullet_speed * TICK_RATE;
        float h = b->y + ALIEN_BULLET_H - old_y;
        if (h < ALIEN_BULLET_H) {
            h = ALIEN_BULLET_H;
        }
        if (rects_overlap(
                b->x, old_y, ALIEN_BULLET_W, h,
                env->player_x, env->player_y,
                (float)env->player_width, (float)env->player_height)) {
            hit_player(env);
            return;
        }
        if (b->y > env->height) {
            b->active = 0;
        }
    }
}

static void step_formation(SpaceOnslaught* env) {
    env->move_timer += 1;
    if (env->move_timer < env->move_interval) {
        return;
    }
    env->move_timer = 0;

    int min_col, max_col, max_row;
    living_bounds(env, &min_col, &max_col, &max_row);
    if (max_col < 0) {
        return;
    }

    int step = env->alien_step_x > 0 ? env->alien_step_x : 1;
    float dx = (float)(env->direction * step);
    float left = alien_x(env, min_col) + dx;
    float right = alien_x(env, max_col) + env->alien_width + dx;
    // reverse and drop if the living formation would leave the play area
    if (left < env->edge_margin || right > env->width - env->edge_margin) {
        env->direction = -env->direction;
        env->formation_y += env->alien_step_down;
    } else {
        env->formation_x += dx;
    }
}

static int formation_invaded(SpaceOnslaught* env) {
    int min_col, max_col, max_row;
    living_bounds(env, &min_col, &max_col, &max_row);
    if (max_row < 0) {
        return 0;
    }
    float bottom = alien_y(env, max_row) + env->alien_height;
    return bottom >= env->player_y;
}

static void step_frame(SpaceOnslaught* env, float action) {
    int act = 0;
    if (isfinite(action)) {
        act = (int)action;
    }
    int move = 0;
    int fire = 0;
    if (act == ACTION_LEFT || act == ACTION_LEFT_FIRE) {
        move = -1;
    } else if (act == ACTION_RIGHT || act == ACTION_RIGHT_FIRE) {
        move = 1;
    }
    if (act == ACTION_FIRE || act == ACTION_LEFT_FIRE || act == ACTION_RIGHT_FIRE) {
        fire = 1;
    }

    env->player_x += move * env->player_speed * TICK_RATE;
    clamp_player(env);

    if (fire) {
        fire_player(env);
    }

    step_player_bullet(env);
    if (env->terminals[0]) {
        return;
    }

    if (env->alien_fire_prob > 0.0f) {
        float u = (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
        if (u < env->alien_fire_prob) {
            spawn_alien_bullet(env);
        }
    }
    step_alien_bullets(env);
    if (env->terminals[0]) {
        return;
    }

    step_formation(env);
    if (formation_invaded(env)) {
        end_episode(env);
    }
}

void c_step(SpaceOnslaught* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0;
    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
        if (env->terminals[0]) {
            break;
        }
    }
    compute_observations(env);
}

void c_render(SpaceOnslaught* env) {
    if (!IsWindowReady()) {
        InitWindow(env->width, env->height, "PufferLib Space Onslaught");
        SetTargetFPS(60 / (env->frameskip > 0 ? env->frameskip : 1));
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    DrawLine(0, (int)env->player_y, env->width, (int)env->player_y, (Color){40, 80, 80, 255});

    Color row_color[NUM_ALIEN_ROWS] = {PUFF_WHITE, PUFF_CYAN, PUFF_CYAN, PUFF_RED, PUFF_RED};
    for (int i = 0; i < NUM_ALIENS; i++) {
        if (!env->alien_alive[i]) {
            continue;
        }
        int row = i / NUM_ALIEN_COLS;
        int col = i % NUM_ALIEN_COLS;
        DrawRectangle(
            (int)alien_x(env, col), (int)alien_y(env, row),
            env->alien_width, env->alien_height, row_color[row]);
    }

    DrawRectangle(
        (int)env->player_x, (int)env->player_y,
        env->player_width, env->player_height, PUFF_CYAN);
    int turret_w = 8;
    int turret_h = 6;
    DrawRectangle(
        (int)(env->player_x + env->player_width * 0.5f - turret_w * 0.5f),
        (int)(env->player_y - turret_h),
        turret_w, turret_h, PUFF_CYAN);

    if (env->player_bullet.active) {
        DrawRectangle(
            (int)env->player_bullet.x, (int)env->player_bullet.y,
            (int)PLAYER_BULLET_W, (int)PLAYER_BULLET_H, PUFF_WHITE);
    }
    for (int i = 0; i < MAX_ALIEN_BULLETS; i++) {
        if (!env->alien_bullets[i].active) {
            continue;
        }
        DrawRectangle(
            (int)env->alien_bullets[i].x, (int)env->alien_bullets[i].y,
            (int)ALIEN_BULLET_W, (int)ALIEN_BULLET_H, PUFF_RED);
    }

    DrawText(TextFormat("score %d", env->score), 12, 8, 20, PUFF_WHITE);
    DrawText(TextFormat("lives %d", env->lives), env->width / 2 - 40, 8, 20, PUFF_WHITE);
    DrawText(TextFormat("wave %d", env->wave), env->width - 90, 8, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(SpaceOnslaught* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}
// cooked by alok ;)
