#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define NOOP 0
#define LEFT 1
#define RIGHT 2
#define MAX_BALL_SPEED 448
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f

#define LOG_BUFFER_SIZE 1024

#define BRICK_INDEX_NO_COLLISION -4
#define BRICK_INDEX_SIDEWALL_COLLISION -3
#define BRICK_INDEX_BACKWALL_COLLISION -2
#define BRICK_INDEX_PADDLE_COLLISION -1

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}
 
typedef struct Breakout Breakout;
struct Breakout {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    int score;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float* brick_x;
    float* brick_y;
    float* brick_states;
    int balls_fired;
    float paddle_width;
    float paddle_height;
    float ball_speed;
    int hits;
    int width;
    int height;
    int num_bricks;
    int brick_rows;
    int brick_cols;
    int ball_width;
    int ball_height;
    int brick_width;
    int brick_height;
    int num_balls;
    int max_score;
    int half_max_score;
    int frameskip;
    unsigned char hit_brick;
};

typedef struct CollisionInfo CollisionInfo;
struct CollisionInfo {
    float t;
    float overlap;
    float x;
    float y;
    float vx; 
    float vy;
    int brick_index;
};

void generate_brick_positions(Breakout* env) {
    env->half_max_score=0;
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = col*env->brick_width;
            env->brick_y[idx] = row*env->brick_height + Y_OFFSET;
            env->half_max_score += 7 - 3 * (idx / env->brick_cols / 2);
        }
    }
    env->max_score=2*env->half_max_score;
}

void init(Breakout* env) {
    env->num_bricks = env->brick_rows * env->brick_cols;
    assert(env->num_bricks > 0);

    env->brick_x = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_y = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_states = (float*)calloc(env->num_bricks, sizeof(float));
    env->num_balls = -1;
    generate_brick_positions(env);
}

void allocate(Breakout* env) {
    init(env);
    env->observations = (float*)calloc(11 + env->num_bricks, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(Breakout* env) {
    free(env->brick_x);
    free(env->brick_y);
    free(env->brick_states);
    free_logbuffer(env->log_buffer);
}

void free_allocated(Breakout* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_initialized(env);
}

void compute_observations(Breakout* env) {
    env->observations[0] = env->paddle_x / env->width;
    env->observations[1] = env->paddle_y / env->height;
    env->observations[2] = env->ball_x / env->width;
    env->observations[3] = env->ball_y / env->height;
    env->observations[4] = env->ball_vx / 512.0f;
    env->observations[5] = env->ball_vy / 512.0f;
    env->observations[6] = env->balls_fired / 5.0f;
    env->observations[8] = env->num_balls / 5.0f;
    env->observations[10] = env->paddle_width / (2.0f * HALF_PADDLE_WIDTH);
    for (int i = 0; i < env->num_bricks; i++) {
        env->observations[11 + i] = env->brick_states[i];
    }
}

// Collision of a stationary vertical line segment (xw,yw) to (xw,yw+hw)
// with a moving line segment (x+vx*t,y+vy*t) to (x+vx*t,y+vy*t+h).
static inline bool calc_vline_collision(float xw, float yw, float hw, float x,
        float y, float vx, float vy, float h, CollisionInfo* col) {
    float t_new = (xw - x) / vx;
    float topmost = fmin(yw + hw, y + h + vy * t_new);
    float botmost = fmax(yw, y + vy * t_new);
    float overlap_new = topmost - botmost;

    // Collision finds the smallest time of collision with the greatest overlap
    // between the ball and the wall.
    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f  && 
        (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = xw;
        col->y = y + vy * t_new;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}
static inline bool calc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, CollisionInfo* col) {
    float t_new = (yw - y) / vy;
    float rightmost = fminf(xw + ww, x + w + vx * t_new);
    float leftmost = fmaxf(xw, x + vx * t_new);
    float overlap_new = rightmost - leftmost;

    // Collision finds the smallest time of collision with the greatest overlap between the ball and the wall.
    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f && 
        (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = x + vx * t_new;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}
static inline void calc_brick_collision(Breakout* env, int idx, 
        CollisionInfo* collision_info) {
    bool collision = false;
    // Brick left wall collides with ball right side
    if (env->ball_vx > 0) {
        if (calc_vline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_height,
                env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height, collision_info)) {
            collision = true;
            collision_info->x -= env->ball_width;
        }
    }

    // Brick right wall collides with ball left side
    if (env->ball_vx < 0) {
        if (calc_vline_collision(env->brick_x[idx] + env->brick_width, env->brick_y[idx], env->brick_height,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height, collision_info)) {
            collision = true;
        }
    }

    // Brick top wall collides with ball bottom side
    if (env->ball_vy > 0) {
        if (calc_hline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_width,
                env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy, env->ball_width, collision_info)) {
            collision = true;
            collision_info->y -= env->ball_height;
        }
    }

    // Brick bottom wall collides with ball top side
    if (env->ball_vy < 0) {
        if (calc_hline_collision(env->brick_x[idx], env->brick_y[idx] + env->brick_height, env->brick_width,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_width, collision_info)) {
            collision = true;
        }
    }
    if (collision) {
        collision_info->brick_index = idx;
    }
}
static inline int column_index(Breakout* env, float x) {
    return (int)(floorf(x / env->brick_width));
}
static inline int row_index(Breakout* env, float y) {
    return (int)(floorf((y - Y_OFFSET) / env->brick_height));
}

void calc_all_brick_collisions(Breakout* env, CollisionInfo* collision_info) {
    int column_from = column_index(env, fminf(env->ball_x + env->ball_vx, env->ball_x));
    column_from = fmaxf(column_from, 0);
    int column_to = column_index(env, fmaxf(env->ball_x + env->ball_width + env->ball_vx, env->ball_x + env->ball_width));
    column_to = fminf(column_to, env->brick_cols - 1);
    int row_from = row_index(env, fminf(env->ball_y + env->ball_vy, env->ball_y));
    row_from = fmaxf(row_from, 0);
    int row_to = row_index(env, fmaxf(env->ball_y + env->ball_height + env->ball_vy, env->ball_y + env->ball_height));
    row_to = fminf(row_to, env->brick_rows - 1);

    for (int row = row_from; row <= row_to; row++) {
        for (int column = column_from; column <= column_to; column++) {
            int brick_index = row * env->brick_cols + column;
            if (env->brick_states[brick_index] == 0.0)
                calc_brick_collision(env, brick_index, collision_info);
        }
    }
}

bool calc_paddle_ball_collisions(Breakout* env, CollisionInfo* collision_info) {
    float base_angle = M_PI / 4.0f;

    // Check if ball is above the paddle
    if (env->ball_y + env->ball_height + env->ball_vy < env->paddle_y) {
        return false;
    }

    // Check for collision
    // If we've found another collision (eg the ball hits the wall before the paddle)
    // this correctly skips the paddle collision.
    if (!calc_hline_collision(env->paddle_x, env->paddle_y, env->paddle_width,
          env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy, env->ball_width,
          collision_info) || collision_info->t > 1.0f) {
        return false;
    }

    collision_info->y -= env->ball_height;
    collision_info->brick_index = BRICK_INDEX_PADDLE_COLLISION;

    env->hit_brick = false;
    float relative_intersection = ((env->ball_x +
                                    env->ball_width / 2) -
                                   env->paddle_x) /
                                  env->paddle_width;
    float angle = -base_angle + relative_intersection * 2 * base_angle;
    env->ball_vx = sin(angle) * env->ball_speed * TICK_RATE;
    env->ball_vy = -cos(angle) * env->ball_speed * TICK_RATE;
    env->hits += 1;
    //env->rewards[0] += 0.1;
    //env->log.episode_return += 0.1;
    if (env->hits % 4 == 0 && env->ball_speed < MAX_BALL_SPEED) {
        env->ball_speed += 64;
    }
    if (env->score == env->half_max_score) {
        for (int i = 0; i < env->num_bricks; i++) {
            env->brick_states[i] = 0.0;
        }
    }
    return true;
}

void calc_all_wall_collisions(Breakout* env, CollisionInfo* collision_info) {
    //bool collision = false;
    if (env->ball_vx < 0) {
        if (calc_vline_collision(0, 0, env->height,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height,
                collision_info)) {
            //collision = true;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vx > 0) {
        if (calc_vline_collision(env->width, 0, env->height,
                 env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height,
                 collision_info)) {
            //collision = true;
            collision_info->x -= env->ball_width;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vy < 0) {
        if (calc_hline_collision(0, 0, env->width,
                 env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_width,
                 collision_info)) {
            //collision = true;
            collision_info->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
        }
    }
}

// With rare floating point conditions, the ball could escape the bounds.
// Let's handle that explicitly.
void check_wall_bounds(Breakout* env) {
    if (env->ball_x < 0)
        env->ball_x += MAX_BALL_SPEED * 1.1f * TICK_RATE;
    if (env->ball_x > env->width)
        env->ball_x -= MAX_BALL_SPEED * 1.1f * TICK_RATE;
    if (env->ball_y < 0)
        env->ball_y += MAX_BALL_SPEED * 1.1f * TICK_RATE;
}

void destroy_brick(Breakout* env, int brick_idx) {
    float gained_points = 7 - 3 * (brick_idx / env->brick_cols / 2);
    env->score += gained_points;
    env->brick_states[brick_idx] = 1.0;
    env->log.episode_return += gained_points;
    env->rewards[0] += gained_points;

    if (brick_idx / env->brick_cols < 3) {
        env->ball_speed = MAX_BALL_SPEED;
    }
}

bool handle_collisions(Breakout* env) {
    CollisionInfo collision_info = {
        .t = 2.0f,
        .overlap = -1.0f,
        .x = 0.0f,
        .y = 0.0f,
        .vx = 0.0f,
        .vy = 0.0f,
        .brick_index = BRICK_INDEX_NO_COLLISION,
    };

    check_wall_bounds(env);

    calc_all_brick_collisions(env, &collision_info);
    calc_all_wall_collisions(env, &collision_info);
    calc_paddle_ball_collisions(env, &collision_info);
    if (collision_info.brick_index != BRICK_INDEX_PADDLE_COLLISION 
            && collision_info.t <= 1.0f) {
        env->ball_x = collision_info.x;
        env->ball_y = collision_info.y;
        env->ball_vx = collision_info.vx;
        env->ball_vy = collision_info.vy;
        if (collision_info.brick_index >= 0) {
            destroy_brick(env, collision_info.brick_index);
        }
        if (collision_info.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            env->paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return collision_info.brick_index != BRICK_INDEX_NO_COLLISION;
}

void reset_round(Breakout* env) {
    env->balls_fired = 0;
    env->hit_brick = false;
    env->hits = 0;
    env->ball_speed = 256;
    env->paddle_width = 2 * HALF_PADDLE_WIDTH;

    env->paddle_x = env->width / 2.0 - env->paddle_width / 2;
    env->paddle_y = env->height - env->paddle_height - 10;


    env->ball_x = env->paddle_x + (env->paddle_width / 2 - env->ball_width / 2);
    env->ball_y = env->height / 2 - 30;

    env->ball_vx = 0.0;
    env->ball_vy = 0.0;
}
void c_reset(Breakout* env) {
    env->log = (Log){0};
    env->score = 0;
    env->num_balls = 5;
    for (int i = 0; i < env->num_bricks; i++) {
        env->brick_states[i] = 0.0;
    }
    reset_round(env);
    compute_observations(env);
}

void step_frame(Breakout* env, int action) {
    if (env->balls_fired == 0) {
        env->balls_fired = 1;
        float direction = M_PI / 3.25f;

        env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
        env->ball_vx = sin(direction) * env->ball_speed * TICK_RATE;
        if (rand() % 2 == 0) {
            env->ball_vx = -env->ball_vx;
        }
    } else if (action == LEFT) {
        env->paddle_x -= 620 * TICK_RATE;
        env->paddle_x = fmaxf(0, env->paddle_x);
    } else if (action == RIGHT) {
        env->paddle_x += 620 * TICK_RATE;
        env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
    }

    //Handle collisions. 
    //Regular timestepping is done only if there are no collisions.
    if(!handle_collisions(env)){
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if (env->ball_y >= env->paddle_y + env->paddle_height) {
        env->num_balls -= 1;
        reset_round(env);
    }
    if (env->num_balls < 0 || env->score == env->max_score) {
        env->dones[0] = 1;
        env->log.score = env->score;
        add_log(env->log_buffer, &env->log);
        c_reset(env);
    }
}

void c_step(Breakout* env) {
    env->dones[0] = 0;
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;

    int action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        step_frame(env, action);
    }

    compute_observations(env);
}

Color BRICK_COLORS[6] = {RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D ball;
};

Client* make_client(Breakout* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Ray Breakout");
    SetTargetFPS(60);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    client->ball = LoadTexture("resources/puffers_128.png");
    return client;
}

void c_render(Client* client, Breakout* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(env->paddle_x, env->paddle_y,
        env->paddle_width, env->paddle_height, (Color){0, 255, 255, 255});

    // Draw ball
    DrawTexturePro(
        client->ball,
        (Rectangle){
            (env->ball_vx > 0) ? 0 : 128,
            0, 128, 128,
        },
        (Rectangle){
            env->ball_x,
            env->ball_y,
            env->ball_width,
            env->ball_height
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    //DrawRectangle(env->ball_x, env->ball_y,
    //    env->ball_width, env->ball_height, WHITE);

    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int brick_idx = row * env->brick_cols + col;
            if (env->brick_states[brick_idx] == 1) {
                continue;
            }
            int x = env->brick_x[brick_idx];
            int y = env->brick_y[brick_idx];
            Color brick_color = BRICK_COLORS[row];
            DrawRectangle(x, y, env->brick_width, env->brick_height, brick_color);
        }
    }

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
