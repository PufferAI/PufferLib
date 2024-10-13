#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define NOOP 0
#define FIRE 1
#define LEFT 2
#define RIGHT 3
#define HALF_MAX_SCORE 432
#define MAX_SCORE 864
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f

#define LOG_BUFFER_SIZE 1024

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
    unsigned int* actions;
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
    int frameskip;
};

void generate_brick_positions(Breakout* env) {
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = col*env->brick_width;
            env->brick_y[idx] = row*env->brick_height + Y_OFFSET;
        }
    }
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
    env->actions = (unsigned int*)calloc(1, sizeof(unsigned int));
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

bool check_collision_discrete(float x, float y, int width, int height,
        float other_x, float other_y, int other_width, int other_height) {
    if (x + width <= other_x || other_x + other_width <= x) {
        return false;
    }
    if (y + height <= other_y || other_y + other_height <= y) {
        return false;
    }
    return true;
}

bool handle_paddle_ball_collisions(Breakout* env) {
    float base_angle = M_PI / 4.0f;

    // Check if ball is above the paddle
    if (env->ball_y + env->ball_height < env->paddle_y) {
        return false;
    }

    // Check for collision
    if (check_collision_discrete(env->paddle_x, env->paddle_y,
            env->paddle_width, env->paddle_height, env->ball_x,
            env->ball_y, env->ball_width, env->ball_height)) {
        float relative_intersection = ((env->ball_x +
            env->ball_width / 2) - env->paddle_x) / env->paddle_width;
        float angle = -base_angle + relative_intersection * 2 * base_angle;
        env->ball_vx = sin(angle) * env->ball_speed * TICK_RATE;
        env->ball_vy = -cos(angle) * env->ball_speed * TICK_RATE;
        env->hits += 1;
        if (env->hits % 4 == 0 && env->hits <= 12) {
            env->ball_speed += 64;
        }
        if (env->score == HALF_MAX_SCORE) {
            env->brick_states[0] = 0.0;
        }
        return true;
    }
    return false;
}

bool handle_wall_ball_collisions(Breakout* env) {
    if (env->ball_x > 0 && env->ball_x
            + env->ball_width < env->width && env->ball_y > 0) {
        return false;
    }

    // Left Wall Collision
    if (check_collision_discrete(-Y_OFFSET, 0, Y_OFFSET, env->height,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_x = 0;
        env->ball_vx *= -1;
        return true;
    }

    // Top Wall Collision
    if (check_collision_discrete(0, -Y_OFFSET, env->width, Y_OFFSET,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_y = 0;
        env->ball_vy *= -1;
        env->paddle_width = HALF_PADDLE_WIDTH;
        return true;
    }

    // Right Wall Collision
    if (check_collision_discrete(env->width, 0, Y_OFFSET, env->height,
            env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
        env->ball_x = env->width - env->ball_width;
        env->ball_vx *= -1;
        return true;
    }

    return false;
}

bool handle_brick_ball_collisions(Breakout* env) {
    if (env->ball_y > env->brick_y[env->num_bricks-1] + env->brick_height) {
        return false;
    }
    
    // Loop over bricks in reverse to check lower bricks first
    for (int brick_idx = env->num_bricks - 1; brick_idx >= 0; brick_idx--) {
        if (env->brick_states[brick_idx] == 1.0) {
            continue;
        }
        if (check_collision_discrete(env->brick_x[brick_idx],
                env->brick_y[brick_idx], env->brick_width, env->brick_height,
                env->ball_x, env->ball_y, env->ball_width, env->ball_height)) {
            env->brick_states[brick_idx] = 1.0;
            float score = 7 - 3 * (brick_idx / env->brick_cols / 2);
            env->rewards[0] += 1.0;
            env->log.episode_return += 1.0;
            env->score += score;

            // Determine collision direction
            if (env->ball_y + env->ball_height <= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from below the brick
                env->ball_vy *= -1;
                return true;
            } else if (env->ball_y >= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from above the brick
                env->ball_vy *= -1;
                return true;
            } else if (env->ball_x + env->ball_width <= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the left
                env->ball_vx *= -1;
                return true;
            } else if (env->ball_x >= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the right
                env->ball_vx *= -1;
                return true;
            }
        }
    }
    return false;
}

void reset_round(Breakout* env) {
    env->balls_fired = 0;
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
void reset(Breakout* env) {
    env->log = (Log){0};
    env->score = 0;
    env->num_balls = 5;
    for (int i = 0; i < env->num_bricks; i++) {
        env->brick_states[i] = 0.0;
    }
    reset_round(env);
    compute_observations(env);
}

void step(Breakout* env) {
    env->dones[0] = 0;
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = env->actions[0];

    for (int i = 0; i < env->frameskip; i++) {
        if (action == FIRE && env->balls_fired == 0) {
            env->balls_fired = 1;
            float direction = M_PI / 3.25f;

            if (rand() % 2 == 0) {
                env->ball_vx = sin(direction) * env->ball_speed * TICK_RATE;
                env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
            } else {
                env->ball_vx = -sin(direction) * env->ball_speed * TICK_RATE;
                env->ball_vy = cos(direction) * env->ball_speed * TICK_RATE;
            }
        } else if (action == LEFT) {
            env->paddle_x -= 620 * TICK_RATE;
            env->paddle_x = fmaxf(0, env->paddle_x);
        } else if (action == RIGHT) {
            env->paddle_x += 620 * TICK_RATE;
            env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
        }

        handle_brick_ball_collisions(env);
        handle_paddle_ball_collisions(env);
        handle_wall_ball_collisions(env);

        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;

        if (env->ball_y >= env->paddle_y + env->paddle_height) {
            env->num_balls -= 1;
            reset_round(env);
        }
        if (env->num_balls < 0 || env->score == MAX_SCORE) {
            env->dones[0] = 1;
            env->log.score = env->score;
            add_log(env->log_buffer, &env->log);
            reset(env);
        }
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

void render(Client* client, Breakout* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
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
