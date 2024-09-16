#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "raylib.h"

#define NOOP 0
#define FIRE 1
#define LEFT 2
#define RIGHT 3
#define HALF_MAX_SCORE 432
#define MAX_SCORE 864
#define HALF_PADDLE_WIDTH 31

typedef struct CBreakout CBreakout;
struct CBreakout {
    float* observations;
    unsigned char* actions;
    unsigned char* dones;
    float* rewards;
    int* scores;
    float* episodic_returns;
    float* paddle_positions;
    float* ball_positions;
    float* ball_velocities;
    float* brick_x;
    float* brick_y;
    float* brick_states;
    int* balls_fired;
    float* wall_positions;
    float* paddle_widths;
    float* paddle_heights;
    float* ball_speeds;
    int* hit_counters;
    int width;
    int height;
    int obs_size;
    int num_bricks;
    int num_brick_rows;
    int num_brick_cols;
    int ball_width;
    int ball_height;
    int brick_width;
    int brick_height;
    int num_agents;
    int* num_balls;
    float dt;
    float substep_dt;
    int num_substeps;
    int* timesteps;
    int frameskip;
};

void generate_brick_positions(CBreakout* env) {
    int y_offset = 50;
    for (int row = 0; row < env->num_brick_rows; row++) {
        for (int col = 0; col < env->num_brick_cols; col++) {
            int idx = row * env->num_brick_cols + col;
            env->brick_x[idx] = col*env->brick_width;
            env->brick_y[idx] = row*env->brick_height + y_offset;
        }
    }
}

CBreakout* init_cbreakout(float dt, int frameskip, unsigned char* actions, float* observations,
        float* rewards, int* scores, float* episodic_returns, int* num_balls, float* paddle_positions,
        float* ball_positions, float* ball_velocities, float* brick_x, float* brick_y, float* brick_states,
        unsigned char* dones, int* balls_fired, float* wall_positions, float* paddle_widths,
        float* paddle_heights, float* ball_speeds, int* hit_counters, int num_agents, int width,
        int height, int ball_width, int ball_height, int brick_width, int brick_height, int obs_size,
        int num_bricks, int num_brick_rows, int num_brick_cols) {

    CBreakout* env = (CBreakout*)calloc(1, sizeof(CBreakout));
    env->dt = dt;
    env->frameskip = frameskip;
    env->actions = actions;
    env->observations = observations;
    env->rewards = rewards;
    env->scores = scores;
    env->episodic_returns = episodic_returns;
    env->num_balls = num_balls;
    env->num_balls[0] = -1;
    env->paddle_positions = paddle_positions;
    env->ball_positions = ball_positions;
    env->ball_velocities = ball_velocities;
    env->brick_x = brick_x;
    env->brick_y = brick_y;
    env->brick_states = brick_states;
    env->dones = dones;
    env->balls_fired = balls_fired;
    env->wall_positions = wall_positions;
    env->wall_positions[4] = width;
    env->paddle_widths = paddle_widths;
    env->paddle_heights = paddle_heights;
    env->ball_speeds = ball_speeds;
    env->hit_counters = hit_counters;
    env->num_agents = num_agents;
    env->width = width;
    env->height = height;
    env->ball_width = ball_width;
    env->ball_height = ball_height;
    env->brick_width = brick_width;
    env->brick_height = brick_height;
    env->obs_size = obs_size;
    env->num_bricks = num_bricks;
    env->num_brick_rows = num_brick_rows;
    env->num_brick_cols = num_brick_cols;
    env->ball_speeds = ball_speeds;
    env->num_agents = num_agents;
    env->num_substeps = 5;
    env->substep_dt = env->dt / env->num_substeps;
    generate_brick_positions(env);
    return env;
}

CBreakout* allocate_cbreakout(float dt, int frameskip, int num_agents,
        int width, int height, int ball_width, int ball_height,
        int brick_width, int brick_height, int obs_size,
        int num_brick_rows, int num_brick_cols) {
    int num_bricks = num_brick_rows * num_brick_cols;
    unsigned char* actions = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    float* observations = (float*)calloc(num_agents*obs_size, sizeof(float));
    unsigned char* dones = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    float* rewards = (float*)calloc(num_agents, sizeof(float));
    int* scores = (int*)calloc(num_agents, sizeof(int));
    float* episodic_returns = (float*)calloc(num_agents, sizeof(float));
    int* num_balls = (int*)calloc(num_agents, sizeof(int));
    float* paddle_positions = (float*)calloc(2, sizeof(float));
    float* ball_positions = (float*)calloc(2, sizeof(float));
    float* ball_velocities = (float*)calloc(2, sizeof(float));
    float* brick_x = (float*)calloc(num_bricks, sizeof(float));
    float* brick_y = (float*)calloc(num_bricks, sizeof(float));
    float* brick_states = (float*)calloc(num_bricks, sizeof(float));
    int* balls_fired = (int*)calloc(num_agents, sizeof(int));
    float* wall_positions = (float*)calloc(6, sizeof(float));
    float* paddle_widths = (float*)calloc(num_agents, sizeof(float));
    float* paddle_heights = (float*)calloc(num_agents, sizeof(float));
    float* ball_speeds = (float*)calloc(num_agents, sizeof(float));
    int* hit_counters = (int*)calloc(num_agents, sizeof(int));

    CBreakout* env = init_cbreakout(dt, frameskip, actions, observations,
        rewards, scores, episodic_returns, num_balls, paddle_positions,
        ball_positions, ball_velocities, brick_x, brick_y, brick_states,
        dones, balls_fired, wall_positions, paddle_widths, paddle_heights,
        ball_speeds, hit_counters, num_agents, width, height, ball_width,
        ball_height, brick_width, brick_height, obs_size,
        num_bricks, num_brick_rows, num_brick_cols);

    return env;
}

void free_allocated_cbreakout(CBreakout* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free(env->scores);
    free(env->episodic_returns);
    free(env->num_balls);
    free(env->paddle_positions);
    free(env->ball_positions);
    free(env->ball_velocities);
    free(env->brick_x);
    free(env->brick_y);
    free(env->brick_states);
    free(env->balls_fired);
    free(env->wall_positions);
    free(env->paddle_widths);
    free(env->paddle_heights);
    free(env->ball_speeds);
    free(env->hit_counters);
    free(env);
}

void compute_observations(CBreakout* env) {
    env->observations[0] = env->paddle_positions[0];
    env->observations[1] = env->paddle_positions[1];
    env->observations[2] = env->ball_positions[0];
    env->observations[3] = env->ball_positions[1];
    env->observations[4] = env->ball_velocities[0];
    env->observations[5] = env->ball_velocities[1];
    env->observations[6] = env->balls_fired[0];
    env->observations[8] = env->num_balls[0];
    env->observations[10] = env->paddle_widths[0];
    env->observations[11] = env->brick_states[0];
}

// TODO: Why can't I inline this?
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

bool handle_paddle_ball_collisions(CBreakout* env) {
    float base_angle = M_PI / 4.0f;

    // Check if ball is above the paddle
    if (env->ball_positions[1] + env->ball_height < env->paddle_positions[1]) {
        return false;
    }

    // Check for collision
    if (check_collision_discrete(env->paddle_positions[0], env->paddle_positions[1],
            env->paddle_widths[0], env->paddle_heights[0], env->ball_positions[0],
            env->ball_positions[1], env->ball_width, env->ball_height)) {
        float relative_intersection = ((env->ball_positions[0] +
            env->ball_width / 2) - env->paddle_positions[0]) / env->paddle_widths[0];
        float angle = -base_angle + relative_intersection * 2 * base_angle;
        env->ball_velocities[0] = sin(angle) * env->ball_speeds[0] * env->substep_dt;
        env->ball_velocities[1] = -cos(angle) * env->ball_speeds[0] * env->substep_dt;
        env->hit_counters[0] += 1;
        if (env->hit_counters[0] % 4 == 0 && env->hit_counters[0] <= 12) {
            env->ball_speeds[0] += 64;
        }
        if (env->scores[0] == HALF_MAX_SCORE) {
            env->brick_states[0] = 0.0;
        }
        return true;
    }
    return false;
}

bool handle_wall_ball_collisions(CBreakout* env) {
    if (env->ball_positions[0] > 0 && env->ball_positions[0]
            + env->ball_width < env->width && env->ball_positions[1] > 0) {
        return false;
    }

    // Left Wall Collision
    if (check_collision_discrete(
            env->wall_positions[0] - 50, env->wall_positions[1],
            50, env->height, env->ball_positions[0], env->ball_positions[1],
            env->ball_width, env->ball_height)) {
        env->ball_positions[0] = 0;
        env->ball_velocities[0] *= -1;
        return true;
    }

    // Top Wall Collision
    if (check_collision_discrete(
            env->wall_positions[2], env->wall_positions[3] - 50,
            env->width, 50, env->ball_positions[0], env->ball_positions[1],
            env->ball_width, env->ball_height)) {
        env->ball_positions[1] = 0;
        env->ball_velocities[1] *= -1;
        env->paddle_widths[0] = HALF_PADDLE_WIDTH;
        return true;
    }

    // Right Wall Collision
    if (check_collision_discrete(
            env->wall_positions[4], env->wall_positions[5],
            50, env->height, env->ball_positions[0], env->ball_positions[1],
            env->ball_width, env->ball_height)) {
        env->ball_positions[0] = env->width - env->ball_width;
        env->ball_velocities[0] *= -1;
        return true;
    }

    return false;
}

bool handle_brick_ball_collisions(CBreakout* env) {
    if (env->ball_positions[1] > env->brick_y[env->num_bricks-1] + env->brick_height) {
        return false;
    }
    
    // Loop over bricks in reverse to check lower bricks first
    for (int brick_idx = env->num_bricks - 1; brick_idx >= 0; brick_idx--) {
        if (env->brick_states[brick_idx] == 1.0) {
            continue;
        }
        if (check_collision_discrete(env->brick_x[brick_idx],
                env->brick_y[brick_idx], env->brick_width, env->brick_height,
                env->ball_positions[0], env->ball_positions[1], env->ball_width, env->ball_height)) {
            env->brick_states[brick_idx] = 1.0;
            float score = 7 - 3 * (brick_idx / env->num_brick_cols / 2);
            env->rewards[0] += score;
            env->scores[0] += score;

            // Determine collision direction
            if (env->ball_positions[1] + env->ball_height <= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from below the brick
                env->ball_velocities[1] *= -1;
                return true;
            } else if (env->ball_positions[1] >= env->brick_y[brick_idx] + (env->brick_height / 2)) {
                // Hit was from above the brick
                env->ball_velocities[1] *= -1;
                return true;
            } else if (env->ball_positions[0] + env->ball_width <= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the left
                env->ball_velocities[0] *= -1;
                return true;
            } else if (env->ball_positions[0] >= env->brick_x[brick_idx] + (env->brick_width / 2)) {
                // Hit was from the right
                env->ball_velocities[0] *= -1;
                return true;
            }
        }
    }
    return false;
}

void reset(CBreakout* env) {
    if (env->num_balls[0] == -1 || env->scores[0] == MAX_SCORE) {
        env->scores[0] = 0;
        env->num_balls[0] = 5;
        for (int i = 0; i < env->num_bricks; i++) {
            env->brick_states[i] = 0.0;
        }
        env->hit_counters[0] = 0;
        env->ball_speeds[0] = 256;
        env->paddle_widths[0] = 2 * HALF_PADDLE_WIDTH;
    }

    env->dones[0] = 0;
    env->balls_fired[0] = 0.0;

    env->paddle_positions[0] = env->width / 2.0 - env->paddle_widths[0] / 2;
    env->paddle_positions[1] = env->height - env->paddle_heights[0] - 10;

    env->ball_positions[0] = env->paddle_positions[0] + (env->paddle_widths[0] / 2 - env->ball_width / 2);
    env->ball_positions[1] = env->height / 2 - 30;

    env->ball_velocities[0] = 0.0;
    env->ball_velocities[1] = 0.0;
}

void step(CBreakout* env) {
    env->rewards[0] = 0.0;
    int action = env->actions[0];

    for (int i = 0; i < env->frameskip; i++) {
        for (int j = 0; j < env->num_substeps; j++) {
            if (action == FIRE && env->balls_fired[0] == 0.0) {
                env->balls_fired[0] = 1.0;
                float direction = M_PI / 3.25f;

                if (rand() % 2 == 0) {
                    env->ball_velocities[0] = sin(direction) * env->ball_speeds[0] * env->substep_dt;
                    env->ball_velocities[1] = cos(direction) * env->ball_speeds[0] * env->substep_dt;
                } else {
                    env->ball_velocities[0] = -sin(direction) * env->ball_speeds[0] * env->substep_dt;
                    env->ball_velocities[1] = cos(direction) * env->ball_speeds[0] * env->substep_dt;
                }
            } else if (action == LEFT) {
                env->paddle_positions[0] -= 620 * env->substep_dt;
                env->paddle_positions[0] = fmaxf(0, env->paddle_positions[0]);
            } else if (action == RIGHT) {
                env->paddle_positions[0] += 620 * env->substep_dt;
                env->paddle_positions[0] = fminf(env->width - env->paddle_widths[0], env->paddle_positions[0]);
            }

            handle_brick_ball_collisions(env);
            handle_paddle_ball_collisions(env);
            handle_wall_ball_collisions(env);

            env->ball_positions[0] += env->ball_velocities[0];
            env->ball_positions[1] += env->ball_velocities[1];
        }

        if (env->ball_positions[1] >= env->paddle_positions[1] + env->paddle_heights[0]) {
            env->num_balls[0] -= 1;
            env->dones[0] = 1;
        }
        if (env->scores[0] == MAX_SCORE) {
            env->dones[0] = 1;
        }
        if (env->dones[0] == 1) {
            env->episodic_returns[0] = env->scores[0];
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
    int paddle_width;
    int paddle_height;
    int num_brick_rows;
    int num_brick_cols;
    float ball_width;
    float ball_height;
    float brick_width;
    float brick_height;
};

Client* make_client(int width, int height, int paddle_width,
        int paddle_height, int ball_width, int ball_height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    client->paddle_width = paddle_width;
    client->paddle_height = paddle_height;
    client->ball_width = ball_width;
    client->ball_height = ball_height;

    InitWindow(width, height, "PufferLib Ray Breakout");
    SetTargetFPS(15);

    //sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")
    //self.sound = rl.LoadSound(sound_path.encode())

    return client;
}

void render(Client* client, CBreakout* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int paddle_x = env->paddle_positions[0];
    int paddle_y = env->paddle_positions[1];

    // Draw the paddle
    DrawRectangle(paddle_x, paddle_y, client->paddle_width,
        client->paddle_height, DARKGRAY);

    // Draw the ball
    DrawRectangle(env->ball_positions[0], env->ball_positions[1],
        client->ball_width, client->ball_height, WHITE);

    // Draw the bricks
    for (int row = 0; row < env->num_brick_rows; row++) {
        for (int col = 0; col < env->num_brick_cols; col++) {
            int brick_idx = row * env->num_brick_cols + col;
            if (env->brick_states[brick_idx] == 1) {
                continue;
            }
            int x = env->brick_x[brick_idx];
            int y = env->brick_y[brick_idx];
            Color brick_color = BRICK_COLORS[row];
            DrawRectangle(x, y, env->brick_width, env->brick_height, brick_color);
        }
    }

    // Draw Score
    DrawText(TextFormat("Score: %i", env->scores[0]), 10, 10, 20, WHITE);

    // Draw Balls
    DrawText(TextFormat("Balls: %i", env->num_balls[0]), client->width - 80, 10, 20, WHITE);

    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
