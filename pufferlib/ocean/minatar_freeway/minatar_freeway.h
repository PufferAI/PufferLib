
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char UP = 2;
const unsigned char RIGHT = 3;
const unsigned char DOWN = 4;
const unsigned char FIRE = 5;

const unsigned char PLAYER_SPEED = 3;
const unsigned short int TIME_LIMIT = 2500;
// 9 moves to get across freeway
const unsigned int MAX_SCORE = (TIME_LIMIT / PLAYER_SPEED) / 9;

const unsigned char FULL_ACTION_SET[6] = {NOOP, LEFT, UP, RIGHT, DOWN, FIRE};
const unsigned char MINIMAL_ACTION_SET[3] = {NOOP, UP, DOWN};

typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;


typedef struct {
    Log log;
    // 10 x 10 x 7
    int* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards;
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int* prev_action;
    bool use_minimal_action_set;
    float sticky_action_prob;
    int** cars; // 8 x 4
    int position;
    int move_timer;
    int terminate_timer;
    float episode_score;
} MinAtarFreeway;

void add_log(MinAtarFreeway* env) {
    env->log.perf += env->episode_score / (float)MAX_SCORE;
    env->log.score += env->episode_score;
    env->log.episode_length += env->terminate_timer;
    env->log.episode_return += env->episode_score;
    env->log.n++;
}

int random_int(int min, int max){
    // from: https://c-faq.com/lib/randrange.html
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int min(int a, int b) {
    if (a < b) {
        return a;
    }
    return b;
}

int max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

void init(MinAtarFreeway* env) {
    env->cars = (int**)(calloc(8, sizeof(int*)));
    for (int i = 0; i < 8; i++) {
        env->cars[i] = (int*)(calloc(4, sizeof(int)));
    }
    env->prev_action = (int*)calloc(1, sizeof(int));
}

void randomize_cars(MinAtarFreeway* env, bool initialize) {
    for (int i = 0; i < 8; i++) {
        int speed = random_int(1, 5);
        int direction = 2 * random_int(0, 1) - 1;
        if (initialize) {
            env->cars[i][0] = 0;
            env->cars[i][1] = i + 1;
        } 
        env->cars[i][2] = speed;
        env->cars[i][3] = speed * direction;
    }
    return;
}

int get_index(int h, int w, int c) {
    return h + 10 * w + 100 * c;
}

void get_obs(MinAtarFreeway* env) {
    memset(env->observations, 0, 10 * 10 * 7*sizeof(int));
    env->observations[get_index(env->position, 4, 0)] = 1;
    for (int i = 0; i < 8; i++) {
        env->observations[get_index(env->cars[i][1], env->cars[i][0], 1)] = 1;
        int back_x0;
        if (env->cars[i][3] > 0) {
            back_x0 = env->cars[i][0] - 1;
        } else {
            back_x0 = env->cars[i][0] + 1;
        }
        if (back_x0 < 0) {
            back_x0 = 9;
        } else if (back_x0 > 9) {
            back_x0 = 0;
        }
        int trail = abs(env->cars[i][3]) + 1;
        env->observations[get_index(env->cars[i][1], back_x0, trail)] = 1;
    }
}

void c_reset(MinAtarFreeway* env) {
    env->position = 9;
    env->episode_score = 0.0f;
    env->move_timer = PLAYER_SPEED;
    env->terminate_timer = 0;
    memset(env->prev_action, 0, sizeof(int));
    randomize_cars(env, true);
    get_obs(env);
}


void c_step(MinAtarFreeway* env) {
    env->terminals[0] = 0;
    int action;
    float reward = 0.0;

    if (rand() < ((RAND_MAX + 1u) * env->sticky_action_prob)){
        action = env->prev_action[0];
    } else {
        if (env->use_minimal_action_set) {
            action = MINIMAL_ACTION_SET[env->actions[0]];
        } else {
            action = FULL_ACTION_SET[env->actions[0]];
        }
    }
    env->prev_action[0] = action;

    // update player

    if (env->move_timer == 0) {
        env->move_timer = PLAYER_SPEED;
        if (action == 2) {
            env->position = max(0, env->position - 1);
        } else if (action == 4) {
            env->position = min(9, env->position + 1);
        }
    } else {
        env->move_timer--;
    }
    
    if (env->position == 0) {
        reward++;
        randomize_cars(env, false);
        env->position = 9;
    }

    // update cars
    for (int i = 0; i < 8; i++) {
        // player is always in column 4
        if ((env->cars[i][0] == 4) && (env->cars[i][1] == env->position)) {
            env->position = 9;
        } else if (env->cars[i][2] == 0) {
            env->cars[i][2] = abs(env->cars[i][3]);
            if (env->cars[i][3] > 0) {
                env->cars[i][0]++;
            } else {
                env->cars[i][0]--;
            }
            if (env->cars[i][0] < 0) {
                env->cars[i][0] = 9;
            } else if (env->cars[i][0] > 9) {
                env->cars[i][0] = 0;
            }
            if ((env->cars[i][0] == 4) && (env->cars[i][1] == env->position)) {
                env->position = 9;
            }
        } else {
            env->cars[i][2]--;
        }
    }

    env->terminate_timer++;
    env->rewards[0] = reward;
    env->episode_score += reward;
    if (env->terminate_timer > TIME_LIMIT) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    get_obs(env);
    return;
}

unsigned char U8(float x) {
    int v = (int)(x * 255.0f + 0.5f);
    if (v < 0) {
        v = 0;
    }
    if (v > 255) {
        v = 255;
    }
    return (unsigned char)v;
}

Color RGBf(float r, float g, float b) {
    return (Color){U8(r), U8(g), U8(b), 255};
}

void c_render(MinAtarFreeway* env) {
    if (!IsWindowReady()) {
        InitWindow(30 * 10, 30 * 10, "PufferLib MinAtar Freeway");
        SetTargetFPS(10);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    // from https://github.com/sotetsuk/pgx-minatar/blob/main/utils.py
    const Color palette[8] = {
        BLACK,
        RGBf(0.1041941874f, 0.1163201922f, 0.2327552016f),
        RGBf(0.0852351161f, 0.3266177900f, 0.2973201283f),
        RGBf(0.2653876155f, 0.4675654910f, 0.1908220645f),
        RGBf(0.6328422475f, 0.4747981096f, 0.2907020921f),
        RGBf(0.8306875711f, 0.5175161304f, 0.6628221029f),
        RGBf(0.7779565181f, 0.7069421943f, 0.9314406084f),
        RGBf(0.7964528048f, 0.9086689735f, 0.9398253501f),
    };
    BeginDrawing();
    ClearBackground(BLACK);

    for (int h = 0; h < 10; h++) {
        for (int w = 0; w < 10; w++) {
            int code = 0;
            for (int c = 0; c < 7; c++) {
                if (env->observations[get_index(h, w, c)]) {
                    code = c + 1;
                }
            }
            int x = w * 30;
            int y = h * 30;
            DrawRectangle(x, y, 30, 30, palette[code]);
        }
    }

    EndDrawing();
}

void c_close(MinAtarFreeway* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->prev_action);
    for (int i = 0; i < 8; i++) {
        free(env->cars[i]);
    }
    free(env->cars);
}
