
#include <stdbool.h>
#include <stdlib.h>
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
const unsigned int MAX_SCORE = (TIME_LIMIT / PLAYER_SPEED) / 8;

const unsigned char FULL_ACTION_SET[6] = {NOOP, LEFT, UP, RIGHT, DOWN, FIRE};
const unsigned char MINIMAL_ACTION_SET[3] = {NOOP, UP, DOWN};

// Required struct. Only use floats!
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
    int* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards;
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int* prev_action;
    bool use_minimal_action_set;
    float sticky_action_prob;
    int** cars;
    int position;
    int move_timer;
    int terminate_timer;
} MinAtarFreeway;

void add_log(MinAtarFreeway* env) {
    env->log.perf += env->rewards[0] / (float)MAX_SCORE;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->terminate_timer;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

int random(int min, int max){
    // from: https://c-faq.com/lib/randrange.html
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void randomize_cars(MinAtarFreeway* env, bool initialize) {
    for (int i = 0; i < 8; i++) {
        int speed = random(1, 5);
        int direction = 2 * random(0, 1) - 1;
        if (initialize) {
            env->cars[i][0] = 0;
            env->cars[i][1] = i + 1;
        } 
        env->cars[i][2] = speed;
        env->cars[i][3] = speed * direction;
    }
    return;
}

inline int get_index(int h, int w, int c) {
    return h + 10 * w + 100 * c;
}

void get_obs(MinAtarFreeway* env) {
    free(env->observations);
    env->observations = (int*)(calloc(10 * 10 * 7, sizeof(int)));
    env->observations[get_index(env->position, 4, 0)] = 1;
    for (int i = 0; i < 8; i++) {
        // todo
    }
}

// Required function
void c_reset(Squared* env) {
    int tiles = env->size*env->size;
    memset(env->observations, 0, tiles*sizeof(unsigned char));
    env->observations[tiles/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx;
    do {
        target_idx = rand() % tiles;
    } while (target_idx == tiles/2);
    env->observations[target_idx] = TARGET;
}


void c_step(MinAtarFreeway* env) {
    int action;
    int reward = 0;

    if (rand() < ((RAND_MAX + 1u) * env->sticky_action_prob)){
        action = env->prev_action[0];
    } else {
        if (env->use_minimal_action_set) {
            action = MINIMAL_ACTION_SET[env->actions[0]];
        } else {
            action = FULL_ACTION_SET[env->actions[0]];
        }
    }

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
        if ((env->cars[i][0] == 4) && (env->cars[i][4] == env->position)) {
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
            if ((env->cars[i][0] == 4) && (env->cars[i][4] == env->position)) {
                env->position = 9;
            }
        } else {
            env->cars[i][2]--;
        }
    }

    env->terminate_timer++;
    if (env->terminate_timer > TIME_LIMIT) {
        env->terminals[0] = 1;
    }
    // todo: get_obs
    add_log(env);
    return;
}

// Required function. Should handle creating the client on first call
void c_render(Squared* env) {
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Squared");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i*env->size + j];
            if (tex == EMPTY) {
                continue;
            }
            Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
            DrawRectangle(j*px, i*px, px, px, color);
        }
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Squared* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
