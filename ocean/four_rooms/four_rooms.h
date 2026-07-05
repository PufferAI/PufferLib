#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"

#define FOUR_ROOMS_VIEW_SIZE 7
#define FOUR_ROOMS_OBS_CHANNELS 3
#define FOUR_ROOMS_NUM_ACTIONS 7
#define FOUR_ROOMS_TIMEOUT_SCALE 4

enum {
    LEFT = 0,
    RIGHT = 1,
    FORWARD = 2,
    PICKUP = 3,
    DROP = 4,
    TOGGLE = 5,
    DONE = 6,
};

enum {
    UNSEEN = 0,
    EMPTY = 1,
    WALL = 2,
    GOAL = 8,
    AGENT = 10,
};

enum {
    COLOR_BLACK = 0,
    COLOR_GREEN = 1,
    COLOR_GREY = 5,
};

static const Color PUFF_RED = (Color){187, 0, 0, 255};
static const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
static const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int size;
    int max_steps;
    int min_steps;
    int episode_steps;
    int tick;
    int steps_since_goal;
    int goals;
    float episode_return;
    int agent_x, agent_y;
    int agent_dir;
    int goal_x, goal_y;
    unsigned char* grid;
    unsigned int rng;
    int texture_loaded;
    Texture2D puffers;
} FourRooms;

static inline int four_rooms_rand(FourRooms* env, int n) {
    return rand_r(&env->rng) % n;
}

static inline int grid_idx(FourRooms* env, int x, int y) {
    return y * env->size + x;
}

void init(FourRooms* env) {
    assert(env->size >= 5);
    if (env->max_steps <= 0) {
        env->max_steps = FOUR_ROOMS_TIMEOUT_SCALE * env->size;
    }
    if (env->min_steps <= 0 || env->min_steps > env->max_steps) {
        env->min_steps = env->max_steps;
    }
    env->grid = (unsigned char*)calloc(env->size * env->size, sizeof(unsigned char));
}

void allocate(FourRooms* env) {
    init(env);
    env->observations = (unsigned char*)calloc(
        FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_OBS_CHANNELS,
        sizeof(unsigned char)
    );
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void add_log(FourRooms* env) {
    env->log.perf += env->tick > 0 ? (float)env->goals / (float)env->tick : 0.0f;
    env->log.score += env->goals;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

static inline void encode_cell(unsigned char object, unsigned char* object_idx,
        unsigned char* color_idx, unsigned char* state) {
    *state = 0;
    if (object == WALL) {
        *object_idx = WALL;
        *color_idx = COLOR_GREY;
    } else if (object == GOAL) {
        *object_idx = GOAL;
        *color_idx = COLOR_GREEN;
    } else {
        *object_idx = EMPTY;
        *color_idx = COLOR_BLACK;
    }
}

static inline void observation_to_world(FourRooms* env, int obs_x, int obs_y,
        int* world_x, int* world_y) {
    int forward_x = 0;
    int forward_y = 0;
    if (env->agent_dir == 0) forward_x = 1;
    else if (env->agent_dir == 1) forward_y = 1;
    else if (env->agent_dir == 2) forward_x = -1;
    else forward_y = -1;

    int right_x = -forward_y;
    int right_y = forward_x;
    int right_offset = obs_x - FOUR_ROOMS_VIEW_SIZE / 2;
    int forward_offset = FOUR_ROOMS_VIEW_SIZE - 1 - obs_y;

    *world_x = env->agent_x + forward_x * forward_offset + right_x * right_offset;
    *world_y = env->agent_y + forward_y * forward_offset + right_y * right_offset;
}

static inline void compute_visibility(unsigned char view[FOUR_ROOMS_VIEW_SIZE][FOUR_ROOMS_VIEW_SIZE],
        unsigned char visible[FOUR_ROOMS_VIEW_SIZE][FOUR_ROOMS_VIEW_SIZE]) {
    memset(visible, 0, FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_VIEW_SIZE * sizeof(unsigned char));
    visible[FOUR_ROOMS_VIEW_SIZE - 1][FOUR_ROOMS_VIEW_SIZE / 2] = 1;

    // MiniGrid propagates visibility from the agent at bottom-center after rotating the view.
    for (int y = FOUR_ROOMS_VIEW_SIZE - 1; y >= 0; y--) {
        for (int x = 0; x < FOUR_ROOMS_VIEW_SIZE - 1; x++) {
            if (!visible[y][x] || view[y][x] == WALL) {
                continue;
            }
            visible[y][x + 1] = 1;
            if (y > 0) {
                visible[y - 1][x] = 1;
                visible[y - 1][x + 1] = 1;
            }
        }

        for (int x = FOUR_ROOMS_VIEW_SIZE - 1; x > 0; x--) {
            if (!visible[y][x] || view[y][x] == WALL) {
                continue;
            }
            visible[y][x - 1] = 1;
            if (y > 0) {
                visible[y - 1][x] = 1;
                visible[y - 1][x - 1] = 1;
            }
        }
    }
}

void generate_observation(FourRooms* env) {
    unsigned char view[FOUR_ROOMS_VIEW_SIZE][FOUR_ROOMS_VIEW_SIZE];
    unsigned char visible[FOUR_ROOMS_VIEW_SIZE][FOUR_ROOMS_VIEW_SIZE];

    for (int y = 0; y < FOUR_ROOMS_VIEW_SIZE; y++) {
        for (int x = 0; x < FOUR_ROOMS_VIEW_SIZE; x++) {
            int world_x, world_y;
            observation_to_world(env, x, y, &world_x, &world_y);
            if (world_x < 0 || world_x >= env->size || world_y < 0 || world_y >= env->size) {
                view[y][x] = WALL;
            } else if (world_x == env->agent_x && world_y == env->agent_y) {
                view[y][x] = EMPTY;
            } else {
                view[y][x] = env->grid[grid_idx(env, world_x, world_y)];
            }
        }
    }

    compute_visibility(view, visible);

    for (int y = 0; y < FOUR_ROOMS_VIEW_SIZE; y++) {
        for (int x = 0; x < FOUR_ROOMS_VIEW_SIZE; x++) {
            int base_idx = (y * FOUR_ROOMS_VIEW_SIZE + x) * FOUR_ROOMS_OBS_CHANNELS;
            if (!visible[y][x]) {
                env->observations[base_idx] = UNSEEN;
                env->observations[base_idx + 1] = COLOR_BLACK;
                env->observations[base_idx + 2] = 0;
                continue;
            }

            encode_cell(
                view[y][x],
                &env->observations[base_idx],
                &env->observations[base_idx + 1],
                &env->observations[base_idx + 2]
            );
        }
    }
}

void create_four_rooms_grid(FourRooms* env) {
    int size = env->size;

    memset(env->grid, EMPTY, size * size * sizeof(unsigned char));

    for (int i = 0; i < size; i++) {
        env->grid[i] = WALL;
        env->grid[(size - 1) * size + i] = WALL;
        env->grid[i * size] = WALL;
        env->grid[i * size + size - 1] = WALL;
    }

    int room_w = size / 2;
    int room_h = size / 2;

    for (int y = 0; y < size; y++) {
        env->grid[y * size + room_w] = WALL;
    }

    for (int x = 0; x < size; x++) {
        env->grid[room_h * size + x] = WALL;
    }

    // One doorway per half-wall, excluding the outer border.
    int gap_y1 = 1 + four_rooms_rand(env, room_h - 1);
    env->grid[gap_y1 * size + room_w] = EMPTY;

    int gap_y2 = room_h + 1 + four_rooms_rand(env, size - room_h - 2);
    env->grid[gap_y2 * size + room_w] = EMPTY;

    int gap_x1 = 1 + four_rooms_rand(env, room_w - 1);
    env->grid[room_h * size + gap_x1] = EMPTY;

    int gap_x2 = room_w + 1 + four_rooms_rand(env, size - room_w - 2);
    env->grid[room_h * size + gap_x2] = EMPTY;
}

void place_goal(FourRooms* env) {
    do {
        env->goal_x = 1 + four_rooms_rand(env, env->size - 2);
        env->goal_y = 1 + four_rooms_rand(env, env->size - 2);
    } while (env->grid[grid_idx(env, env->goal_x, env->goal_y)] != EMPTY ||
             (env->goal_x == env->agent_x && env->goal_y == env->agent_y));

    env->grid[grid_idx(env, env->goal_x, env->goal_y)] = GOAL;
}

void c_reset(FourRooms* env) {
    create_four_rooms_grid(env);

    do {
        env->agent_x = 1 + four_rooms_rand(env, env->size - 2);
        env->agent_y = 1 + four_rooms_rand(env, env->size - 2);
    } while (env->grid[grid_idx(env, env->agent_x, env->agent_y)] != EMPTY);

    env->grid[grid_idx(env, env->agent_x, env->agent_y)] = AGENT;
    place_goal(env);

    env->agent_dir = four_rooms_rand(env, 4);
    env->tick = 0;
    env->steps_since_goal = 0;
    env->goals = 0;
    env->episode_steps = env->min_steps + four_rooms_rand(env, env->max_steps - env->min_steps + 1);
    env->episode_return = 0.0f;

    generate_observation(env);
}

void c_step(FourRooms* env) {
    env->tick += 1;
    env->steps_since_goal += 1;

    int action = (int)env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    env->grid[grid_idx(env, env->agent_x, env->agent_y)] = EMPTY;

    int new_x = env->agent_x;
    int new_y = env->agent_y;
    int new_dir = env->agent_dir;

    if (action == LEFT) {
        new_dir = (env->agent_dir + 3) % 4;
    } else if (action == RIGHT) {
        new_dir = (env->agent_dir + 1) % 4;
    } else if (action == FORWARD) {
        if (env->agent_dir == 0) new_x += 1;
        else if (env->agent_dir == 1) new_y += 1;
        else if (env->agent_dir == 2) new_x -= 1;
        else if (env->agent_dir == 3) new_y -= 1;

        if (new_x >= 0 && new_x < env->size && new_y >= 0 && new_y < env->size &&
            env->grid[grid_idx(env, new_x, new_y)] != WALL) {
            env->agent_x = new_x;
            env->agent_y = new_y;
        }
    }

    env->agent_dir = new_dir;

    if (env->agent_x == env->goal_x && env->agent_y == env->goal_y) {
        env->rewards[0] = 1.0f - 0.9f * (float)env->steps_since_goal / (float)env->max_steps;
        env->goals += 1;
        env->steps_since_goal = 0;
        env->grid[grid_idx(env, env->agent_x, env->agent_y)] = AGENT;
        place_goal(env);
    } else {
        env->grid[grid_idx(env, env->agent_x, env->agent_y)] = AGENT;
    }

    if (env->tick >= env->episode_steps) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0;
        env->episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }

    env->episode_return += env->rewards[0];
    generate_observation(env);
}

void c_render(FourRooms* env) {
    if (!IsWindowReady()) {
        InitWindow(32*env->size, 32*env->size, "PufferLib FourRooms");
        SetTargetFPS(10);
        env->puffers = LoadTexture("resources/shared/puffers_128.png");
        env->texture_loaded = 1;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int px = 32;

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            int cell = env->grid[y * env->size + x];
            Color color = PUFF_BACKGROUND;

            if (cell == WALL) color = PUFF_BACKGROUND2;
            else if (cell == GOAL) color = PUFF_RED;

            if (cell != EMPTY && cell != AGENT) {
                DrawRectangle(x*px, y*px, px, px, color);
            }
        }
    }

    Color obs_overlay = (Color){180, 180, 180, 80};
    for (int y = 0; y < FOUR_ROOMS_VIEW_SIZE; y++) {
        for (int x = 0; x < FOUR_ROOMS_VIEW_SIZE; x++) {
            int world_x, world_y;
            observation_to_world(env, x, y, &world_x, &world_y);
            if (world_x >= 0 && world_x < env->size && world_y >= 0 && world_y < env->size) {
                DrawRectangle(world_x*px, world_y*px, px, px, obs_overlay);
            }
        }
    }

    int starting_sprite_x = 0;
    int rotation = 90 * env->agent_dir;
    if (rotation == 180) {
        starting_sprite_x = 128;
        rotation = 0;
    }

    DrawTexturePro(
        env->puffers,
        (Rectangle){starting_sprite_x, 0, 128, 128},
        (Rectangle){
            env->agent_x * px + px/2,
            env->agent_y * px + px/2,
            px,
            px
        },
        (Vector2){px/2, px/2},
        rotation,
        WHITE
    );

    EndDrawing();
}

void c_close(FourRooms* env) {
    if (env->texture_loaded) {
        UnloadTexture(env->puffers);
        env->texture_loaded = 0;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env->grid) {
        free(env->grid);
        env->grid = NULL;
    }
}

void free_allocated(FourRooms* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}
