#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define PASS 0
#define NORTH 1
#define SOUTH 2
#define EAST 3
#define WEST 4

#define EMPTY 0
#define WALL 1
#define LAVA 2
#define GOAL 3
#define REWARD 4
#define OBJECT 5
#define AGENT_1 6
#define AGENT_2 7
#define AGENT_3 8
#define AGENT_4 9
#define AGENT_5 10
#define AGENT_6 11
#define AGENT_7 12
#define AGENT_8 13
#define KEY_1 14
#define KEY_2 15
#define KEY_3 16
#define KEY_4 17
#define KEY_5 19
#define KEY_6 19
#define DOOR_LOCKED_1 20
#define DOOR_LOCKED_2 21
#define DOOR_LOCKED_3 22
#define DOOR_LOCKED_4 23
#define DOOR_LOCKED_5 24
#define DOOR_LOCKED_6 25
#define DOOR_OPEN_1 26
#define DOOR_OPEN_2 27
#define DOOR_OPEN_3 28
#define DOOR_OPEN_4 29
#define DOOR_OPEN_5 30
#define DOOR_OPEN_6 31

#define NUM_AGENTS 8
#define NUM_OBJECTS 6
#define NUM_KEYS 6
#define NUM_DOORS 6

int rand_color() {
    return AGENT_1 + rand()%(AGENT_4 - AGENT_1 + 1);
}

int is_agent(int tile) {
    return tile >= AGENT_1 && tile <= AGENT_8;
}

int is_key(int tile) {
    return tile >= KEY_1 && tile <= KEY_6;
}

int is_locked_door(int tile) {
    return tile >= DOOR_LOCKED_1 && tile <= DOOR_LOCKED_6;
}

int is_open_door(int tile) {
    return tile >= DOOR_OPEN_1 && tile <= DOOR_OPEN_6;
}

typedef struct Agent Agent;
struct Agent {
    float y;
    float x;
    float spawn_y;
    float spawn_x;
    int color;
    int direction;
    int held_object;
    int keys[NUM_KEYS];
};

typedef struct Env Env;
struct Env {
    int width;
    int height;
    int num_agents;
    int horizon;
    int vision;
    float speed;
    bool discretize;
    int obs_size;

    int tick;
    float episode_return;

    unsigned char* grid;
    Agent* agents;
    unsigned char* observations;
    unsigned int* actions;
    float* rewards;
    float* dones;
};

Env* init_grid(
        unsigned char* observations, unsigned int* actions, float* rewards, float* dones,
        int width, int height, int num_agents, int horizon,
        int vision, float speed, bool discretize) {
    Env* env = (Env*)calloc(1, sizeof(Env));

    env->width = width;
    env->height = height;
    env->num_agents = num_agents;
    env->horizon = horizon;
    env->vision = vision;
    env->speed = speed;
    env->discretize = discretize;
    env->obs_size = 2*vision + 1;

    env->grid = (unsigned char*)calloc(width*height, sizeof(unsigned char));
    env->agents = (Agent*)calloc(num_agents, sizeof(Agent));
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->dones = dones;
    return env;
}

Env* allocate_grid(int width, int height, int num_agents, int horizon,
        int vision, float speed, bool discretize) {

    int obs_size = 2*vision + 1;
    unsigned char* observations = (unsigned char*)calloc(num_agents*obs_size*obs_size, sizeof(unsigned char));
    unsigned int* actions = (unsigned int*)calloc(num_agents, sizeof(unsigned int));
    float* rewards = (float*)calloc(num_agents, sizeof(float));
    float* dones = (float*)calloc(num_agents, sizeof(float));

    return init_grid(observations, actions, rewards, dones,
        width, height, num_agents, horizon, vision, speed, discretize);
}

void free_env(Env* env) {
    free(env->grid);
    free(env->agents);
    free(env);
}

void free_allocated_grid(Env* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
    free_env(env);
}

int grid_offset(Env* env, int y, int x) {
    return y*env->width + x;
}

void compute_observations(Env* env) {
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        float y = agent->y;
        float x = agent->x;
        int r = y;
        int c = x;

        int obs_offset = agent_idx*env->obs_size*env->obs_size;
        for (int dr = -env->vision; dr <= env->vision; dr++) {
            for (int dc = -env->vision; dc <= env->vision; dc++) {
                int rr = r + dr;
                int cc = c + dc;
                int adr = grid_offset(env, rr, cc);
                env->observations[obs_offset] = env->grid[adr];
                obs_offset++;
            }
        }
    }
}

void reset(Env* env, int seed) {
    env->tick = 0;
    env->episode_return = 0;

    // Add borders
    int left = env->speed * env->vision;
    int right = env->width - env->speed*env->vision - 1;
    int bottom = env->height - env->speed*env->vision - 1;
    for (int r = 0; r < left; r++) {
        for (int c = 0; c < env->width; c++) {
            int adr = grid_offset(env, r, c);
            env->grid[adr] = WALL;
        }
    }
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < left; c++) {
            int adr = grid_offset(env, r, c);
            env->grid[adr] = WALL;
        }
    }
    for (int c = right; c < env->width; c++) {
        for (int r = 0; r < env->height; r++) {
            env->grid[grid_offset(env, r, c)] = WALL;
        }
    }
    for (int r = bottom; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            env->grid[grid_offset(env, r, c)] = WALL;
        }
    }

    // Agent spawning
    for (int i = 0; i < env->num_agents; i++) {
        Agent* agent = &env->agents[i];
        int adr = grid_offset(env, agent->spawn_y, agent->spawn_x);
        assert(env->grid[adr] == EMPTY);
        assert(is_agent(agent->color));
        agent->y = agent->spawn_y;
        agent->x = agent->spawn_x;
        env->grid[adr] = agent->color;
        agent->direction = EAST;
        agent->held_object = -1;
        for (int j = 0; j < NUM_KEYS; j++) {
            agent->keys[j] = -1;
        }
    }
    compute_observations(env);
}

bool step(Env* env) {
    // TODO: Handle discrete vs continuous
    /*
    if self.discretize:
        actions_discrete = np_actions
    else:
        actions_continuous = np_actions
    */
    bool done = false;
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        // Discrete case only
        int atn = env->actions[agent_idx];
        float vel_y = 0;
        float vel_x = 0;
        if (atn == PASS) {
            continue;
        } else if (atn == SOUTH) {
            vel_y = 1;
        } else if (atn == NORTH) {
            vel_y = -1;
        } else if (atn == EAST) {
            vel_x = 1;
        } else if (atn == WEST) {
            vel_x = -1;
        } else {
            printf("Invalid action: %i\n", atn);
            exit(1);
        }

        Agent* agent = &env->agents[agent_idx];
        float y = agent->y;
        float x = agent->x;
        float dest_y = env->speed*vel_y + y;
        float dest_x = env->speed*vel_x + x;

        int adr = grid_offset(env, y, x);
        int dest_adr = grid_offset(env, dest_y, dest_x);
        int dest_tile = env->grid[dest_adr];

        if (dest_tile == REWARD || dest_tile == GOAL) {
            env->grid[dest_adr] = EMPTY;
            env->rewards[agent_idx] = 1.0;
            env->episode_return += 1.0;
            dest_tile = EMPTY;
            done = true;
        } else if (is_key(dest_tile)) {
            env->grid[dest_adr] = EMPTY;
            agent->keys[dest_tile - KEY_1] = 1;
            dest_tile = EMPTY;
        } else if (is_locked_door(dest_tile) && agent->keys[dest_tile - DOOR_LOCKED_1] == 1) {
            env->grid[dest_adr] = EMPTY;
            dest_tile = EMPTY;
        }

        if (dest_tile == EMPTY || is_open_door(dest_tile)) {
            env->grid[adr] = EMPTY;
            env->grid[dest_adr] = agent->color;

            // Continuous position update
            agent->y = dest_y;
            agent->x = dest_x;
        }
    }
    compute_observations(env);

    env->tick += 1;
    if (env->tick >= env->horizon) {
        done = true;
    }

    return done;
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 255, 0, 255},
};

Rectangle UV_COORDS[7] = {
    (Rectangle){0, 0, 0, 0},
    (Rectangle){512, 0, 128, 128},
    (Rectangle){0, 0, 0, 0},
    (Rectangle){0, 0, 128, 128},
    (Rectangle){128, 0, 128, 128},
    (Rectangle){256, 0, 128, 128},
    (Rectangle){384, 0, 128, 128},
};

typedef struct {
    int cell_size;
    int width;
    int height;
    Texture2D puffer;
} Renderer;

Renderer* init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;

    InitWindow(width*cell_size, height*cell_size, "PufferLib Ray Grid");
    SetTargetFPS(10);

    renderer->puffer = LoadTexture("minipuff.png");
    return renderer;
}

void close_renderer(Renderer* renderer) {
    CloseWindow();
    free(renderer);
}

void render_global(Renderer* renderer, Env* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int ts = renderer->cell_size;
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++){
            int adr = grid_offset(env, r, c);
            int tile = env->grid[adr];
            if (tile == EMPTY) {
                continue;
            } else if (tile == WALL) {
                DrawRectangle(c*ts, r*ts, ts, ts, (Color){128, 128, 128, 255});
            } else {
                int u = 128*(tile % 8);
                int v = 128*(tile / 8);
                Rectangle source_rect = (Rectangle){u, v, 128, 128};
                Rectangle dest_rect = (Rectangle){c*ts, r*ts, ts, ts};
                DrawTexturePro(renderer->puffer, source_rect, dest_rect,
                    (Vector2){0, 0}, 0, WHITE);
            }
        }
    }
    EndDrawing();
}

// Presets
Env* make_locked_room_env(unsigned char* observations,
        unsigned int* actions, float* rewards, float* dones) {
    int width = 19;
    int height = 19;
    int num_agents = 1;
    int horizon = 9999999;
    float agent_speed = 1;
    int vision = 3;
    bool discretize = true;

    Env* env = init_grid(observations, actions, rewards, dones,
        width, height, num_agents, horizon, vision, agent_speed, discretize);
 
    env->agents[0].spawn_y = 9+vision;
    env->agents[0].spawn_x = 9+vision;
    env->agents[0].color = AGENT_1;

    for (int r = 0; r < env->height; r++) {
        int adr = grid_offset(env, r, 7+vision);
        env->grid[adr] = WALL;
        adr = grid_offset(env, r, 11+vision);
        env->grid[adr] = WALL;
    }
    for (int c = 0+vision; c < 7+vision; c++) {
        int adr = grid_offset(env, 6+vision, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12+vision, c);
        env->grid[adr] = WALL;
    }
    for (int c = 11+vision; c < env->width; c++) {
        int adr = grid_offset(env, 6+vision, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12+vision, c);
        env->grid[adr] = WALL;
    }
    int adr = grid_offset(env, 3+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_1;
    adr = grid_offset(env, 9+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_2;
    adr = grid_offset(env, 15+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_3;
    adr = grid_offset(env, 3+vision, 11+vision);
    env->grid[adr] = DOOR_OPEN_4;
    adr = grid_offset(env, 9+vision, 11+vision);
    env->grid[adr] = DOOR_OPEN_5;
    adr = grid_offset(env, 15+vision, 11+vision);
    env->grid[adr] = DOOR_LOCKED_6;

    adr = grid_offset(env, 4+vision, 15+vision);
    env->grid[adr] = KEY_6;

    adr = grid_offset(env, 16+vision, 17+vision);
    env->grid[adr] = GOAL;

    return env;
}

Env* alloc_locked_room_env() {
    int width = 19;
    int height = 19;
    int num_agents = 1;
    int horizon = 512;
    float agent_speed = 1;
    int vision = 3;
    bool discretize = true;

    Env* env = allocate_grid(width+2*vision, height+2*vision, num_agents, horizon,
            vision, agent_speed, discretize);

    env->agents[0].spawn_y = 9+vision;
    env->agents[0].spawn_x = 9+vision;
    env->agents[0].color = AGENT_1;
    return env;
}

void reset_locked_room(Env* env) {
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int adr = grid_offset(env, r, c);
            env->grid[adr] = EMPTY;
        }
    }
    reset(env, 0);

    int vision = 3;
    int adr = grid_offset(env, 7+vision, 9+vision);
    //env->grid[adr] = GOAL;

    for (int r = 0; r < env->height; r++) {
        int adr = grid_offset(env, r, 7+vision);
        env->grid[adr] = WALL;
        adr = grid_offset(env, r, 11+vision);
        env->grid[adr] = WALL;
    }
    for (int c = 0+vision; c < 7+vision; c++) {
        int adr = grid_offset(env, 6+vision, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12+vision, c);
        env->grid[adr] = WALL;
    }
    for (int c = 11+vision; c < env->width; c++) {
        int adr = grid_offset(env, 6+vision, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12+vision, c);
        env->grid[adr] = WALL;
    }
    adr = grid_offset(env, 3+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_1;
    adr = grid_offset(env, 9+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_2;
    adr = grid_offset(env, 15+vision, 7+vision);
    env->grid[adr] = DOOR_OPEN_3;
    adr = grid_offset(env, 3+vision, 11+vision);
    env->grid[adr] = DOOR_OPEN_4;
    adr = grid_offset(env, 9+vision, 11+vision);
    env->grid[adr] = DOOR_OPEN_5;
    adr = grid_offset(env, 15+vision, 11+vision);
    env->grid[adr] = DOOR_LOCKED_6;

    adr = grid_offset(env, 4+vision, 15+vision);
    env->grid[adr] = KEY_6;

    adr = grid_offset(env, 16+vision, 17+vision);
    env->grid[adr] = GOAL;
}

void free_envs(Env** envs, int num_envs) {
    for (int i = 0; i < num_envs; i++) {
        free_env(envs[i]);
    }
    free(envs);
}

Env** make_locked_rooms(unsigned char* observations,
        unsigned int* actions, float* rewards, float* dones, int num_envs) {
    Env** envs = (Env**)calloc(num_envs, sizeof(Env*));
    for (int i = 0; i < num_envs; i++) {
        envs[i] = alloc_locked_room_env();
        envs[i]->observations = &observations[i*7*7];
        envs[i]->actions = &actions[i];
        envs[i]->rewards = &rewards[i];
        envs[i]->dones = &dones[i];
    }
    return envs;
}


