#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

#define TWO_PI 2.0*PI
#define MAX_SIZE 40

#define ATN_PASS 0
#define ATN_FORWARD 1
#define ATN_LEFT 2
#define ATN_RIGHT 3
#define ATN_BACK 4

#define DIR_WEST 0.0;
#define DIR_NORTH PI/2.0;
#define DIR_EAST PI;
#define DIR_SOUTH 3.0*PI/2.0;

#define EMPTY 0
#define WALL 1
#define LAVA 2
#define GOAL 3
#define REWARD 4
#define OBJECT 5
#define AGENT 6
#define KEY 14
#define DOOR_LOCKED 20
#define DOOR_OPEN 26

#define LOG_BUFFER_SIZE 4096

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

// 8 unique agents
bool is_agent(int idx) {
    return idx >= AGENT && idx < AGENT + 8;
}
int rand_color() {
    return AGENT + rand()%8;
}

// 6 unique keys and doors
bool is_key(int idx) {
    return idx >= KEY && idx < KEY + 6;
}
bool is_locked_door(int idx) {
    return idx >= DOOR_LOCKED && idx < DOOR_LOCKED + 6;
}
bool is_open_door(int idx) {
    return idx >= DOOR_OPEN && idx <= DOOR_OPEN + 6;
}
bool is_correct_key(int key, int door) {
    return key == door - 6;
}

typedef struct Agent Agent;
struct Agent {
    float y;
    float x;
    float prev_y;
    float prev_x;
    float spawn_y;
    float spawn_x;
    int color;
    float direction;
    int held;
};

typedef struct Renderer Renderer;
typedef struct State State;
typedef struct Grid Grid;
struct Grid{
    Renderer* renderer;
    State* levels;
    int num_maps;
    int width;
    int height;
    int num_agents;
    int horizon;
    int vision;
    int tick;
    float speed;
    int obs_size;
    int max_size;
    bool discretize;
    Log log;
    Agent* agents;
    unsigned char* grid;
    int* counts;
    unsigned char* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
};

void init_grid(Grid* env) {
    env->num_agents = 1;
    env->vision = 5;
    env->speed = 1;
    env->discretize = true;
    env->obs_size = 2*env->vision + 1;
    int env_mem= env->max_size * env->max_size;
    env->grid = calloc(env_mem, sizeof(unsigned char));
    env->counts = calloc(env_mem, sizeof(int));
    env->agents = calloc(env->num_agents, sizeof(Agent));
}

Grid* allocate_grid(int max_size, int num_agents, int horizon,
        int vision, float speed, bool discretize) {
    Grid* env = (Grid*)calloc(1, sizeof(Grid));
    env->max_size = max_size;
    env->num_agents = num_agents;
    env->horizon = horizon;
    env->vision = vision;
    env->speed = speed;
    env->discretize = discretize;
    int obs_size = 2*vision + 1;
    env->observations = calloc(
        num_agents*obs_size*obs_size, sizeof(unsigned char));
    env->actions = calloc(num_agents, sizeof(float));
    env->rewards = calloc(num_agents, sizeof(float));
    env->terminals = calloc(num_agents, sizeof(unsigned char));
    init_grid(env);
    return env;
}

void free_env(Grid* env) {
    free(env->grid);
    free(env->agents);
    free(env);
}

void free_allocated_grid(Grid* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_env(env);
}

bool in_bounds(Grid* env, int y, int c) {
    return (y >= 0 && y <= env->height
        && c >= 0 && c <= env->width);
}

int grid_offset(Grid* env, int y, int x) {
    return y*env->max_size + x;
}

void add_log(Grid* env, int idx) {
    env->log.perf += env->rewards[idx];
    env->log.score += env->rewards[idx];
    env->log.episode_return += env->rewards[idx];
    env->log.episode_length += env->tick;
    env->log.n += 1.0;
}
 
void compute_observations(Grid* env) {
    memset(env->observations, 0, env->obs_size*env->obs_size*env->num_agents);
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        float y = agent->y;
        float x = agent->x;
        int start_r = y - env->vision;
        if (start_r < 0) {
            start_r = 0;
        }

        int start_c = x - env->vision;
        if (start_c < 0) {
            start_c = 0;
        }

        int end_r = y + env->vision;
        if (end_r >= env->max_size) {
            end_r = env->max_size - 1;
        }

        int end_c = x + env->vision;
        if (end_c >= env->max_size) {
            end_c = env->max_size - 1;
        }

        int obs_offset = agent_idx*env->obs_size*env->obs_size;
        for (int r = start_r; r <= end_r; r++) {
            for (int c = start_c; c <= end_c; c++) {
                int r_idx = r - y + env->vision;
                int c_idx = c - x + env->vision;
                int obs_adr = obs_offset + r_idx*env->obs_size + c_idx;
                int adr = grid_offset(env, r, c);
                env->observations[obs_adr] = env->grid[adr];
            }
        }
        /*
        int obs_adr = 0;
        for (int r = 0; r < env->obs_size; r++) {
            for (int c = 0; c < env->obs_size; c++) {
                printf("%d ", env->observations[obs_adr]);
                obs_adr++;
            }
            printf("\n");
        }
        */
    }
}

void make_border(Grid*env) {
    for (int r = 0; r < env->height; r++) {
        int adr = grid_offset(env, r, 0);
        env->grid[adr] = WALL;
        adr = grid_offset(env, r, env->width-1);
        env->grid[adr] = WALL;
    }
    for (int c = 0; c < env->width; c++) {
        int adr = grid_offset(env, 0, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, env->height-1, c);
        env->grid[adr] = WALL;
    }
}

void spawn_agent(Grid* env, int idx, int x, int y) {
    Agent* agent = &env->agents[idx];
    int spawn_y = y;
    int spawn_x = x;
    assert(in_bounds(env, spawn_y, spawn_x));
    int adr = grid_offset(env, spawn_y, spawn_x);
    assert(env->grid[adr] == EMPTY);
    agent->spawn_y = spawn_y;
    agent->spawn_x = spawn_x;
    agent->y = agent->spawn_y;
    agent->x = agent->spawn_x;
    agent->prev_y = agent->y;
    agent->prev_x = agent->x;
    env->grid[adr] = agent->color;
    agent->direction = 0;
    agent->held = -1;
    agent->color = AGENT;
}

struct State {
    int width;
    int height;
    int num_agents;
    Agent* agents;
    unsigned char* grid;
};

void init_state(State* state, int max_size, int num_agents) {
    state->agents = calloc(num_agents, sizeof(Agent));
    state->grid = calloc(max_size*max_size, sizeof(unsigned char));
}

void free_state(State* state) {
    free(state->agents);
    free(state->grid);
    free(state);
}

void get_state(Grid* env, State* state) {
    state->width = env->width;
    state->height = env->height;
    state->num_agents = env->num_agents;
    memcpy(state->agents, env->agents, env->num_agents*sizeof(Agent));
    memcpy(state->grid, env->grid, env->max_size*env->max_size);
}

void set_state(Grid* env, State* state) {
    env->width = state->width;
    env->height = state->height;
    env->horizon = 2*env->width*env->height;
    env->num_agents = state->num_agents;
    memcpy(env->agents, state->agents, env->num_agents*sizeof(Agent));
    memcpy(env->grid, state->grid, env->max_size*env->max_size);
}

void c_reset(Grid* env) {
    memset(env->grid, 0, env->max_size*env->max_size);
    memset(env->counts, 0, env->max_size*env->max_size*sizeof(int));
    env->tick = 0;
    int idx = rand() % env->num_maps;
    set_state(env, &env->levels[idx]);
    compute_observations(env);
}

int move_to(Grid* env, int agent_idx, float y, float x) {
    Agent* agent = &env->agents[agent_idx];
    if (!in_bounds(env, y, x)) {
        return 1;
    }

    int adr = grid_offset(env, round(y), round(x));
    int dest = env->grid[adr];
    if (dest == WALL) {
        return 1;
    } else if (dest == REWARD || dest == GOAL) {
        env->rewards[agent_idx] = 1.0;
        env->terminals[agent_idx] = 1;
        add_log(env, agent_idx);
    } else if (is_key(dest)) {
        if (agent->held != -1) {
            return 1;
        }
        agent->held = dest;
    } else if (is_locked_door(dest)) { if (!is_correct_key(agent->held, dest)) { return 1;
        }
        agent->held = -1;
        env->grid[adr] = EMPTY;
    }

    int start_y = round(agent->y);
    int start_x = round(agent->x);
    int start_adr = grid_offset(env, start_y, start_x);
    env->grid[start_adr] = EMPTY;

    env->grid[adr] = agent->color;
    agent->y = y;
    agent->x = x;
    return 0;
}
 
bool step_agent(Grid* env, int idx) {
    Agent* agent = &env->agents[idx];
    agent->prev_y = agent->y;
    agent->prev_x = agent->x;

    float atn = env->actions[idx];
    float direction = agent->direction;

    if (env->discretize) {
        int iatn = (int)atn;
        if (iatn == ATN_PASS) {
            return true;
        } else if (iatn == ATN_FORWARD) {
        } else if (iatn == ATN_LEFT) {
            direction -= PI/2.0;
        } else if (iatn == ATN_RIGHT) {
            direction += PI/2.0;
        } else if (iatn == ATN_BACK) {
            direction += PI;
        } else {
            printf("Invalid action: %f\n", atn);
            exit(1);
        }
        if (direction < 0) {
            direction += TWO_PI;
        } else if (direction >= TWO_PI) {
            direction -= TWO_PI;
        }
    } else {
        assert(atn >= -1.0);
        assert(atn <= 1.0);
        direction += PI*atn;
    }

    float x = agent->x;
    float y = agent->y;
    float dx = env->speed*cos(direction);
    float dy = env->speed*sin(direction);
    agent->direction = direction;
    if (env->discretize) {
        float dest_x = x + dx;
        float dest_y = y + dy;
        if (!in_bounds(env, dest_y, dest_x)) {
            return false;
        }
        int err = move_to(env, idx, dest_y, dest_x);
        if (err) {
            return false;
        }
    } else {
        for (int substep = 1; substep <= 4; substep++) {
            float dest_x = x + dx/(float)substep;
            float dest_y = y + dy/(float)substep;
            int err = move_to(env, idx, dest_y, dest_x);
            if (!err) {
                continue;
            } else if (substep == 1) {
                return false;
            } else {
                break;
            }
        }
    }

    int x_int = agent->x;
    int y_int = agent->y;
    int adr = grid_offset(env, y_int, x_int);
    env->counts[adr]++;
    //env->rewards[idx] += 0.01 / (float)env->counts[adr];
    //env->log.episode_return += 0.01 / (float)env->counts[adr];
    return true;
}

void c_step(Grid* env) {
    memset(env->terminals, 0, env->num_agents);
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    env->tick++;

    for (int i = 0; i < env->num_agents; i++) {
        step_agent(env, i);
    }
    compute_observations(env);

    bool done = true;
    for (int i = 0; i < env->num_agents; i++) {
        if (!env->terminals[i]) {
            done = false;
            break;
        }
    }

    if (env->tick >= env->horizon) {
        done = true;
        add_log(env, 0);
    }

    if (done) {
        c_reset(env);
        int idx = rand() % env->num_maps;
        set_state(env, &env->levels[idx]);
        compute_observations(env);
    }
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

struct Renderer {
    int cell_size;
    int width;
    int height;
    Texture2D puffer;
    float* overlay;
};

Renderer* init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;

    renderer->overlay = (float*)calloc(width*height, sizeof(float));

    InitWindow(width*cell_size, height*cell_size, "PufferLib Ray Grid");
    SetTargetFPS(60);

    renderer->puffer = LoadTexture("resources/puffers_128.png");
    return renderer;
}

void clear_overlay(Renderer* renderer) {
    memset(renderer->overlay, 0, renderer->width*renderer->height*sizeof(float));
}

void close_renderer(Renderer* renderer) {
    CloseWindow();
    free(renderer->overlay);
    free(renderer);
}

void c_render(Grid* env) {
    // TODO: fractional rendering
    float frac = 0.0;
    float overlay = 0.0;
    if (env->renderer == NULL) {
        env->renderer = init_renderer(16, env->width, env->height);
    }
    Renderer* renderer = env->renderer;
 
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Agent* agent = &env->agents[0];
    int r = agent->y;
    int c = agent->x;
    int adr = grid_offset(env, r, c);
    //renderer->overlay[adr] = overlay;
    //renderer->overlay[adr] -= 0.1;
    //renderer->overlay[adr] = -1 + 1.0/(float)env->counts[adr];

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int ts = renderer->cell_size;
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++){
            adr = grid_offset(env, r, c);
            int tile = env->grid[adr];
            if (tile == EMPTY) {
                continue;
                overlay = renderer->overlay[adr];
                if (overlay == 0) {
                    continue;
                }
                Color color;
                if (overlay < 0) {
                    overlay = -fmaxf(-1.0, overlay);
                    color = (Color){255.0*overlay, 0, 0, 255};
                } else {
                    overlay = fminf(1.0, overlay);
                    color = (Color){0, 255.0*overlay, 0, 255};
                }
                DrawRectangle(c*ts, r*ts, ts, ts, color);
            }

            Color color;
            if (tile == WALL) {
                color = (Color){128, 128, 128, 255};
            } else if (tile == GOAL) {
                color = GREEN;
            } else if (is_locked_door(tile)) {
                int weight = 40*(tile - DOOR_LOCKED);
                color = (Color){weight, 0, 0, 255};
            } else if (is_open_door(tile)) {
                int weight = 40*(tile - DOOR_OPEN);
                color = (Color){0, weight, 0, 255};
            } else if (is_key(tile)) {
                int weight = 40*(tile - KEY);
                color = (Color){0, 0, weight, 255};
            } else {
                continue;
            }
 
            DrawRectangle(c*ts, r*ts, ts, ts, color);
       }
    }

    for (int i = 0; i < env->num_agents; i++) {
        agent = &env->agents[0];
        float y = agent->y + (frac - 1)*(agent->y - agent->prev_y);
        float x = agent->x + (frac - 1)*(agent->x - agent->prev_x);
        int u = 0;
        int v = 0;
        Rectangle source_rect = (Rectangle){u, v, 128, 128};
        Rectangle dest_rect = (Rectangle){x*ts, y*ts, ts, ts};
        DrawTexturePro(renderer->puffer, source_rect, dest_rect,
            (Vector2){0, 0}, 0, WHITE);
    }
 
    EndDrawing();
}

void generate_locked_room(Grid* env) {
    assert(env->max_size >= 19);
    env->width = 19;
    env->height = 19;
    env->num_agents = 1;
    env->horizon = 1000;
    env->speed = 1;
    env->vision = 3;
    env->discretize = true;

    Agent* agent = &env->agents[0];
    agent->x = 9;
    agent->y = 9;
    agent->prev_x = 9;
    agent->prev_y = 9;
    agent->spawn_y = 9;
    agent->spawn_x = 9;
    agent->color = 6;
    agent->held = -1;

    make_border(env);

    for (int r = 0; r < env->height; r++) {
        int adr = grid_offset(env, r, 7);
        env->grid[adr] = WALL;
        adr = grid_offset(env, r, 11);
        env->grid[adr] = WALL;
    }
    for (int c = 0; c < 7; c++) {
        int adr = grid_offset(env, 6, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12, c);
        env->grid[adr] = WALL;
    }
    for (int c = 11; c < env->width; c++) {
        int adr = grid_offset(env, 6, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, 12, c);
        env->grid[adr] = WALL;
    }
    int adr = grid_offset(env, 3, 7);
    env->grid[adr] = DOOR_OPEN;
    adr = grid_offset(env, 9, 7);
    env->grid[adr] = DOOR_OPEN + 1;
    adr = grid_offset(env, 15, 7);
    env->grid[adr] = DOOR_OPEN + 2;
    adr = grid_offset(env, 3, 11);
    env->grid[adr] = DOOR_OPEN + 3;
    adr = grid_offset(env, 9, 11);
    env->grid[adr] = DOOR_OPEN + 4;
    adr = grid_offset(env, 15, 11);
    env->grid[adr] = DOOR_LOCKED + 5;

    adr = grid_offset(env, 4, 15);
    env->grid[adr] = KEY + 5;

    adr = grid_offset(env, 16, 17);
    env->grid[adr] = GOAL;
}

void generate_growing_tree_maze(unsigned char* grid,
        int width, int height, int max_size, float difficulty, int seed) {
    srand(seed);
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    int dirs[4] = {0, 1, 2, 3};
    int cells[2*width*height];
    int num_cells = 1;

    bool visited[width*height];
    memset(visited, false, width*height);

    memset(grid, WALL, max_size*height);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*max_size + c;
            if (r % 2 == 1 && c % 2 == 1) {
                grid[adr] = EMPTY;
            }
        }
    }

    int x_init = rand() % (width - 1);
    int y_init = rand() % (height - 1);

    if (x_init % 2 == 0) {
        x_init++;
    }
    if (y_init % 2 == 0) {
        y_init++;
    }

    int adr = y_init*height + x_init;
    visited[adr] = true;
    cells[0] = x_init;
    cells[1] = y_init;

    //int cell = 32;
    //InitWindow(width*cell, height*cell, "PufferLib Ray Grid");
    //SetTargetFPS(60);

    while (num_cells > 0) {
        if (rand() % 1000 > 1000*difficulty) {
            int i = rand() % num_cells;
            int tmp_x = cells[2*num_cells - 2];
            int tmp_y = cells[2*num_cells - 1];
            cells[2*num_cells - 2] = cells[2*i];
            cells[2*num_cells - 1] = cells[2*i + 1];
            cells[2*i] = tmp_x;
            cells[2*i + 1] = tmp_y;
 
        }

        int x = cells[2*num_cells - 2];
        int y = cells[2*num_cells - 1];
 
        int nx, ny;

        // In-place direction shuffle
        for (int i = 0; i < 4; i++) {
            int ii = i + rand() % (4 - i);
            int tmp = dirs[i];
            dirs[i] = dirs[ii];
            dirs[ii] = tmp;
        }

        bool made_path = false;
        for (int dir_i = 0; dir_i < 4; dir_i++) {
            int dir = dirs[dir_i];
            nx = x + 2*dx[dir];
            ny = y + 2*dy[dir];
           
            if (nx <= 0 || nx >= width-1 || ny <= 0 || ny >= height-1) {
                continue;
            }

            int visit_adr = ny*width + nx;
            if (visited[visit_adr]) {
                continue;
            }

            visited[visit_adr] = true;
            cells[2*num_cells] = nx;
            cells[2*num_cells + 1] = ny;

            nx = x + dx[dir];
            ny = y + dy[dir];

            int adr = ny*max_size + nx;
            grid[adr] = EMPTY;
            num_cells++;

            made_path = true;

            /*
            if (IsKeyPressed(KEY_ESCAPE)) {
                exit(0);
            }
            BeginDrawing();
            ClearBackground((Color){6, 24, 24, 255});
            Color color = (Color){128, 128, 128, 255};
            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++){
                    int adr = r*max_size + c;
                    int tile = grid[adr];
                    if (tile == WALL) {
                        DrawRectangle(c*cell, r*cell, cell, cell, color);
                    }
               }
            }
            EndDrawing();
            */

            break;
        }
        if (!made_path) {
            num_cells--;
        }
    }
}

void create_maze_level(Grid* env, int width, int height, float difficulty, int seed) {
    env->width = width;
    env->height = height;
    generate_growing_tree_maze(env->grid, width, height, env->max_size, difficulty, seed);
    make_border(env);
    spawn_agent(env, 0, 1, 1);
    int goal_adr = grid_offset(env, env->height - 2, env->width - 2);
    env->grid[goal_adr] = GOAL;
}
