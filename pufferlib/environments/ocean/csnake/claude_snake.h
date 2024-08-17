// Port of PufferLib's Cython multi-snake env to C. There is no performance
// benefit to this implementation. It is purely so we can compile to WASM
#include <stdlib.h>
#include <stdbool.h>
#include "raylib.h"

#define EMPTY 0
#define FOOD 1
#define CORPSE 2
#define WALL 3

Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
};

typedef struct {
    char* grid;
    char* observations;
    int* snake;
    int* snake_lengths;
    int* snake_ptr;
    int* snake_lifetimes;
    int* snake_colors;
    unsigned int* actions;
    float* rewards;
    int num_snakes;
    int width;
    int height;
    int max_snake_length;
    int food;
    int vision;
    int window;
    int obs_size;
    bool leave_corpse_on_death;
    float reward_food;
    float reward_corpse;
    float reward_death;
} CSnake;

CSnake* init_csnake(char* grid, int* snake, char* observations, int* snake_lengths,
        int* snake_ptr, int* snake_lifetimes, int* snake_colors, unsigned int* actions,
        float* rewards, int num_snakes, int width, int height, int max_snake_length,
        int food, int vision, bool leave_corpse_on_death, float reward_food,
        float reward_corpse, float reward_death) {
    CSnake* env = (CSnake*)malloc(sizeof(CSnake));
    env->grid = grid;
    env->observations = observations;
    env->snake = snake;
    env->snake_lengths = snake_lengths;
    env->snake_ptr = snake_ptr;
    env->snake_lifetimes = snake_lifetimes;
    env->snake_colors = snake_colors;
    env->actions = actions;
    env->rewards = rewards;
    env->num_snakes = num_snakes;
    env->width = width;
    env->height = height;
    env->max_snake_length = max_snake_length;
    env->food = food;
    env->vision = vision;
    env->window = 2*vision+1;
    env->obs_size = env->window*env->window;
    env->leave_corpse_on_death = leave_corpse_on_death;
    env->reward_food = reward_food;
    env->reward_corpse = reward_corpse;
    env->reward_death = reward_death;
    return env;
}

CSnake* allocate_csnake(int num_snakes, int width, int height,
        int max_snake_length, int food, int vision, bool leave_corpse_on_death,
        float reward_food, float reward_corpse, float reward_death) {
    int obs_size = (2*vision+1) * (2*vision+1);
    char* grid = (char*)calloc(width*height, sizeof(char));
    char* observations = (char*)calloc(num_snakes*obs_size, sizeof(char));
    int* snake = (int*)calloc(num_snakes*2*max_snake_length, sizeof(int));
    int* snake_lengths = (int*)calloc(num_snakes, sizeof(int));
    int* snake_ptr = (int*)calloc(num_snakes, sizeof(int));
    int* snake_lifetimes = (int*)calloc(num_snakes, sizeof(int));
    int* snake_colors = (int*)calloc(num_snakes, sizeof(int));
    unsigned int* actions = (unsigned int*)calloc(num_snakes, sizeof(unsigned int));
    float* rewards = (float*)calloc(num_snakes, sizeof(float));
    return init_csnake(grid, snake, observations, snake_lengths, snake_ptr, snake_lifetimes,
        snake_colors, actions, rewards, num_snakes, width, height, max_snake_length,
        food, vision, leave_corpse_on_death, reward_food, reward_corpse, reward_death);
    }

void free_csnake(CSnake* env) {
    if (env == NULL)
        return;

    free(env->grid);
    free(env->observations);
    free(env->snake);
    free(env->snake_lengths);
    free(env->snake_ptr);
    free(env->snake_lifetimes);
    free(env->snake_colors);
    free(env->actions);
    free(env->rewards);
    free(env);
}

void compute_observations(CSnake* env) {
    for (int i = 0; i < env->num_snakes; i++) {
        int head_ptr = i*2*env->max_snake_length + 2*env->snake_ptr[i];
        int r_offset = env->snake[head_ptr] - env->vision;
        int c_offset = env->snake[head_ptr+1] - env->vision;
        for (int r = 0; r < 2 * env->vision + 1; r++) {
            for (int c = 0; c < 2 * env->vision + 1; c++) {
                env->observations[i*env->obs_size + r*env->window + c] = env->grid[
                    (r_offset + r)*env->width + c_offset + c];
            }
        }
    }
}

void delete_snake(CSnake* env, int snake_id) {
    while (env->snake_lengths[snake_id] > 0) {
        int head_ptr = env->snake_ptr[snake_id];
        int head_offset = 2*env->max_snake_length*snake_id + 2*head_ptr;
        int head_r = env->snake[head_offset];
        int head_c = env->snake[head_offset + 1];
        if (env->leave_corpse_on_death && env->snake_lengths[snake_id] % 2 == 0)
            env->grid[head_r*env->width + head_c] = CORPSE;
        else
            env->grid[head_r*env->width + head_c] = EMPTY;

        env->snake[head_offset] = -1;
        env->snake[head_offset + 1] = -1;
        env->snake_lengths[snake_id]--;
        if (head_ptr == 0)
            env->snake_ptr[snake_id] = env->max_snake_length - 1;
        else
            env->snake_ptr[snake_id]--;
    }
}

void spawn_snake(CSnake* env, int snake_id) {
    int head_r, head_c, tile, grid_idx;
    delete_snake(env, snake_id);
    do {
        head_r = rand() % (env->height - 1);
        head_c = rand() % (env->width - 1);
        grid_idx = head_r*env->width + head_c;
        tile = env->grid[grid_idx];
    } while (tile != EMPTY && tile != CORPSE);
    int snake_offset = 2*env->max_snake_length*snake_id;
    env->snake[snake_offset] = head_r;
    env->snake[snake_offset + 1] = head_c;
    env->snake_lengths[snake_id] = 1;
    env->snake_ptr[snake_id] = 0;
    env->snake_lifetimes[snake_id] = 0;
    env->grid[grid_idx] = env->snake_colors[snake_id];
}

void spawn_food(CSnake* env) {
    int idx, tile;
    do {
        int r = rand() % (env->height - 1);
        int c = rand() % (env->width - 1);
        idx = r*env->width + c;
        tile = env->grid[idx];
    } while (tile != EMPTY && tile != CORPSE);
    env->grid[idx] = FOOD;
}

void reset(CSnake* env) {
    for (int r = 0; r < env->vision; r++) {
        for (int c = 0; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int r = env->height - env->vision; r < env->height; r++) {
        for (int c = 0; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->vision; c++)
            env->grid[r*env->width + c] = WALL;
        for (int c = env->width - env->vision; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int i = 0; i < env->num_snakes; i++)
        spawn_snake(env, i);
    for (int i = 0; i < env->food; i++)
        spawn_food(env);

    compute_observations(env);
}

void step_snake(CSnake* env, int i) {
    int atn = env->actions[i];
    int dr = 0;
    int dc = 0;
    switch (atn) {
        case 0: dr = -1; break; // up
        case 1: dr = 1; break;  // down
        case 2: dc = -1; break; // left
        case 3: dc = 1; break;  // right
    }

    int head_ptr = env->snake_ptr[i];
    int snake_offset = 2*env->max_snake_length*i;
    int head_offset = snake_offset + 2*head_ptr;
    int next_r = dr + env->snake[head_offset];
    int next_c = dc + env->snake[head_offset + 1];
    int tile = env->grid[next_r*env->width + next_c];
    if (tile >= WALL) {
        env->rewards[i] = env->reward_death;
        spawn_snake(env, i);
        return;
    }

    head_ptr++; // Circular buffer
    if (head_ptr >= env->max_snake_length)
        head_ptr = 0;

    head_offset = snake_offset + 2*head_ptr;
    env->snake[head_offset] = next_r;
    env->snake[head_offset + 1] = next_c;
    env->snake_ptr[i] = head_ptr;
    env->snake_lifetimes[i]++;

    bool grow;
    if (tile == FOOD) {
        env->rewards[i] = env->reward_food;
        spawn_food(env);
        grow = true;
    } else if (tile == CORPSE) {
        env->rewards[i] = env->reward_corpse;
        grow = true;
    } else {
        env->rewards[i] = 0.0;
        grow = false;
    }

    int snake_length = env->snake_lengths[i];
    if (grow && snake_length < env->max_snake_length - 1) {
        env->snake_lengths[i]++;
    } else {
        int tail_ptr = head_ptr - snake_length;
        if (tail_ptr < 0) // Circular buffer
            tail_ptr = env->max_snake_length + tail_ptr;

        int tail_r = env->snake[snake_offset + 2*tail_ptr];
        int tail_c = env->snake[snake_offset + 2*tail_ptr + 1];
        int tail_offset = 2*env->max_snake_length*i + 2*tail_ptr;
        env->snake[tail_offset] = -1;
        env->snake[tail_offset + 1] = -1;
        env->grid[tail_r*env->width + tail_c] = EMPTY;
    }
    env->grid[next_r*env->width + next_c] = env->snake_colors[i];
}

void step(CSnake* env){
    for (int i = 0; i < env->num_snakes; i++)
        step_snake(env, i);

    compute_observations(env);
}

typedef struct {
    int cell_size;
    int width;
    int height;
} Renderer;

Renderer* init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)malloc(sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;
    InitWindow(width*cell_size, height*cell_size, "Snake Game");
    SetTargetFPS(10);
    return renderer;
}

void close_renderer(Renderer* renderer) {
    CloseWindow();
    free(renderer);
}

void render(Renderer* renderer, CSnake* env) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    int sz = renderer->cell_size;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++)
            DrawRectangle(x*sz, y*sz, sz, sz, COLORS[env->grid[y*env->width + x]]);
    }
    EndDrawing();
}
