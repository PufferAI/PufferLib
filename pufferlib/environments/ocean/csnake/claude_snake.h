#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
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
    bool leave_corpse_on_death;
    float reward_food;
    float reward_corpse;
    float reward_death;
} CSnake;

/*
void step_all(CSnake** envs, int n) {
    for (int i = 0; i < n; i++) {
        step(envs[i]);
    }
}
*/

CSnake* init_csnake(char* grid, int* snake, char* observations, int* snake_lengths,
                    int* snake_ptr, int* snake_lifetimes, int* snake_colors,
                    unsigned int* actions, float* rewards, int num_snakes,
                    int width, int height, int max_snake_length, int food,
                    int vision, bool leave_corpse_on_death, float reward_food,
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
    env->leave_corpse_on_death = leave_corpse_on_death;
    env->reward_food = reward_food;
    env->reward_corpse = reward_corpse;
    env->reward_death = reward_death;
    return env;
}

void compute_observations(CSnake* env) {
    for (int i = 0; i < env->num_snakes; i++) {
        int head_ptr = env->snake_ptr[i];
        int head_r = env->snake[i * env->max_snake_length * 2 + head_ptr * 2];
        int head_c = env->snake[i * env->max_snake_length * 2 + head_ptr * 2 + 1];
        
        for (int r = 0; r < 2 * env->vision + 1; r++) {
            for (int c = 0; c < 2 * env->vision + 1; c++) {
                env->observations[i * (2 * env->vision + 1) * (2 * env->vision + 1) + r * (2 * env->vision + 1) + c] =
                    env->grid[(head_r - env->vision + r) * env->width + (head_c - env->vision + c)];
            }
        }
    }
}

void spawn_snake(CSnake* env, int snake_id) {
    int head_ptr, head_r, head_c;

    // Delete the snake from the grid
    while (env->snake_lengths[snake_id] > 0) {
        head_ptr = env->snake_ptr[snake_id];
        head_r = env->snake[snake_id * env->max_snake_length * 2 + head_ptr * 2];
        head_c = env->snake[snake_id * env->max_snake_length * 2 + head_ptr * 2 + 1];

        if (env->leave_corpse_on_death && env->snake_lengths[snake_id] % 2 == 0) {
            env->grid[head_r * env->width + head_c] = CORPSE;
        } else {
            env->grid[head_r * env->width + head_c] = EMPTY;
        }

        env->snake[snake_id * env->max_snake_length * 2 + head_ptr * 2] = -1;
        env->snake[snake_id * env->max_snake_length * 2 + head_ptr * 2 + 1] = -1;
        env->snake_lengths[snake_id]--;

        if (head_ptr == 0) {
            env->snake_ptr[snake_id] = env->max_snake_length - 1;
        } else {
            env->snake_ptr[snake_id]--;
        }
    }

    // Spawn a new snake
    int tile;
    do {
        head_r = rand() % (env->height - 1);
        head_c = rand() % (env->width - 1);
        tile = env->grid[head_r * env->width + head_c];
    } while (tile != EMPTY && tile != CORPSE);

    env->grid[head_r * env->width + head_c] = env->snake_colors[snake_id];
    env->snake[snake_id * env->max_snake_length * 2] = head_r;
    env->snake[snake_id * env->max_snake_length * 2 + 1] = head_c;
    env->snake_lengths[snake_id] = 1;
    env->snake_ptr[snake_id] = 0;
    env->snake_lifetimes[snake_id] = 0;
}

void spawn_food(CSnake* env) {
    int r, c, tile;
    do {
        r = rand() % (env->height - 1);
        c = rand() % (env->width - 1);
        tile = env->grid[r * env->width + c];
    } while (tile != EMPTY && tile != CORPSE);
    
    env->grid[r * env->width + c] = FOOD;
}

void reset(CSnake* env) {
    // Set walls
    for (int r = 0; r < env->vision; r++) {
        for (int c = 0; c < env->width; c++) {
            env->grid[r * env->width + c] = WALL;
        }
    }
    for (int r = env->height - env->vision; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            env->grid[r * env->width + c] = WALL;
        }
    }
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->vision; c++) {
            env->grid[r * env->width + c] = WALL;
        }
        for (int c = env->width - env->vision; c < env->width; c++) {
            env->grid[r * env->width + c] = WALL;
        }
    }

    // Spawn snakes and food
    for (int i = 0; i < env->num_snakes; i++) {
        spawn_snake(env, i);
    }
    for (int i = 0; i < env->food; i++) {
        spawn_food(env);
    }

    compute_observations(env);
}

void step(CSnake* env) {
    int atn, dr, dc, head_ptr, head_r, head_c, next_r, next_c, tile, tail_ptr, tail_r, tail_c, snake_length;
    float reward;
    bool grow;

    for (int i = 0; i < env->num_snakes; i++) {
        atn = env->actions[i];
        dr = 0;
        dc = 0;
        
        switch (atn) {
            case 0: dr = -1; break; // up
            case 1: dr = 1; break;  // down
            case 2: dc = -1; break; // left
            case 3: dc = 1; break;  // right
        }

        snake_length = env->snake_lengths[i];
        head_ptr = env->snake_ptr[i];
        head_r = env->snake[i * env->max_snake_length * 2 + head_ptr * 2];
        head_c = env->snake[i * env->max_snake_length * 2 + head_ptr * 2 + 1];
        next_r = head_r + dr;
        next_c = head_c + dc;

        tile = env->grid[next_r * env->width + next_c];
        if (tile >= WALL) {
            env->rewards[i] = env->reward_death;
            spawn_snake(env, i);
            continue;
        }

        head_ptr++;
        if (head_ptr >= env->max_snake_length) {
            head_ptr = 0;
        }

        env->snake[i * env->max_snake_length * 2 + head_ptr * 2] = next_r;
        env->snake[i * env->max_snake_length * 2 + head_ptr * 2 + 1] = next_c;
        env->snake_ptr[i] = head_ptr;
        env->snake_lifetimes[i]++;

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

        if (grow && snake_length < env->max_snake_length - 1) {
            env->snake_lengths[i]++;
        } else {
            tail_ptr = head_ptr - snake_length;
            if (tail_ptr < 0) {
                tail_ptr = env->max_snake_length + tail_ptr;
            }

            tail_r = env->snake[i * env->max_snake_length * 2 + tail_ptr * 2];
            tail_c = env->snake[i * env->max_snake_length * 2 + tail_ptr * 2 + 1];
            env->snake[i * env->max_snake_length * 2 + tail_ptr * 2] = -1;
            env->snake[i * env->max_snake_length * 2 + tail_ptr * 2 + 1] = -1;
            env->grid[tail_r * env->width + tail_c] = EMPTY;
        }

        env->grid[next_r * env->width + next_c] = env->snake_colors[i];

    }

    compute_observations(env);
}

CSnake* allocate_csnake(int num_snakes, int width, int height, int max_snake_length,
                        int food, int vision, bool leave_corpse_on_death,
                        float reward_food, float reward_corpse, float reward_death) {
    CSnake* env = (CSnake*)malloc(sizeof(CSnake));
    
    env->grid = (char*)calloc(width * height, sizeof(char));
    env->observations = (char*)calloc(num_snakes * (2 * vision + 1) * (2 * vision + 1), sizeof(char));
    env->snake = (int*)calloc(num_snakes * max_snake_length * 2, sizeof(int));
    env->snake_lengths = (int*)calloc(num_snakes, sizeof(int));
    env->snake_ptr = (int*)calloc(num_snakes, sizeof(int));
    env->snake_lifetimes = (int*)calloc(num_snakes, sizeof(int));

    env->snake_colors = (int*)calloc(num_snakes, sizeof(int));
    env->actions = (unsigned int*)calloc(num_snakes, sizeof(unsigned int));
    env->rewards = (float*)calloc(num_snakes, sizeof(float));
    
    env->num_snakes = num_snakes;
    env->width = width;
    env->height = height;
    env->max_snake_length = max_snake_length;
    env->food = food;
    env->vision = vision;
    env->leave_corpse_on_death = leave_corpse_on_death;
    env->reward_food = reward_food;
    env->reward_corpse = reward_corpse;
    env->reward_death = reward_death;
    
    return env;
}

void free_csnake(CSnake* env) {
    if (env != NULL) {
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
    
    InitWindow(width * cell_size, height * cell_size, "Snake Game");
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

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int cell = env->grid[y * env->width + x];
            Color color = COLORS[cell];
            DrawRectangle(x * renderer->cell_size, y * renderer->cell_size,
                          renderer->cell_size, renderer->cell_size, color);
        }
    }

    EndDrawing();
}
