#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "raylib.h"

#include "grid.h"

#define EMPTY 0
#define NORMAL_FOOD 1
#define INTERACTIVE_FOOD 2
// Anything above Wall should be obstacles
#define WALL 3
#define AGENTS 4

#define LOG_BUFFER_SIZE 8192

#define SET_BIT(arr, i) (arr[(i) / 8] |= (1 << ((i) % 8)))
#define CLEAR_BIT(arr, i) (arr[(i) / 8] &= ~(1 << ((i) % 8)))
#define CHECK_BIT(arr, i) (arr[(i) / 8] & (1 << ((i) % 8)))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define REWARD_20_HP -0
#define REWARD_80_HP 0
#define REWARD_DEATH -1.0f


#define LOG_SCORE_REWARD_SMALL 1 
#define LOG_SCORE_REWARD_MEDIUM 5
#define LOG_SCORE_REWARD_MOVE - 0.0
#define LOG_SCORE_REWARD_DEATH -1

#define HP_REWARD_FOOD_MEDIUM 50
#define HP_REWARD_FOOD_SMALL 20
#define HP_LOSS_PER_STEP 1
#define MAX_HP 100 

typedef struct Log Log;
struct Log {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float moves;
  float food_nb;
  float agents_alive;
  float alive_steps;
  float n;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
  Log *logs;
  int length;
  int idx;
};

LogBuffer *allocate_logbuffer(int size) {
  LogBuffer *logs = (LogBuffer *)calloc(1, sizeof(LogBuffer));
  logs->logs = (Log *)calloc(size, sizeof(Log));
  logs->length = size;
  logs->idx = 0;
  return logs;
}

void free_logbuffer(LogBuffer *buffer) {
  free(buffer->logs);
  free(buffer);
}

void add_log(LogBuffer *logs, Log *log) {
  if (logs->idx == logs->length) {
    return;
  }
  log->episode_return = log->score;
  log->perf = fmaxf(0, 1.0 - 0.01*log->episode_length);
  logs->logs[logs->idx] = *log;
  logs->idx += 1;
}

Log aggregate_and_clear(LogBuffer *logs) {
  Log log = {0};
  if (logs->idx == 0) {
    return log;
  }
  for (int i = 0; i < logs->idx; i++) {
    log.score += logs->logs[i].score;
    log.perf += logs->logs[i].perf;
    log.episode_return += logs->logs[i].episode_return;
    log.episode_length += logs->logs[i].episode_length;
    log.moves += logs->logs[i].moves;
    log.food_nb += logs->logs[i].food_nb;
    log.agents_alive += logs->logs[i].agents_alive;
    log.alive_steps += logs->logs[i].alive_steps;
    log.n += 1;
  }
  log.score /= logs->idx;
  log.perf /= logs->idx;
  log.episode_return /= logs->idx;
  log.episode_length /= logs->idx;
  log.moves /= logs->idx;
  log.food_nb /= logs->idx;
  log.agents_alive /= logs->idx;
  log.alive_steps /= logs->idx;
  log.n /= logs->idx;
  logs->idx = 0;
  return log;
}

typedef struct Agent Agent;
struct Agent {
  int r;
  int c;
  int id;
  float hp;
  int direction;
};

typedef struct FoodList FoodList;
struct FoodList {
  int *indexes; // Grid flattened index positions
  int size;
};

FoodList *allocate_foodlist(int size) {
  FoodList *foods = (FoodList *)calloc(1, sizeof(FoodList));
  foods->indexes = (int *)calloc(size, sizeof(int));
  foods->size = 0;
  return foods;
}

void free_foodlist(FoodList *foods) {
  free(foods->indexes);
  free(foods);
}

typedef struct CCpr CCpr;
struct CCpr {
  int width;
  int height;
  int num_agents;

  int vision;
  int vision_window;
  int obs_size;

  int tick;

  float reward_food;
  float reward_move;
  float interactive_food_reward;

  unsigned char *grid;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  unsigned char *truncations;
  unsigned char *masks;

  Agent *agents;

  LogBuffer *log_buffer;
  Log *log;

  uint8_t *interactive_food_agent_count;

  FoodList *foods;
  float food_base_spawn_rate;
};

void init_ccpr(CCpr *env) {
  env->grid =
      (unsigned char *)calloc(env->width * env->height, sizeof(unsigned char));
  env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
  env->vision_window = 2 * env->vision + 1;
  env->obs_size = env->vision_window * env->vision_window + 1;
  env->log = (Log *)calloc(1, sizeof(Log));
  env->interactive_food_agent_count =
      (uint8_t *)calloc((env->width * env->height + 7) / 8, sizeof(uint8_t));
  env->foods = allocate_foodlist(env->width * env->height);
}

void allocate_ccpr(CCpr *env) {
  // Called by C stuff
  int obs_size = (2 * env->vision + 1) * (2 * env->vision + 1) + 1;
  env->observations = (unsigned char *)calloc(env->num_agents * obs_size,
                                              sizeof(unsigned char));
  env->actions = (int *)calloc(env->num_agents, sizeof(unsigned int));
  env->rewards = (float *)calloc(env->num_agents, sizeof(float));
  env->terminals =
      (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
  env->truncations = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
  env->masks = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
  env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
  init_ccpr(env);
}

void free_CCpr(CCpr *env) {
  if (env == NULL) {
    return;
  }

  free(env->grid);
  free(env->agents);
  free(env->observations);
  free(env->actions);
  free(env->rewards);
  free(env->terminals);
  free(env->truncations);
  free(env->masks);
  free_logbuffer(env->log_buffer);
  free(env->log);
  free(env->interactive_food_agent_count);
  free_foodlist(env->foods);
}

int grid_index(CCpr *env, int r, int c) { return r * env->width + c; }
int get_agent_tile_from_id(int agent_id) { return AGENTS + agent_id; }

int get_agent_id_from_tile(int tile) { return tile - AGENTS; }

void add_food(CCpr *env, int grid_idx, int food_type) {
  // Add food to the grid and the food_list at grid_idx
  assert(env->grid[grid_idx] == EMPTY);
  env->grid[grid_idx] = food_type;
  FoodList *foods = env->foods;
  foods->indexes[foods->size++] = grid_idx;
}

void reward_agent(CCpr *env, int agent_id, float reward) {
  // We don't reward if agent is full life
  Agent *agent = &env->agents[agent_id];
  if (agent->hp >= MAX_HP) {
    return;
  }
  env->rewards[agent_id] += reward;
}

void spawn_food(CCpr *env, int food_type) {
  // Randomly spawns such food in the grid
  int idx, tile;
  do {
    int r = rand() % (env->height - 1);
    int c = rand() % (env->width - 1);
    idx = r * env->width + c;
    tile = env->grid[idx];
  } while (tile != EMPTY);
  add_food(env, idx, food_type);
}

void remove_food(CCpr *env, int grid_idx) {
  // Removes food from the grid and food_list
  env->grid[grid_idx] = EMPTY;
  FoodList *foods = env->foods;
  for (int i = 0; i < foods->size; i++) {
    if (foods->indexes[i] == grid_idx) {
      foods->indexes[i] = foods->indexes[foods->size - 1];
      foods->size--;
      return;
    }
  }
}

void init_foods(CCpr *env) {
  // On reset spawns x number of each food randomly.
  int available_tiles = (env->width * env->height) -
                        (2 * env->vision * env->width +
                         2 * env->vision * (env->height - 2 * env->vision));
  int normalizer = (env->width * env->height) / 576;
  int normal = available_tiles / (20 * normalizer);
  int interactive = available_tiles / (50 * normalizer);
  for (int i = 0; i < normal; i++) {
    spawn_food(env, NORMAL_FOOD);
  }
  for (int i = 0; i < interactive; i++) {
    spawn_food(env, INTERACTIVE_FOOD);
  }
}

void spawn_foods(CCpr *env) {
  // After each step, check existing foods and spawns new food in the
  // neighborhood Iterates over food_list for efficiency instead of the entire
  // grid.
  FoodList *foods = env->foods;
  int original_size = foods->size;
  for (int i = 0; i < original_size; i++) {
    int idx = foods->indexes[i];
    int offset = idx - env->width - 1; // Food spawn in 1 radius
    int r = offset / env->width;
    int c = offset % env->width;
    for (int ri = 0; ri < 3; ri++) {
      for (int ci = 0; ci < 3; ci++) {
        int grid_idx = grid_index(env, (r + ri), (c + ci));
        if (env->grid[grid_idx] != EMPTY) {
          continue;
        }
        switch (env->grid[idx]) {
        // %Chance spawning new food
        case NORMAL_FOOD:
          if ((rand() / (double)RAND_MAX) < env->food_base_spawn_rate) {
            add_food(env, grid_idx, env->grid[idx]);
          }
          break;
        case INTERACTIVE_FOOD:
          if ((rand() / (double)RAND_MAX) <
              (env->food_base_spawn_rate / 10.0)) {
            add_food(env, grid_idx, env->grid[idx]);
          }
          break;
        }
      }
    }
  }

  // // Each turn there is random probability for a food to spawn at a random
  // // location To cope with resource depletion
  // int normalizer = (env->width * env->height) / 576;
  // if ((rand() / (double)RAND_MAX) <
  //     min((env->food_base_spawn_rate * 2 * normalizer), 1e-2)) {
  //   spawn_food(env, NORMAL_FOOD);
  // }
  // if ((rand() / (double)RAND_MAX) <
  //     min((env->food_base_spawn_rate / 5.0 * normalizer), 5e-3)) {
  //   spawn_food(env, INTERACTIVE_FOOD);
  // }
}

void compute_observations(CCpr *env) {
  // For full obs
  // memcpy(env->observations, env->grid,
  //        env->width * env->height * sizeof(unsigned char));
  // return;

  // For partial obs
  for (int i = 0; i < env->num_agents; i++) {
    Agent *agent = &env->agents[i];
    env->observations[env->vision_window*env->vision_window + i*env->obs_size] = agent->hp;
    if (agent->hp == 0) {
      continue;
    }
    int obs_offset = i * env->obs_size;
    int r_offset = agent->r - env->vision;
    int c_offset = agent->c - env->vision;
    for (int r = 0; r < 2 * env->vision + 1; r++) {
      for (int c = 0; c < 2 * env->vision + 1; c++) {
        int grid_idx = (r_offset + r) * env->width + c_offset + c;
        int obs_idx = obs_offset + r * env->vision_window + c;
        env->observations[obs_idx] = env->grid[grid_idx];
      }
    }

    
  }
}

void add_hp(CCpr *env, int agent_id, float hp) {
  Agent *agent = &env->agents[agent_id];
  agent->hp += hp;
  if (agent->hp > MAX_HP) {
    agent->hp = MAX_HP;
  } else if (agent->hp <= 0) {
    agent->hp = 0;
    env->log->score += LOG_SCORE_REWARD_DEATH;
    env->rewards[agent->id] += REWARD_DEATH;
    env->terminals[agent->id] = 1;
  }
}
void remove_hp(CCpr *env, int agent_id, float hp) { add_hp(env, agent_id, -hp); }

void save_grid_to_file(CCpr *env, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    fprintf(file, "#ifndef GRID_H\n#define GRID_H\n\n");
    fprintf(file, "#define GRID_HEIGHT %d\n", env->height);
    fprintf(file, "#define GRID_WIDTH %d\n\n", env->width);
    fprintf(file, "static const unsigned char grid[GRID_HEIGHT][GRID_WIDTH] = {\n");

    for (int r = 0; r < env->height; r++) {
        fprintf(file, "    {");
        for (int c = 0; c < env->width; c++) {
            unsigned char val = env->grid[r * env->width + c];
            fprintf(file, "0x%02X%s", val, (c == env->width - 1) ? "" : ", ");
        }
        fprintf(file, "}%s\n", (r == env->height - 1) ? "" : ",");
    }
    fprintf(file, "};\n\n#endif // GRID_H\n");
    fclose(file);
}

void make_grid_from_scratch(CCpr *env){
  memset(env->grid, EMPTY, (env->height * env->width) * sizeof(env->grid[0]));
  // top walling
  for (int r = 0; r < env->vision; r++) {
    memset(env->grid + (r * env->width), WALL,
           env->width * sizeof(env->grid[0]));
  }
  // left side walling
  for (int r = 0; r < env->height; r++) {
    memset(env->grid + (r * env->width), WALL,
           env->vision * sizeof(env->grid[0]));
  }
  // bottom walling
  for (int r = env->height - env->vision; r < env->height; r++) {
    memset(env->grid + (r * env->width), WALL,
           env->width * sizeof(env->grid[0]));
  }

  // right side walling
  for (int r = 0; r < env->height; r++) {
    memset(env->grid + (r * env->width) + (env->width - env->vision), WALL,
           env->vision * sizeof(env->grid[0]));
  }
  save_grid_to_file(env, "grid.h");
}

void spawn_agent(CCpr *env, int i){
  Agent *agent = &env->agents[i];
  agent->id = i;
  agent->hp = 80;
  int adr = 0;

  bool allocated = false;
  while (!allocated) {
    adr = rand() % (env->height * env->width);
    if (env->grid[adr] == EMPTY) {
      int r = adr / env->width;
      int c = adr % env->width;
      agent->r = r;
      agent->c = c;
      allocated = true;
    }
  }
  assert(env->grid[adr] == EMPTY);
  env->grid[adr] = get_agent_tile_from_id(agent->id);
}
void c_reset(CCpr *env) {
  env->tick = 0;
  memset(env->log, 0, sizeof(Log));
  env->foods->size = 0;
  memset(env->foods->indexes, 0, env->width * env->height * sizeof(int));
  // make_grid_from_scratch(env);
  memcpy(env->grid, grid_32_32_3v, env->width * env->height * sizeof(unsigned char));

  for (int i = 0; i < env->num_agents; i++) {
    spawn_agent(env, i);
  }

  init_foods(env);
  memset(env->observations, 0, env->num_agents * env->obs_size * sizeof(unsigned char));
  memset(env->truncations, 0, env->num_agents * sizeof(unsigned char));
  memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
  memset(env->masks, 1, env->num_agents * sizeof(unsigned char));
  compute_observations(env);
}

void reward_agents_near(CCpr *env, int food_index) {
  int food_r = food_index / env->width;
  int food_c = food_index % env->width;

  // TODO: could iterate over neighbors of food index and check if is agent
  // (remove iteration cost)
  for (int i = 0; i < env->num_agents; i++) {
    int ac = env->agents[i].c;
    int ar = env->agents[i].r;

    if ((ac == food_c && (ar == food_r - 1 || ar == food_r + 1)) ||
        (ar == food_r && (ac == food_c - 1 || ac == food_c + 1))) {
      reward_agent(env, i, env->interactive_food_reward);
      env->log->score += LOG_SCORE_REWARD_MEDIUM;
      add_hp(env, i, HP_REWARD_FOOD_MEDIUM);
      // add_log(env->log_buffer, env->log);
      // memset(env->log, 0, sizeof(Log));
    }
  }
  remove_food(env, food_index);
}

void step_agent(CCpr *env, int i) {

  Agent *agent = &env->agents[i];

  int action = env->actions[i];
  env->log->episode_length += 1;

  int dr = 0;
  int dc = 0;

  switch (action) {
  case 0:
    dr = -1;
    agent->direction = 3;
    break; // UP
  case 1:
    dr = 1;
    agent->direction = 1;
    break; // DOWN
  case 2:
    dc = -1;
    agent->direction = 2;
    break; // LEFT
  case 3:
    dc = 1;
    agent->direction = 0;
    break; // RIGHT
  case 4:
    return; // No moves
  }
  env->log->moves += 1;

  // Get next row and column

  int next_r = agent->r + dr;
  int next_c = agent->c + dc;

  int prev_grid_idx = grid_index(env, agent->r, agent->c);
  int next_grid_idx = env->width * next_r + next_c;
  int tile = env->grid[next_grid_idx];

  // Anything above should be obstacle
  if (tile >= INTERACTIVE_FOOD) {
    env->log->score += LOG_SCORE_REWARD_MOVE;
    env->rewards[i] += env->reward_move;
    next_r = agent->r;
    next_c = agent->c;
    next_grid_idx = env->width * next_r + next_c;
    tile = env->grid[next_grid_idx];
  }

  // Interactive food logic
  int neighboors[4] = {
      grid_index(env, next_r - 1, next_c), // Up
      grid_index(env, next_r + 1, next_c), // Down
      grid_index(env, next_r, next_c + 1), // Right
      grid_index(env, next_r, next_c - 1)  // Left
  };

  for (int j = 0; j < 4; j++) {
    int grid_idx = neighboors[j];
    // If neighbooring grid tile is interactive food
    if (env->grid[grid_idx] == INTERACTIVE_FOOD) {
      // If was already marked as "ready to collect"
      if (CHECK_BIT(env->interactive_food_agent_count, grid_idx)) {
        reward_agents_near(env, grid_idx);
      } else {
        // First agent detected
        SET_BIT(env->interactive_food_agent_count, grid_idx);
      }
    }
  }

  switch (tile) {
  case NORMAL_FOOD:
    env->log->score += LOG_SCORE_REWARD_SMALL;
    reward_agent(env, i, env->reward_food);
    add_hp(env, i, HP_REWARD_FOOD_SMALL);
    remove_food(env, next_grid_idx);
    // add_log(env->log_buffer, env->log);
    // memset(env->log, 0, sizeof(Log));
    break;
  case EMPTY:
    env->log->score += LOG_SCORE_REWARD_MOVE;
    env->rewards[i] += env->reward_move;
    break;
  }

  int agent_tile = get_agent_tile_from_id(agent->id);

  env->grid[prev_grid_idx] = EMPTY;
  env->grid[next_grid_idx] = agent_tile;
  agent->r = next_r;
  agent->c = next_c;

  return;
}

void clear_agent(CCpr *env, int agent_id) {
  Agent *agent = &env->agents[agent_id];
  if (agent->r < 0 || agent->c < 0) {
    return;
  }
  int grid_idx = grid_index(env, agent->r, agent->c);
  env->grid[grid_idx] = EMPTY;
  agent->r = -1;
  agent->c = -1;
}

void c_step(CCpr *env) {
  env->tick++;

  // memset(env->truncations, 0, env->num_agents * sizeof(unsigned char));
  memset(env->rewards, 0, env->num_agents * sizeof(float));
  // memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
  memset(env->interactive_food_agent_count, 0,
         (env->width * env->height + 7) / 8);

  for (int i = 0; i < env->num_agents; i++) {
    if (env->agents[i].hp == 0) {
      env->masks[i] = 0;
      clear_agent(env, i);
      continue;
    }
    step_agent(env, i);
    remove_hp(env, i, HP_LOSS_PER_STEP);
  }

  spawn_foods(env);

  //We loop again here because in the future an entity might have attacked an agent in the process
  int alive_agents = 0;
  for (int i = 0; i < env->num_agents; i++) {
    if (env->agents[i].hp > 0) {
      alive_agents += 1;
      if (env->agents[i].hp < 20) {
        env->rewards[i] += REWARD_20_HP;
        env->log->score += REWARD_20_HP;
      } else if (env->agents[i].hp > 80) {
        env->rewards[i] += REWARD_80_HP;
        env->log->score += REWARD_80_HP;
      }
    } 
    // else {
      // int grid_idx = grid_index(env, env->agents[i].r, env->agents[i].c);
      // env->grid[grid_idx] = EMPTY;
      // spawn_agent(env, i);
    // }
  }
  if (alive_agents == 0) {
    env->log->moves = 0;
  }else{
    env->log->moves /= alive_agents;
  }
  env->log->food_nb = env->foods->size;
  env->log->agents_alive = alive_agents;
  env->log->alive_steps = env->tick;
  compute_observations(env);
  
  if (alive_agents == 0|| env->tick > 1000) {
    c_reset(env);
    if (alive_agents == 0) {
      memset(env->terminals, 1, env->num_agents * sizeof(unsigned char)); 
    } else {
      //Agents get rewarded for going all the way to the end
      for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].hp > 0) {
          env->rewards[i] += 1;
        }
      }
      memset(env->truncations, 1, env->num_agents * sizeof(unsigned char));
    }
  }
  add_log(env->log_buffer, env->log);
  memset(env->log, 0, sizeof(Log));
}

// Raylib client
Color COLORS[] = {
    (Color){255, 0, 0, 255},     (Color){170, 170, 170, 255},
    (Color){255, 255, 0, 255},   (Color){0, 255, 0, 255},
    (Color){0, 255, 255, 255},   (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255}, (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255}, (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255}, (Color){0, 255, 255, 255},
    (Color){0, 0, 255, 255},     (Color){6, 24, 24, 255},
};

Rectangle UV_COORDS[7] = {
    (Rectangle){0, 0, 0, 0},       (Rectangle){512, 0, 128, 128},
    (Rectangle){0, 0, 0, 0},       (Rectangle){0, 0, 128, 128},
    (Rectangle){128, 0, 128, 128}, (Rectangle){256, 0, 128, 128},
    (Rectangle){384, 0, 128, 128},
};

typedef struct Renderer Renderer;
struct Renderer {
  int cell_size;
  int width;
  int height;
  Texture2D puffer;
};

Renderer *init_renderer(int cell_size, int width, int height) {
  Renderer *renderer = (Renderer *)calloc(1, sizeof(Renderer));
  renderer->cell_size = cell_size;
  renderer->width = width;
  renderer->height = height;

  InitWindow(width * cell_size, height * cell_size, "CPR");
  SetTargetFPS(10);

  renderer->puffer = LoadTexture("resources/common_pool_resource/puffers_128.png");
  return renderer;
}

void close_renderer(Renderer *renderer) {
  CloseWindow();
  free(renderer);
}

void c_render(Renderer *renderer, CCpr *env) {
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  int ts = renderer->cell_size;
  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int adr = grid_index(env, r, c);
      int tile = env->grid[adr];
      if (tile == EMPTY) {
        continue;
      } else if (tile == WALL) {
        DrawRectangle(c * ts, r * ts, ts, ts, (Color){227, 227, 227, 255});
      } else if (tile == NORMAL_FOOD || tile == INTERACTIVE_FOOD) {
        DrawRectangle(c * ts, r * ts, ts, ts, COLORS[tile]);
      } else {

        int agent_id = get_agent_id_from_tile(tile);
        int col_id = agent_id % (sizeof(COLORS) / sizeof(COLORS[0]));
        Color color = COLORS[col_id];
        int starting_sprite_x = 0;
        float rotation = env->agents[agent_id].direction * 90.0f;
        if (rotation == 180) {
          starting_sprite_x = 128;
          rotation = 0;
        }
        Rectangle source_rect = (Rectangle){starting_sprite_x, 0, 128, 128};
        Rectangle dest_rect = (Rectangle){c * ts + ts/2, r * ts + ts/2, ts, ts};        
        DrawTexturePro(renderer->puffer, source_rect, dest_rect,
                       (Vector2){ts/2, ts/2}, rotation, color);
      }
    }
  }
  EndDrawing();
}
