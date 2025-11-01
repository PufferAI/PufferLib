#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "raylib.h"

#include "terrain.h"

#define MAX_TIMESTEPS 1000 // If no agent died by then, we reset

#define EMPTY 0

// Tiles
#define TILE_DIRT 0
#define TILE_WATER 4
#define TILE_GRASS 8
#define TILE_HOUSE 3

// Items
#define ITEM_FOOD 10
#define ITEM_WOOD 13
#define ITEM_BED 8

// Entities
#define ENTITY_AGENT 0

#define MAX_CELL_OBS 3 // Maximum number of info per cell in observations
#define LOG_BUFFER_SIZE 8192

#define SET_BIT(arr, i) (arr[(i) / 8] |= (1 << ((i) % 8)))
#define CLEAR_BIT(arr, i) (arr[(i) / 8] &= ~(1 << ((i) % 8)))
#define CHECK_BIT(arr, i) (arr[(i) / 8] & (1 << ((i) % 8)))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define REWARD_DEATH -1.0f

#define LOG_SCORE_REWARD_DEATH -1

#define MAX_INVENTORY_ITEM 100
#define HP_REWARD_FOOD 20
#define HP_LOSS_PER_STEP 1
#define MAX_HP 100 

#define DOWN 0 
#define UP 1
#define RIGHT 2
#define LEFT 3
#define NO_MOVE 4
#define INTERACT 5
#define EAT 6

#define SPRITE_SIZE 128 
#define TILE_SIZE 64

#define HEALTH_BAR_WIDTH 48
#define HEALTH_BAR_HEIGHT 6

typedef struct Log Log;
struct Log {
  float perf;
  float score;
  float episode_return;
  float steals;
  float collects;
  float n;
};

typedef struct Agent Agent;
struct Agent {
  int r;
  int c;
  int id;
  int direction;
  int held_food;
  float hp;
  int start_tick;
  unsigned char anim;
};

// typedef struct FoodList FoodList;
// struct FoodList {
//   int *indexes; // Grid flattened index positions
//   int size;
// };

// FoodList *allocate_foodlist(int size) {
//   FoodList *foods = (FoodList *)calloc(1, sizeof(FoodList));
//   foods->indexes = (int *)calloc(size, sizeof(int));
//   foods->size = 0;
//   return foods;
// }

// void free_foodlist(FoodList *foods) {
//   free(foods->indexes);
//   free(foods);
// }


// Usefull for biomes where resource can spawn into
typedef struct Biome_idx Biome_idx;
struct Biome_idx {
  int *grass_idx;
  int grass_count;

  int *dirt_idx;
  int dirt_count;
};

typedef struct Renderer Renderer;
typedef struct PredPrey PredPrey;
struct PredPrey {
  Renderer* client;
  int width;
  int height;
  int num_agents;

  int vision;
  int vision_window;
  int obs_size;

  int tick;
  int last_agent_dead_tick;

  float reward_food;

  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  unsigned char *truncations;
  unsigned char *masks;

  Agent *agents;

  Log log;
  Log* agent_logs;

  // FoodList *foods;
  float food_base_spawn_rate;
  float max_food;
  int food_count;

  unsigned char *terrain; // Array of terrain types, size width*height
  unsigned char *items; // Array of item types, size width*height
  short *pids; // Array of entity idx, size width*height

  Biome_idx biome_idxs;
};

void init_biome_idx(PredPrey *env) {
  // I only do that once on load - will need to do that every reset if map changes
  env->biome_idxs.grass_idx = (int *)calloc(env->width * env->height, sizeof(int));
  env->biome_idxs.dirt_idx = (int *)calloc(env->width * env->height, sizeof(int));

  env->biome_idxs.grass_count = 0;
  env->biome_idxs.dirt_count = 0;

  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int grid_idx = r * env->width + c;
      unsigned char tile = env->terrain[grid_idx];
      if (tile == TILE_GRASS) {
        env->biome_idxs.grass_idx[env->biome_idxs.grass_count++] = grid_idx;
      } else if (tile == TILE_DIRT) {
        env->biome_idxs.dirt_idx[env->biome_idxs.dirt_count++] = grid_idx;
      }
    }
  }
}

void add_log(PredPrey *env, Log *log) {
  //TODO fix perf calculation
  env->log.perf = fmaxf(0, log->score/MAX_TIMESTEPS);
  env->log.steals += log->steals;
  env->log.episode_return += log->episode_return;
  env->log.score += log->score;
  env->log.collects += log->collects;
  env->log.n += 1;
}

void init_cenv(PredPrey *env) {
  env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
  env->vision_window = 2 * env->vision + 1;
  env->obs_size = (env->vision_window * env->vision_window) * MAX_CELL_OBS + 1;
  // env->foods = allocate_foodlist(env->width * env->height);
  env->agent_logs = (Log *)calloc(env->num_agents, sizeof(Log));
  env->masks = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
  // Arbitrarly set max food to a proportion of available tiles
  env->max_food = 0.55 * (
    (env->width * env->height) - (
      env->vision*env->width*2 +
      env->vision*2*(env->height - 2*env->vision)
    ) - env->num_agents); 
  env->terrain = (unsigned char *)calloc(env->width * env->height, sizeof(unsigned char));
  env->items = (unsigned char *)calloc(env->width * env->height, sizeof(unsigned char));
  env->pids = (short *)calloc(env->width * env->height, sizeof(short));
}

void allocate_cenv(PredPrey *env) {
  // Called by C stuff
  int obs_size = ((2 * env->vision + 1) * (2 * env->vision + 1)) * MAX_CELL_OBS + 1;
  env->observations = (float *)calloc(env->num_agents * obs_size,
                                              sizeof(float));
  env->actions = (int *)calloc(env->num_agents, sizeof(unsigned int));
  env->rewards = (float *)calloc(env->num_agents, sizeof(float));
  env->terminals =
      (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
  env->truncations = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
  init_cenv(env);
}

void free_biome(PredPrey *env) {
  free(env->biome_idxs.grass_idx);
  free(env->biome_idxs.dirt_idx);
}

void c_close(PredPrey *env) {
  free(env->agents);
  // free_foodlist(env->foods);
  free(env->masks);
  free(env->agent_logs);
  free(env->terrain);
  free(env->items);
  free(env->pids);
  free_biome(env);
}

void free_CEnv(PredPrey *env) {
  free(env->observations);
  free(env->actions);
  free(env->rewards);
  free(env->terminals);
  free(env->truncations);
  c_close(env);
}

int flat_idx(PredPrey *env, int r, int c) { return r * env->width + c; }

void reward_agent(PredPrey *env, int agent_id, float reward) {
  // Simple helper function which loggs as well
  env->rewards[agent_id] += reward;
  env->agent_logs[agent_id].episode_return += reward;
}

void init_foods(PredPrey *env) {
  // Fill dirt area with food
  for (int i = 0; i < env->biome_idxs.dirt_count; i++) {
    int grid_idx = env->biome_idxs.dirt_idx[i];
    if (env->items[grid_idx] == EMPTY) {
      env->items[grid_idx] = ITEM_FOOD;
      env->food_count += 1;
    }
  }
}

// void spawn_foods_organic(PredPrey *env) {
//   // After each step, check existing foods and spawns new food in the
//   // neighborhood Iterates over food_list for efficiency instead of the entire
//   // grid.
//   // Only do it if the number of foods is less than max_foods
//   if (env->foods->size >= env->max_food) {
//     return;
//   }
//   FoodList *foods = env->foods;
//   int original_size = foods->size;
//   for (int i = 0; i < original_size; i++) {
//     int idx = foods->indexes[i];
//     int offset = idx - env->width - 1; // Food spawn in 1 radius
//     int r = offset / env->width;
//     int c = offset % env->width;
//     for (int ri = 0; ri < 3; ri++) {
//       for (int ci = 0; ci < 3; ci++) {
//         int neighboor_idx = flat_idx(env, (r + ri), (c + ci));
//         if (env->terrain[neighboor_idx] != TILE_DIRT && env->items[neighboor_idx] != 0) {
//           continue;
//         }
//         switch (env->items[idx]) {
//         // %Chance spawning new food
//         case ITEM_FOOD:
//           if ((rand() / (double)RAND_MAX) < env->food_base_spawn_rate) {
//             add_food(env, neighboor_idx, env->items[idx]);
//           }
//           break;
//         }
//       }
//     }
//   }

//   // // Each turn there is random probability for a food to spawn at a random
//   // // location To cope with resource depletion
//   // int normalizer = (env->width * env->height) / 576;
//   // if ((rand() / (double)RAND_MAX) <
//   //     min((env->food_base_spawn_rate * 2 * normalizer), 1e-2)) {
//   //   spawn_food_random(env, NORMAL_FOOD);
//   // }
// }

bool spawn_food_random(PredPrey *env){
  // Try x time to spawn food wihtin the DIRT area
  if (env->food_count >= env->max_food) {
    return false;
  }
  int attempts = 0;
  while (attempts ++ < 100){
    int rand_idx = rand() % env->biome_idxs.dirt_count;
    int grid_idx = env->biome_idxs.dirt_idx[rand_idx];

    if (env->items[grid_idx] == EMPTY) {
      env->items[grid_idx] = ITEM_FOOD;
      env->food_count++;
      return true;
    }
  }
  return false;
}

void regrow_food(PredPrey *env){
  // Regrow food in all dirt tiles that do not have food already with some probability
  for (int i = 0; i < env->biome_idxs.dirt_count; i++) {
    int grid_idx = env->biome_idxs.dirt_idx[i];
    if (env->items[grid_idx] == EMPTY && rand() / (double)RAND_MAX < env->food_base_spawn_rate) {
      env->items[grid_idx] = ITEM_FOOD;
      env->food_count += 1;
    }
  }
}

void spawn_items(PredPrey *env) {
  // Currently items are only spawning every step in their corresponding BIOME
  regrow_food(env);
  // if (rand() / (double)RAND_MAX < env->food_base_spawn_rate) {
  //   spawn_food_random(env);
  // }
}

void compute_observations(PredPrey *env) {
  for (int i = 0; i < env->num_agents; i++) {
    int obs_idx = i * env->obs_size;
    Agent *agent = &env->agents[i];
    if (agent->hp <= 0) {
      for (int j = 0; j < env->obs_size; j++) {
        env->observations[obs_idx++] = 0.0f;
      }
      continue;
    }
    // int obs_offset = (i * env->obs_size);
    int r_offset = agent->r - env->vision;
    int c_offset = agent->c - env->vision;
    for (int r = 0; r < env->vision_window; r++) {
      for (int c = 0; c < env->vision_window; c++) {
        int grid_idx = flat_idx(env,r_offset + r, c_offset + c);
        unsigned char item_idx = env->items[grid_idx];
        short entity_id = env->pids[grid_idx];

        // First obs is items if any
        env->observations[obs_idx++] = item_idx;
        float hp_norm = 0.0f;
        float food_norm = 0.0f;
        if (entity_id != -1) {
            Agent *grid_agent = &env->agents[entity_id]; 
            hp_norm = grid_agent->hp / (float)MAX_HP;
            food_norm = grid_agent->held_food / (float)MAX_INVENTORY_ITEM;
        }

        env->observations[obs_idx++] = hp_norm;
        env->observations[obs_idx++] = food_norm;

      } 
    }
    //Agent also get its direction
    env->observations[obs_idx++] = agent->direction;
  }
}

void remove_agent(PredPrey *env, int agent_id) {
  Agent *agent = &env->agents[agent_id];
  if (agent->r < 0 || agent->c < 0) {
    return;
  }
  int grid_idx = flat_idx(env, agent->r, agent->c);
  env->pids[grid_idx] = -1;
  agent->r = -1;
  agent->c = -1;
}

void add_hp(PredPrey *env, int agent_id, float hp) {
  Agent *agent = &env->agents[agent_id];
  agent->hp += hp;
  if (agent->hp > MAX_HP) {
    agent->hp = MAX_HP;
  } else if (agent->hp <= 0) {
    agent->hp = 0;
    env->agent_logs[agent->id].score = env->tick - agent->start_tick;
    reward_agent(env, agent_id, REWARD_DEATH);
    env->terminals[agent->id] = 1;
    add_log(env, &env->agent_logs[agent_id]);
    remove_agent(env, agent_id);
    env->last_agent_dead_tick = env->tick;
  }
}

void remove_hp(PredPrey *env, int agent_id, float hp) {
    add_hp(env, agent_id, -hp);
}

void save_terrain_to_file(PredPrey *env, const char *filename) {
    char filepath[512];
    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    char *last_slash = strrchr(dir, '/');
    if (last_slash) *last_slash = '\0';
    snprintf(filepath, sizeof(filepath), "%s/%s", dir, filename);
    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    fprintf(file, "#ifndef H\n#define H\n\n");
    fprintf(file, "#define HEIGHT %d\n", env->height);
    fprintf(file, "#define WIDTH %d\n\n", env->width);
    fprintf(file, "static const unsigned char terrain[HEIGHT][WIDTH] = {\n");

    for (int r = 0; r < env->height; r++) {
        fprintf(file, "    {");
        for (int c = 0; c < env->width; c++) {
            unsigned char val = env->terrain[r * env->width + c];
            fprintf(file, "0x%02X%s", val, (c == env->width - 1) ? "" : ", ");
        }
        fprintf(file, "}%s\n", (r == env->height - 1) ? "" : ",");
    }
    fprintf(file, "};\n\n#endif // H\n");
    fclose(file);
}

void make_grid_from_scratch(PredPrey *env){
  // top walling
  for (int r = 0; r < env->vision; r++) {
    memset(env->terrain + (r * env->width), TILE_WATER,
           env->width * sizeof(env->terrain[0]));
  }
  // left side walling
  for (int r = 0; r < env->height; r++) {
    memset(env->terrain + (r * env->width), TILE_WATER,
           env->vision * sizeof(env->terrain[0]));
  }
  // bottom walling
  for (int r = env->height - env->vision; r < env->height; r++) {
    memset(env->terrain + (r * env->width), TILE_WATER,
           env->width * sizeof(env->terrain[0]));
  }

  // right side walling
  for (int r = 0; r < env->height; r++) {
    memset(env->terrain + (r * env->width) + (env->width - env->vision), TILE_WATER,
           env->vision * sizeof(env->terrain[0]));
  }

  // Calculate dimensions
  int inner_width = env->width - (2 * env->vision);
  int inner_height = env->height - (2 * env->vision);
  int house_size = inner_height / 3;
  int field_width = inner_width - house_size;

  // Fill inner terrain
  for (int r = env->vision; r < env->height - env->vision; r++) {
    int row_offset = r * env->width + env->vision; // row offset in global grid
    int row_in_inner = r - env->vision; // row index in inner area reference
    
    if (row_in_inner >= inner_height - house_size) {
      // Bottom rows: HOUSE on left, DIRT on right
      memset(env->terrain + row_offset, TILE_HOUSE, house_size * sizeof(env->terrain[0]));
      memset(env->terrain + row_offset + house_size, TILE_DIRT, field_width * sizeof(env->terrain[0]));
    } else {
      // Top rows: all GRASS
      memset(env->terrain + row_offset, TILE_GRASS, inner_width * sizeof(env->terrain[0]));
    }
  }

  save_terrain_to_file(env, "terrain.h");
}

void spawn_agent(PredPrey *env, int i){
  Agent *agent = &env->agents[i];
  agent->id = i;
  agent->hp = 100;
  agent->start_tick = env->tick;
  agent->held_food = 0;

  int adr = 0;

  bool allocated = false;
  while (!allocated) {
    adr = rand() % (env->height * env->width);
    if (env->pids[adr] == -1 && env->terrain[adr] != TILE_WATER) {
      int r = adr / env->width;
      int c = adr % env->width;
      agent->r = r;
      agent->c = c;
      allocated = true;
    }
  }
  assert(env->pids[adr] == -1);
  env->pids[adr] = agent->id;
  env->agent_logs[i] = (Log){0};
}
void c_reset(PredPrey *env) {
  env->tick = 0;
  memset(env->agent_logs, 0, env->num_agents * sizeof(Log));
  env->log = (Log){0};
  env->food_count = 0;
  // env->foods->size = 0;
  // memset(env->foods->indexes, 0, env->width * env->height * sizeof(int));

  // make_grid_from_scratch(env);
  memcpy(env->terrain, terrain, env->width * env->height * sizeof(unsigned char));
  init_biome_idx(env); // TODO dump that for current map & memcpy

  memset(env->items, EMPTY, env->width * env->height * sizeof(unsigned char));
  // Carrefull here but -1 works with memset
  memset(env->pids, -1, env->width * env->height * sizeof(short));

  for (int i = 0; i < env->num_agents; i++) {
    spawn_agent(env, i);
  }

  init_foods(env);
  memset(env->observations, 0, env->num_agents * env->obs_size * sizeof(float));
  memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
  memset(env->masks, 1, env->num_agents * sizeof(unsigned char));
  compute_observations(env);
}

bool is_obstacle(PredPrey *env, int idx) {
  int tile = env->terrain[idx];
  if (tile == TILE_WATER) {
    return true;
  }

  short entity_id = env->pids[idx];
  if (entity_id != -1){
    return true;
  }
  return false; 
}

void step_agent(PredPrey *env, int i) {

  Agent *agent = &env->agents[i];

  int action = env->actions[i];

  /////////////////////////////////
  // Movement
  ////////////////////////////////
  int dr = 0;
  int dc = 0;

  switch (action) {
  case UP:
    dr = -1;
    agent->direction = UP;
    break;
  case DOWN:
    dr = 1;
    agent->direction = DOWN;
    break; 
  case LEFT:
    dc = -1;
    agent->direction = LEFT;
    break; 
  case RIGHT:
    dc = 1;
    agent->direction = RIGHT;
    break;
  case NO_MOVE:
    return;
  }
  // Get next row and column
  int next_r = agent->r + dr;
  int next_c = agent->c + dc;

  int prev_grid_idx = flat_idx(env, agent->r, agent->c);
  int next_grid_idx = flat_idx(env, next_r, next_c);
  if (is_obstacle(env, next_grid_idx)) {
    next_grid_idx = prev_grid_idx;
    next_r = agent->r;
    next_c = agent->c;
  }
  // update the grid tiles values
  env->pids[prev_grid_idx] = -1;
  env->pids[next_grid_idx] = agent->id;
  agent->r = next_r;
  agent->c = next_c;


  /////////////////////////////////
  // Interaction / Eating
  ////////////////////////////////
  if (action == INTERACT) {
    int facing_tile_idx = 0;
    switch (agent->direction) {
    case UP:
      facing_tile_idx = flat_idx(env, agent->r - 1, agent->c);
      break;
    case DOWN:
      facing_tile_idx = flat_idx(env, agent->r + 1, agent->c);
      break;
    case LEFT:
      facing_tile_idx = flat_idx(env, agent->r, agent->c - 1);
      break;
    case RIGHT:
      facing_tile_idx = flat_idx(env, agent->r, agent->c + 1);
      break;
    }

    int facing_tile = env->pids[facing_tile_idx];
    if (facing_tile != -1) {
      Agent *other_agent = &env->agents[facing_tile];
      // Steal food from other agent
      if (other_agent->held_food > 0) {
        agent->held_food = other_agent->held_food;
        other_agent->held_food = 0;
        env->agent_logs[i].steals += 1;
      }
    } 

    if (env->items[next_grid_idx] == ITEM_FOOD) {
      // Pick up food
      agent->held_food += 1;
      if (agent->held_food > MAX_INVENTORY_ITEM) {
        agent->held_food = MAX_INVENTORY_ITEM;
      }
      env->items[next_grid_idx] = EMPTY;
      env->food_count -= 1;
      env->agent_logs[i].collects += 1;

    }
  }
  
  if (action == EAT) {
    if (agent->held_food > 0) {
      agent->held_food -= 1;
      add_hp(env, i, HP_REWARD_FOOD);
      reward_agent(env, i, env->reward_food);
    }
  }
  return;
}

void c_step(PredPrey *env) {
  env->tick++;

  memset(env->rewards, 0, env->num_agents * sizeof(float));

  for (int i = 0; i < env->num_agents; i++) {
    if (env->agents[i].hp == 0) {
      spawn_agent(env, i);
      continue;
    }
    step_agent(env, i);
    remove_hp(env, i, HP_LOSS_PER_STEP);
  }

  if (env->tick - env->last_agent_dead_tick >= MAX_TIMESTEPS) {
    c_reset(env);
    return;
  }
  spawn_items(env);
  compute_observations(env);
}

// Animations
#define ANIM_IDLE 0
#define ANIM_MOVE 1
#define ANIM_DEATH 2
#define ANIM_ATTACK 3

typedef struct Animation Animation;
struct Animation {
    int num_frames;
    int tiles_traveled;
    int offset; // Number of tiles from the top of the sheet
    int frames[10]; // Order of frames in sheet, left to right
};

Animation ANIMATIONS[4] = {
    (Animation){ // ANIM_IDLE
        .num_frames = 1,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_MOVE
        .num_frames = 6,
        .tiles_traveled = 1,
        .offset = 4,
        .frames = {0, 1, 2, 3, 4, 5, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_DEATH
        .num_frames = 3,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {5, 6, 7, 0, 0, 0, 0, 0, 0, 0}
    },
    (Animation){ // ANIM_ATTACK
        .num_frames = 2,
        .tiles_traveled = 0,
        .offset = 0,
        .frames = {1, 2, 0, 0, 0, 0, 0, 0, 0, 0}
    },
};

//////////////////////////////////////////////////////////////////
// Rendering
//////////////////////////////////////////////////////////////////

#define BLANK      CLITERAL(Color){ 0, 0, 0, 0 }           // Blank (Transparent)

struct Renderer {
  int cell_size;
  int width;
  int height;
  Texture2D agents[5][10];
  Texture2D tiles;
  Texture2D items;
  Font font;
};

Renderer *init_renderer(int width, int height) {
  Renderer *renderer = (Renderer *)calloc(1, sizeof(Renderer));
  renderer->width = width;
  renderer->height = height;

  InitWindow(width * TILE_SIZE, height * TILE_SIZE, "Predator Prey");
  SetTargetFPS(10);

  for (int i = 0; i < 10; i++) {
    renderer->agents[0][i] = LoadTexture(TextFormat("resources/nmmo3/neutral_%d.png", i));
    renderer->agents[1][i] = LoadTexture(TextFormat("resources/nmmo3/fire_%d.png", i));
    renderer->agents[2][i] = LoadTexture(TextFormat("resources/nmmo3/water_%d.png", i));
    renderer->agents[3][i] = LoadTexture(TextFormat("resources/nmmo3/earth_%d.png", i));
    renderer->agents[4][i] = LoadTexture(TextFormat("resources/nmmo3/air_%d.png", i));
  }
  renderer->tiles = LoadTexture("resources/nmmo3/merged_sheet.png");
  renderer->items = LoadTexture("resources/nmmo3/items_condensed.png");
  renderer->font = LoadFont("resources/nmmo3/ManaSeedBody.ttf");

  return renderer;
}

void close_renderer(Renderer *renderer) {
  CloseWindow();
  free(renderer);
}

   
void draw_health_bar(int bar_x, int bar_y, int health, int max_health) {
    DrawRectangle(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, RED);
    DrawRectangle(bar_x, bar_y,
        HEALTH_BAR_WIDTH * health / max_health,
        HEALTH_BAR_HEIGHT, GREEN);
    DrawRectangleLines(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, BLACK);
}

void c_render(PredPrey *env) {
  if (env->client == NULL) {
      env->client = init_renderer(env->width, env->height);
  };
  Renderer *renderer = env->client;

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(BLANK);

  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int adr = flat_idx(env, r, c);

      int terrain_type = env->terrain[adr];
      int item_type = env->items[adr];
      int entity_id = env->pids[adr];
      
      Vector2 pos = {
          .x = c * TILE_SIZE,
          .y = r * TILE_SIZE,
      };

      // Drawing terrain
      Rectangle source_rect = {
        .x = 0,
        .y = terrain_type * TILE_SIZE,
        .width = TILE_SIZE,
        .height = TILE_SIZE
      };
      DrawTextureRec(renderer->tiles, source_rect, pos, WHITE);

      // Drawing items
      if (item_type != 0) {
        Rectangle source_rect = {
          .x = 4 * TILE_SIZE,
          .y = item_type * TILE_SIZE,
          .width = TILE_SIZE,
          .height = TILE_SIZE
        };
        DrawTextureRec(renderer->items, source_rect, pos, WHITE);
      }
      
      if (entity_id != -1) {
        Agent *agent = &env->agents[entity_id];
        Animation animation = ANIMATIONS[agent->anim];
        int starting_sprite_y = (env->agents[entity_id].direction) * SPRITE_SIZE;
        int x_pos = (c - 0.5f)*TILE_SIZE;
        int y_pos = (r - 0.5f)*TILE_SIZE;
        Vector2 pos = {
            .x = x_pos,
            .y = y_pos,
        };        
        Rectangle source_rect = {
          .x = 0,
          .y = starting_sprite_y,
          .width = SPRITE_SIZE,
          .height = SPRITE_SIZE
        };
        DrawTextureRec(renderer->agents[entity_id%5][entity_id%10], source_rect, pos, WHITE);

        // Draw HP bar 
        int bar_x = x_pos + TILE_SIZE - HEALTH_BAR_WIDTH/2;
        int bar_y = y_pos;
        draw_health_bar(bar_x, bar_y, agent->hp, MAX_HP);

        // Food Number in inventory
        char* txt;
        Color color;
        txt = (char*) TextFormat("%d: F: %d", entity_id, agent->held_food);
        color = GREEN;

        Vector2 text_pos = {.x = bar_x, .y = bar_y - 20};
        DrawTextEx(renderer->font, txt, text_pos, 21, 1, color);

      }
    }
  }
  EndDrawing();
}
