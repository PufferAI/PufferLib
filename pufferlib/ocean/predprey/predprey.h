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


// Time
#define TICK_PER_HOUR 4 //20
#define HOURS_PER_DAY 4 //24
#define DAY_PER_MONTH 3 //5
#define MONTHS_PER_YEAR 4 //4
#define TICK_PER_DAY (TICK_PER_HOUR * HOURS_PER_DAY)
#define TICK_PER_MONTH (TICK_PER_DAY * DAY_PER_MONTH)
#define TICK_PER_YEAR (TICK_PER_MONTH * MONTHS_PER_YEAR)

#define MAX_TIMESTEPS (TICK_PER_YEAR*10) //2000

// Tiles
#define TILE_SOIL 0 
#define TILE_FLOOR_WOOD 1
#define TILE_WATER 2
#define TILE_GRASS 3
// #define TILE_WATER 4
// #define TILE_GRASS 8

// Items
#define EMPTY 0
#define ITEM_WOOD 1
#define ITEM_FOOD 2
#define ITEM_CHEST 3
#define ITEM_FIREPLACE_LIT 4
#define ITEM_FIREPLACE 5

// Entities
#define ENTITY_AGENT 0

// Animations
#define ANIM_IDLE 0
#define ANIM_INTERACT 1
#define ANIM_EAT 2


#define MAX_CELL_OBS 5 // Maximum number of info per cell in observations
#define LOG_BUFFER_SIZE 8192

#define SET_BIT(arr, i) (arr[(i) / 8] |= (1 << ((i) % 8)))
#define CLEAR_BIT(arr, i) (arr[(i) / 8] &= ~(1 << ((i) % 8)))
#define CHECK_BIT(arr, i) (arr[(i) / 8] & (1 << ((i) % 8)))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define REWARD_DEATH -1.0f

#define LOG_SCORE_REWARD_DEATH -1

#define MAX_INVENTORY_ITEM 100
#define MAX_CHEST_CAPACITY 1000
#define MAX_COLDNESS 100
#define MAX_FIRE_TIME 10
#define HP_REWARD_FOOD 20
#define HP_LOSS_PER_HOUR 2
#define COLDNESS_LOSS_PER_HOUR 2
#define HP_LOSS_COLD 10
#define MAX_HP 100 
#define START_HP 80

// Actions
#define DOWN 0 
#define UP 1
#define RIGHT 2
#define LEFT 3
#define NO_MOVE 4
#define INTERACT 5
#define EAT 6

// Spawn rates
#define FOOD_SPAWN_RATE 0.1f
#define WOOD_SPAWN_RATE 0.1f

#define SPRITE_SIZE_ENTITY 128 
#define TILE_SIZE_ENV 64
#define SPRITE_SIZE 64

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
  int food_amt;
  int wood_amt;
  float hp;
  int start_tick;
  unsigned char anim;
  unsigned char coldness;
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

  int *house_idx;
  int house_count;
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

  float reward_death_scale;
  float reward_eat;
  float reward_collect;
  float timestep_reward;
  float hp_reward_scale;
  float held_food_reward_scale;
  float reward_fireplace_lit;
  float reward_store_chest;

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
  float max_food;
  int food_count;
  float max_wood;
  int wood_count;

  unsigned char *terrain; // Array of terrain types, size width*height
  unsigned char *items; // Array of item types, size width*height
  short *pids; // Array of entity idx, size width*height

  Biome_idx biome_idxs;

  int chest_food_amt;
  bool is_fireplace_lit;
  int fire_time_remaining;
};

void init_biome_idx(PredPrey *env) {
  // I only do that once on load - will need to do that every reset if map changes
  env->biome_idxs.grass_idx = (int *)calloc(env->width * env->height, sizeof(int));
  env->biome_idxs.dirt_idx = (int *)calloc(env->width * env->height, sizeof(int));
  env->biome_idxs.house_idx = (int *)calloc(env->width * env->height, sizeof(int));

  env->biome_idxs.grass_count = 0;
  env->biome_idxs.dirt_count = 0;
  env->biome_idxs.house_count = 0;

  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int grid_idx = r * env->width + c;
      unsigned char tile = env->terrain[grid_idx];
      if (tile == TILE_GRASS) {
        env->biome_idxs.grass_idx[env->biome_idxs.grass_count++] = grid_idx;
      } else if (tile == TILE_SOIL) {
        env->biome_idxs.dirt_idx[env->biome_idxs.dirt_count++] = grid_idx;
      } else if (tile == TILE_FLOOR_WOOD) {
        env->biome_idxs.house_idx[env->biome_idxs.house_count++] = grid_idx;
      }
    }
  }
}

void add_log(PredPrey *env, Log *log) {
  //TODO fix perf calculation
  env->log.perf += fmaxf(0, log->score/MAX_TIMESTEPS);
  env->log.steals += log->steals;
  env->log.episode_return += log->episode_return;
  env->log.score += log->score;
  env->log.collects += log->collects;
  env->log.n += 1;
}

void add_agent_log(PredPrey *env, int agent_id) {
  int time_alive = env->tick - env->agents[agent_id].start_tick;
  assert(time_alive > 0);
  env->agent_logs[agent_id].score = time_alive;
  env->agent_logs[agent_id].steals /= time_alive;
  env->agent_logs[agent_id].collects /= time_alive;
  add_log(env, &env->agent_logs[agent_id]);
  
  // //I don't fully reset because the agent might not be dead yet
  // //So I still need to keep track of collect & steal counts
  // //episode_returns will accumulate over life
  // //score will be overwritten next time
  // env->agent_logs[agent_id].steals *= time_alive;
  // env->agent_logs[agent_id].collects *= time_alive;
}

void save_terrain_to_file(PredPrey *env, const char *filename) {
    char filepath[1024];
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
      // Bottom rows: HOUSE on left, SOIL on right
      memset(env->terrain + row_offset, TILE_FLOOR_WOOD, house_size * sizeof(env->terrain[0]));
      memset(env->terrain + row_offset + house_size, TILE_SOIL, field_width * sizeof(env->terrain[0]));
    } else if (row_in_inner == inner_height - house_size - 1) {
      // One row of Water between house and grass 
      memset(env->terrain + row_offset, TILE_WATER, inner_width * sizeof(env->terrain[0]));
      // With three tiles of GRASS on the left 
      memset(env->terrain + row_offset, TILE_GRASS, 3 * sizeof(env->terrain[0]));
    } else {
      // Top rows: all GRASS
      memset(env->terrain + row_offset, TILE_GRASS, inner_width * sizeof(env->terrain[0]));
    }
  }

  save_terrain_to_file(env, "terrain.h");
}

void init_cenv(PredPrey *env) {
  env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
  env->vision_window = 2 * env->vision + 1;
  env->obs_size = (env->vision_window * env->vision_window) * MAX_CELL_OBS + 5;
  // env->foods = allocate_foodlist(env->width * env->height);
  env->agent_logs = (Log *)calloc(env->num_agents, sizeof(Log));
  env->masks = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
  env->terrain = (unsigned char *)calloc(env->width * env->height, sizeof(unsigned char));
  env->items = (unsigned char *)calloc(env->width * env->height, sizeof(unsigned char));
  env->pids = (short *)calloc(env->width * env->height, sizeof(short));

  // make_grid_from_scratch(env);
  memcpy(env->terrain, terrain, env->width * env->height * sizeof(unsigned char));
  init_biome_idx(env);

  // Arbitrarly set max food to a proportion of available tiles
  env->max_food = 0.9 * env->biome_idxs.dirt_count;
  env->max_wood = 0.5 * env->biome_idxs.grass_count;
}

void allocate_cenv(PredPrey *env) {
  // Called by C stuff
  int obs_size = ((2 * env->vision + 1) * (2 * env->vision + 1)) * MAX_CELL_OBS + 5;
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
  free(env->biome_idxs.house_idx);
}

void c_close(PredPrey *env) {
  free(env->agents);
  // free_foodlist(env->foods);
  free(env->agent_logs);
  free(env->terrain);
  free(env->items);
  free(env->pids);
  free_biome(env);
  env->client = NULL;
}

void free_CEnv(PredPrey *env) {
  free(env->observations);
  free(env->actions);
  free(env->masks);
  free(env->rewards);
  free(env->terminals);
  free(env->truncations);
  c_close(env);
}

int flat_idx(PredPrey *env, int r, int c) { return r * env->width + c; }
int hour_of_day(PredPrey *env) {
  int tick_in_day = env->tick % TICK_PER_DAY;
  return tick_in_day / TICK_PER_HOUR;
}
int day_of_month(PredPrey *env) {
  int tick_in_month = env->tick % TICK_PER_MONTH;
  return tick_in_month / TICK_PER_DAY;
}
int month_of_year(PredPrey *env) {
  int tick_in_year = env->tick % TICK_PER_YEAR;
  return tick_in_year / TICK_PER_MONTH;
}

void reward_agent(PredPrey *env, int agent_id, float reward) {
  // Simple helper function which loggs as well
  env->rewards[agent_id] += reward;
  env->agent_logs[agent_id].episode_return += reward;
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

void init_foods(PredPrey *env) {
  // Fill dirt area with food
  for (int i = 0; i < env->biome_idxs.dirt_count; i++) {
    int grid_idx = env->biome_idxs.dirt_idx[i];
    if (env->items[grid_idx] == EMPTY && 
        rand() / (double)RAND_MAX < 0.2 &&
        env->food_count < env->max_food
    ) {
      env->items[grid_idx] = ITEM_FOOD;
      env->food_count += 1;
    }
  }
}

void init_woods(PredPrey *env) {
  // Fill grass area with wood
  for (int i = 0; i < env->biome_idxs.grass_count; i++) {
    int grid_idx = env->biome_idxs.grass_idx[i];
    if (env->items[grid_idx] == EMPTY && 
        rand() / (double)RAND_MAX < 0.3 &&
        env->wood_count < env->max_wood
      ) { 
      env->items[grid_idx] = ITEM_WOOD;
      env->wood_count += 1;
    }
  }
}

void init_items(PredPrey *env) {
  init_foods(env);
  // Randomly place fireplace and chest in house area
  bool allocated_fireplace = false;
  while (!allocated_fireplace) {
    int rand_idx = rand() % env->biome_idxs.house_count;
    int grid_idx = env->biome_idxs.house_idx[rand_idx];
    if (env->items[grid_idx] == EMPTY) {
      env->items[grid_idx] = ITEM_FIREPLACE;
      allocated_fireplace = true;
    }
  }
  bool allocated_chest = false;
  while (!allocated_chest) {
    int rand_idx = rand() % env->biome_idxs.house_count;
    int grid_idx = env->biome_idxs.house_idx[rand_idx];
    if (env->items[grid_idx] == EMPTY) {
      env->items[grid_idx] = ITEM_CHEST;
      allocated_chest = true;
    }
  }
  init_woods(env);
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
//         if (env->terrain[neighboor_idx] != TILE_SOIL && env->items[neighboor_idx] != 0) {
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
  
  // Regrow only if we are in the first half of the year
  if (month_of_year(env) >= MONTHS_PER_YEAR / 2){
    return;
  }

  for (int i = 0; i < env->biome_idxs.dirt_count; i++) {
    int grid_idx = env->biome_idxs.dirt_idx[i];
    if (
      env->items[grid_idx] == EMPTY && 
      rand() / (double)RAND_MAX < FOOD_SPAWN_RATE &&
      env->food_count < env->max_food
    ) {
      env->items[grid_idx] = ITEM_FOOD;
      env->food_count += 1;
    }
  }
}

void regrow_wood(PredPrey *env){
  // Wood regrow only in the second part of the year
  if (month_of_year(env) < MONTHS_PER_YEAR / 2){
    return;
  }
  // Regrow wood in all grass tiles that do not have wood already with some probability
  for (int i = 0; i < env->biome_idxs.grass_count; i++) {
    int grid_idx = env->biome_idxs.grass_idx[i];
    if (env->items[grid_idx] == EMPTY && 
        rand() / (double)RAND_MAX < WOOD_SPAWN_RATE &&
        env->wood_count < env->max_wood
      ) {
      env->items[grid_idx] = ITEM_WOOD;
      env->wood_count += 1;
    }
  }
}
void spawn_items(PredPrey *env) {
  // Currently items are only spawning every step in their corresponding BIOME
  regrow_food(env);
  regrow_wood(env);
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

        // First obs is terrain
        env->observations[obs_idx++] = (float)env->terrain[grid_idx];
        // Second is item 
        env->observations[obs_idx++] = (float)item_idx;
        // Thirds is entity id 
        env->observations[obs_idx++] = (float)entity_id;
        float hp_norm = 0.0f;
        float food_norm = 0.0f;
        if (entity_id != -1) {
            Agent *grid_agent = &env->agents[entity_id]; 
            hp_norm = grid_agent->hp / (float)MAX_HP;
            food_norm = grid_agent->food_amt / (float)MAX_INVENTORY_ITEM;
        }

        env->observations[obs_idx++] = hp_norm;
        env->observations[obs_idx++] = food_norm;

      } 
    }
    //Agent also get its direction
    env->observations[obs_idx++] = agent->direction;
    env->observations[obs_idx++] = (float)env->is_fireplace_lit;
    env->observations[obs_idx++] = (float)env->fire_time_remaining/(float)MAX_FIRE_TIME;
    env->observations[obs_idx++] = (float)env->chest_food_amt/(float)MAX_CHEST_CAPACITY;
    env->observations[obs_idx++] = (float)agent->coldness/(float)MAX_COLDNESS;
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
  if (agent->hp == 0) {
    return;
  }
  agent->hp += hp;
  if (agent->hp > MAX_HP) {
    agent->hp = MAX_HP;
  } else if (agent->hp <= 0) {
    agent->hp = 0;
    int time_alive = env->tick - agent->start_tick;
    float reward = (((float)time_alive-START_HP) / (float)MAX_TIMESTEPS) * env->reward_death_scale;
    reward_agent(env, agent_id, reward);
    env->terminals[agent->id] = 1;
    add_agent_log(env, agent_id);    
    remove_agent(env, agent_id);
    env->last_agent_dead_tick = env->tick;
  }
}

void remove_hp(PredPrey *env, int agent_id, float hp) {
    add_hp(env, agent_id, -hp);
}

void spawn_agent(PredPrey *env, int agent_id){
  Agent *agent = &env->agents[agent_id];
  agent->id = agent_id;
  agent->hp = START_HP;
  agent->coldness = 0;
  agent->start_tick = env->tick;
  agent->food_amt = 0;
  agent->wood_amt = 0;

  // Spawn only in the house area
  int adr = 0;
  bool allocated = false;
  while (!allocated){
    adr = env->biome_idxs.house_idx[rand() % env->biome_idxs.house_count];
    if (is_obstacle(env, adr)){
      continue;
    }
    int r = adr / env->width;
    int c = adr % env->width;
    agent->r = r;
    agent->c = c;
    allocated = true;
  }
  assert(env->pids[adr] == -1);
  env->pids[adr] = agent->id;
  env->agent_logs[agent_id] = (Log){0};
}

void teleport_rnd(PredPrey *env, int agent_id){
  // Teleport an agent to a random free tile on the map
  Agent *agent = &env->agents[agent_id];
  int old_idx = flat_idx(env, agent->r, agent->c);

  int adr = 0;
  bool allocated = false;
  while (!allocated){
    adr = rand() % (env->width * env->height);
    if (is_obstacle(env, adr)){
      continue;
    }
    int r = adr / env->width;
    int c = adr % env->width;
    agent->r = r;
    agent->c = c;
    allocated = true;
  }
  assert(env->pids[adr] == -1);
  env->pids[old_idx] = -1;
  env->pids[adr] = agent->id;
}
void c_reset(PredPrey *env) {
  
  env->tick = 0;
  env->last_agent_dead_tick = 0;

  memset(env->agent_logs, 0, env->num_agents * sizeof(Log));
  env->log = (Log){0};

  memset(env->items, EMPTY, env->width * env->height * sizeof(unsigned char));
  for (int r = 0; r < env->height; r++){
    for (int c = 0; c < env->width; c++){
      int grid_idx = flat_idx(env, r, c);
      env->pids[grid_idx] = -1;
    }
  }

  for (int i = 0; i < env->num_agents; i++) {
    spawn_agent(env, i);
  }

  env->food_count = 0;
  env->wood_count = 0;
  init_items(env);

  memset(env->observations, 0, env->num_agents * env->obs_size * sizeof(float));
  memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
  memset(env->masks, 1, env->num_agents * sizeof(unsigned char));

  compute_observations(env);
}

typedef void (*InteractFn)(PredPrey*, int);

void interact_nothing(PredPrey* env, int agent_id){
  return;
};
void interact_food(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  int curr_grid_idx = flat_idx(env, agent->r, agent->c);
  if (agent->food_amt >= MAX_INVENTORY_ITEM) {
    return;
  }
  // Pick up food
  agent->food_amt += 1;
  env->items[curr_grid_idx] = EMPTY;
  env->food_count -= 1;
  env->agent_logs[agent_id].collects += 1;
  agent->anim = ANIM_INTERACT;
  reward_agent(env, agent_id, env->reward_collect);
};

void interact_wood(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  int curr_grid_idx = flat_idx(env, agent->r, agent->c);
  if (agent->wood_amt >= MAX_INVENTORY_ITEM) {
    return;
  }
  // Pick up wood
  agent->wood_amt += 1;
  env->items[curr_grid_idx] = EMPTY;
  env->wood_count -= 1;
  env->agent_logs[agent_id].collects += 1;
  agent->anim = ANIM_INTERACT;
  reward_agent(env, agent_id, env->reward_collect);
};

void interact_chest(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  // If agent has food, deposit all in chest
  if (agent->food_amt > 0) {
    env->chest_food_amt = fmin(
      env->chest_food_amt + agent->food_amt, 
      MAX_CHEST_CAPACITY
    );
    agent->food_amt = 0;
    reward_agent(env, agent_id, env->reward_store_chest);
  } else {
    // If agent has no food, withdraw one from chest 
    if (env->chest_food_amt <= 0) {
      return;
    }
    agent->food_amt += 1;
    env->chest_food_amt -= 1;
  }
  agent->anim = ANIM_INTERACT;
  return;
};

int facing_delta_row[4] = { 1, -1, 0, 0 };
int facing_delta_col[4] = { 0, 0, 1, -1 };
void interact_agent(PredPrey* env, int agent_id){
  // Remove stealing ability for now
  return;
  // Agent* agent = &env->agents[agent_id];
  // int dr = facing_delta_row[agent->direction];
  // int dc = facing_delta_col[agent->direction];
  // int facing_tile_idx = flat_idx(env, agent->r + dr, agent->c + dc);
  
  // int facing_agent = env->pids[facing_tile_idx];
  // if (facing_agent != -1) {
  //   Agent *other_agent = &env->agents[facing_agent];
  //   // Steal food from other agent
  //   if (other_agent->food_amt > 0) {
  //     agent->food_amt += 1;
  //     other_agent->food_amt -= 1;
  //     env->agent_logs[agent_id].steals += 1;
  //     agent->anim = ANIM_INTERACT;
  //     reward_agent(env, agent_id, env->reward_steal);
  //   }
  // } 

};

void interact_fireplace(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  // If agent has wood, light the fireplace
  if (agent->wood_amt <= 0) {
    return;
  }
  agent->wood_amt -= 1;
  env->is_fireplace_lit = true;
  env->fire_time_remaining += MAX_FIRE_TIME;
  agent->anim = ANIM_INTERACT;
  reward_agent(env, agent_id, env->reward_fireplace_lit);
};

InteractFn interaction_fn[] = {
    [EMPTY] = interact_nothing,
    [ITEM_FOOD] = interact_food,
    [ITEM_WOOD] = interact_wood,
    [ITEM_CHEST] = interact_chest,
    [ITEM_FIREPLACE] = interact_fireplace,
};

void handle_eat(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  if (agent->food_amt <= 0){
    return;
  }
  agent->food_amt -= 1;
  add_hp(env, agent_id, HP_REWARD_FOOD);
  agent->anim = ANIM_EAT;
};

void update_coldness(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  int pos_idx = flat_idx(env, agent->r, agent->c);
  bool protected = (env->terrain[pos_idx] == TILE_FLOOR_WOOD && env->is_fireplace_lit);

  if (!protected){
    agent->coldness++;
  }

  if (agent->coldness >= MAX_COLDNESS){
    remove_hp(env, agent->id, HP_LOSS_COLD);
    return;
  }

  if (protected){
    agent->coldness = fmax(0, agent->coldness - COLDNESS_LOSS_PER_HOUR);
  }
};

void move_agent(PredPrey* env, Agent* agent, int nr, int nc){
  int curr_idx = flat_idx(env, agent->r, agent->c);
  int new_idx = flat_idx(env, nr, nc);

  env->pids[curr_idx] = -1;
  env->pids[new_idx] = agent->id;

  agent->r = nr;
  agent->c = nc;
};

bool is_valid_move(PredPrey* env, int nr, int nc){
  if (nr < 0 || nr >= env->height || nc < 0 || nc >= env->width){
    return false;
  }
  int new_idx = flat_idx(env, nr, nc);
  if (is_obstacle(env, new_idx)){
    return false;
  }
  return true;
}

int delta_row[7] = { 1, -1, 0, 0, 0, 0, 0 };
int delta_col[7] = { 0, 0, 1, -1, 0, 0, 0 };
void update_movement(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  int action = env->actions[agent_id];

  int dr = delta_row[action];
  int dc = delta_col[action];

  int nr = agent->r + dr;
  int nc = agent->c + dc;

  if (dr == 0 && dc == 0){
    return;
  }

  agent->direction = action;

  if (!is_valid_move(env, nr, nc))
      return;

  move_agent(env, agent, nr, nc);
}

void apply_base_rewards(PredPrey* env, int agent_id){
  Agent* agent = &env->agents[agent_id];
  reward_agent(env, agent_id, env->timestep_reward);
  float reward_hp = (agent->hp / (float)MAX_HP) * env->hp_reward_scale;
  reward_agent(env, agent_id, reward_hp);
  float reward_food = (agent->food_amt / (float)MAX_INVENTORY_ITEM) * env->held_food_reward_scale;
  reward_agent(env, agent_id, reward_food);
}

void step_agent(PredPrey *env, int i) {
  Agent *a = &env->agents[i];
  a->anim = ANIM_IDLE;

  apply_base_rewards(env, i);
  update_movement(env, i);

  if (env->actions[i] == INTERACT){
    int tile_idx = flat_idx(env, a->r, a->c);
    interaction_fn[env->items[tile_idx]](env, i);
  }

  if (env->actions[i] == EAT){
    handle_eat(env, i);
  }

  update_coldness(env, i);
}

void c_step(PredPrey *env) {
  env->tick++;

  memset(env->rewards, 0, env->num_agents * sizeof(float));
  memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));

  if (env->is_fireplace_lit) {
    env->fire_time_remaining -= 1;
    if (env->fire_time_remaining <= 0) {
      env->is_fireplace_lit = false;
    }
  }
  for (int i = 0; i < env->num_agents; i++) {
    step_agent(env, i);

    if (env->tick % TICK_PER_HOUR == 0) {
      // Hourly HP decay
      remove_hp(env, i, HP_LOSS_PER_HOUR);
    }
    
    // If agent survived long enough, reward and reset agent. 
    if ((env->tick - env->agents[i].start_tick) >= MAX_TIMESTEPS && env->agents[i].hp > 0) {
      remove_agent(env, i);
      env->terminals[i] = 1;
      reward_agent(env, i, env->reward_death_scale);
      add_agent_log(env, i);
      spawn_agent(env, i);
      continue;
    }

    // Immediate respawn if died
    if (env->agents[i].hp <= 0) {
      spawn_agent(env, i);
      continue;
    } 
  }

  spawn_items(env);
  compute_observations(env);
}
//////////////////////////////////////////////////////////////////
// Sprites & Animations
//////////////////////////////////////////////////////////////////
typedef struct SpriteInfo SpriteInfo;
struct SpriteInfo {
  int src_x, src_y; // top-left position in sprite sheet
  int width, height; // dimensions of the sprite
  int offset_x, offset_y; // offset to add to the env position when rendering
};
SpriteInfo SPRITES_TILES[4] = {
    (SpriteInfo){ // TILE_SOIL
        .src_x = 0,
        .src_y = 0,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // TILE_FLOOR_WOOD
        .src_x = 0,
        .src_y = SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // TILE_WATER
        .src_x = 0,
        .src_y = 2 * SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // TILE_GRASS
        .src_x = 0,
        .src_y = 3 * SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
};
SpriteInfo SPRITE_ITEMS[6] = {
    (SpriteInfo){ // EMPTY
        .src_x = 0,
        .src_y = 0,
        .width = 0,
        .height = 0,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // ITEM_WOOD
        .src_x = 0,
        .src_y = SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // ITEM_FOOD
        .src_x = 0,
        .src_y = 2 * SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // ITEM_CHEST
        .src_x = 0,
        .src_y = 3 * SPRITE_SIZE,
        .width = SPRITE_SIZE,
        .height = SPRITE_SIZE,
        .offset_x = 0,
        .offset_y = 0,
    },
    (SpriteInfo){ // ITEM_FIREPLACE_LIT
        .src_x = 0,
        .src_y = 4 * SPRITE_SIZE,
        .width = 2 * SPRITE_SIZE,
        .height = 3 * SPRITE_SIZE,
        .offset_x = - (2 * SPRITE_SIZE) / 2 + 32,
        .offset_y = - 2 * SPRITE_SIZE,
    },
    (SpriteInfo){ // ITEM_FIREPLACE
        .src_x = 0,
        .src_y = 7 * SPRITE_SIZE,
        .width = 2 * SPRITE_SIZE,
        .height = 3 * SPRITE_SIZE,
        .offset_x = - (2 * SPRITE_SIZE) / 2 + 32,
        .offset_y = - 2 * SPRITE_SIZE,
    },
};
// Simplified animations to just be poses (1 frame only)
typedef struct Animation Animation;
struct Animation {
    int x; // x position in sprite sheet 
};

Animation ANIMATIONS[3] = {
    (Animation){ // ANIM_IDLE
      .x = 0,
    },
    (Animation){ // ANIM_INTERACT
        .x = 5
    },
    (Animation){ // ANIM_EAT
      .x = 3,
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

  InitWindow(width * TILE_SIZE_ENV, height * TILE_SIZE_ENV, "Predator Prey");
  SetTargetFPS(10);

  for (int i = 0; i < 10; i++) {
    renderer->agents[0][i] = LoadTexture(TextFormat("resources/nmmo3/neutral_%d.png", i));
    renderer->agents[1][i] = LoadTexture(TextFormat("resources/nmmo3/fire_%d.png", i));
    renderer->agents[2][i] = LoadTexture(TextFormat("resources/nmmo3/water_%d.png", i));
    renderer->agents[3][i] = LoadTexture(TextFormat("resources/nmmo3/earth_%d.png", i));
    renderer->agents[4][i] = LoadTexture(TextFormat("resources/nmmo3/air_%d.png", i));
  }
  renderer->tiles = LoadTexture("resources/harvest/tiles.png");
  renderer->items = LoadTexture("resources/harvest/items.png");
  renderer->font = LoadFont("resources/nmmo3/ManaSeedBody.ttf");

  return renderer;
}

void close_renderer(Renderer *renderer) {
  UnloadTexture(renderer->tiles);
  UnloadTexture(renderer->items);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j <10; j++) {
      UnloadTexture(renderer->agents[i][j]);
    }
  }
  UnloadFont(renderer->font);
  CloseWindow();
  free(renderer);
}

   
void draw_ui_bar(int bar_x, int bar_y, int health, int max_health, Color bar_color) {
    DrawRectangle(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, RED);
    DrawRectangle(bar_x, bar_y,
        HEALTH_BAR_WIDTH * health / max_health,
        HEALTH_BAR_HEIGHT, bar_color);
    DrawRectangleLines(bar_x, bar_y, HEALTH_BAR_WIDTH,
        HEALTH_BAR_HEIGHT, BLACK);
}

void draw_time_info(PredPrey *env, Renderer *renderer) {
    int hour = hour_of_day(env);
    int day = day_of_month(env);
    int month = month_of_year(env);

    char time_text[64];
    snprintf(time_text, sizeof(time_text), "Time: %02d:00 Day: %02d Month: %02d", hour, day, month);

    DrawTextEx(renderer->font, time_text, (Vector2){10, 10}, 50, 1, BLACK);
}

void draw_chest_info(PredPrey *env, Renderer *renderer) {
    char chest_text[64];
    snprintf(chest_text, sizeof(chest_text), "Chest Food: %d / %d", env->chest_food_amt, MAX_CHEST_CAPACITY);

    DrawTextEx(renderer->font, chest_text, (Vector2){10, 70}, 30, 1, BLACK);
}
void draw_fireplace_info(PredPrey *env, Renderer *renderer) {
    char fireplace_text[64];
    if (env->is_fireplace_lit) {
        snprintf(fireplace_text, sizeof(fireplace_text), "Fireplace is lit! Time remaining: %d", env->fire_time_remaining);
    } else {
        snprintf(fireplace_text, sizeof(fireplace_text), "Fireplace is not lit.");
    }

    DrawTextEx(renderer->font, fireplace_text, (Vector2){10, 100}, 30, 1, BLACK);
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

  // Draw terrain
  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int adr = flat_idx(env, r, c);

      int terrain_type = env->terrain[adr];
      SpriteInfo terrain_info = SPRITES_TILES[terrain_type];
      Vector2 pos = {
          .x = c * TILE_SIZE_ENV + terrain_info.offset_x,
          .y = r * TILE_SIZE_ENV + terrain_info.offset_y,
      };

      Rectangle source_rect = {
        .x = terrain_info.src_x,
        .y = terrain_info.src_y,
        .width = terrain_info.width,
        .height = terrain_info.height
      };
      DrawTextureRec(renderer->tiles, source_rect, pos, WHITE);
    }
  }

  // Draw items
  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int adr = flat_idx(env, r, c);
      int item_type = env->items[adr];

      // TODO: change data structure to avoid this ugly
      if (item_type == ITEM_FIREPLACE && env->is_fireplace_lit) {
        item_type = ITEM_FIREPLACE_LIT;
      }
      
      SpriteInfo item_info = SPRITE_ITEMS[item_type];
      Vector2 pos = {
          .x = c * TILE_SIZE_ENV + item_info.offset_x,
          .y = r * TILE_SIZE_ENV + item_info.offset_y,
      };

      if (item_type != EMPTY) {
        Rectangle source_rect = {
          .x = item_info.src_x,
          .y = item_info.src_y,
          .width = item_info.width,
          .height = item_info.height
        };
        DrawTextureRec(renderer->items, source_rect, pos, WHITE);
      }
    }
  }

  // Draw entities
  for (int r = 0; r < env->height; r++) {
    for (int c = 0; c < env->width; c++) {
      int adr = flat_idx(env, r, c);
      int entity_id = env->pids[adr];

      Vector2 pos = {
          .x = c * TILE_SIZE_ENV,
          .y = r * TILE_SIZE_ENV,
      };
      if (entity_id != -1) {
        Agent *agent = &env->agents[entity_id];
        Animation animation = ANIMATIONS[agent->anim];
        int starting_sprite_y = (env->agents[entity_id].direction) * SPRITE_SIZE_ENTITY;
        int x_pos = (c - 0.5f)*TILE_SIZE_ENV;
        int y_pos = (r - 0.5f)*TILE_SIZE_ENV;
        Vector2 pos = {
            .x = x_pos,
            .y = y_pos,
        };        
        Rectangle source_rect = {
          .x = animation.x * SPRITE_SIZE_ENTITY,
          .y = starting_sprite_y,
          .width = SPRITE_SIZE_ENTITY,
          .height = SPRITE_SIZE_ENTITY
        };
        DrawTextureRec(renderer->agents[entity_id%5][entity_id%10], source_rect, pos, WHITE);

        // Draw HP bar 
        int bar_x = x_pos + TILE_SIZE_ENV - HEALTH_BAR_WIDTH/2;
        int bar_y = y_pos;
        draw_ui_bar(bar_x, bar_y, agent->hp, MAX_HP, GREEN);
        // Draw Coldness bar
        draw_ui_bar(bar_x, bar_y + 10, MAX_COLDNESS - agent->coldness, MAX_COLDNESS, BLUE);

        // Food Number in inventory
        char* txt;
        Color color;
        txt = (char*) TextFormat("%d: F: %d | W: %d", entity_id, agent->food_amt, agent->wood_amt);
        color = GREEN;

        Vector2 text_pos = {.x = bar_x, .y = bar_y - 20};
        DrawTextEx(renderer->font, txt, text_pos, 21, 1, color);

      }
    }
  }
  draw_time_info(env, renderer);
  draw_chest_info(env, renderer);
  draw_fireplace_info(env, renderer);
  EndDrawing();
}
