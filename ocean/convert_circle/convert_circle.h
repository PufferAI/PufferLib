/* ConvertCircle: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float n;
} Log;

typedef struct {
  Texture2D sprites;
} Client;

typedef struct {
  float x;
  float y;
  float heading;
  float speed;
  int item;
  int episode_length;
} Agent;

typedef struct {
  float x;
  float y;
  float heading;
  int item;
} Factory;

typedef struct {
  Log log;
  Client *client;
  Agent *agents;
  Factory *factories;
  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int width;
  int height;
  int num_agents;
  int num_factories;
  int num_resources;
  int equidistant;
  int radius;
} ConvertCircle;

static inline float random_float(float low, float high) {
  return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

void init(ConvertCircle *env) {
  env->agents = calloc(env->num_agents, sizeof(Agent));
  env->factories = calloc(env->num_factories, sizeof(Factory));
}

int compare_floats(const void *a, const void *b) {
  return (*(float *)a - *(float *)b) > 0;
}

void compute_observations(ConvertCircle *env) {
  int obs_idx = 0;
  for (int a = 0; a < env->num_agents; a++) {
    Agent *agent = &env->agents[a];
    float dists[env->num_resources];
    for (int i = 0; i < env->num_resources; i++) {
      dists[i] = 999999;
    }
    for (int f = 0; f < env->num_factories; f++) {
      Factory *factory = &env->factories[f];
      float dx = factory->x - agent->x;
      float dy = factory->y - agent->y;
      float dd = dx * dx + dy * dy;
      int type = f % env->num_resources;
      if (dd < dists[type]) {
        dists[type] = dd;
        env->observations[obs_idx + 2 * type] = dx / env->width;
        env->observations[obs_idx + 2 * type + 1] = dy / env->height;
      }
    }
    obs_idx += 2 * env->num_resources;
    env->observations[obs_idx++] = agent->heading / (2 * PI);
    env->observations[obs_idx++] = env->rewards[a];
    env->observations[obs_idx++] = agent->x / env->width;
    env->observations[obs_idx++] = agent->y / env->height;
    memset(&env->observations[obs_idx], 0, env->num_resources * sizeof(float));
    env->observations[obs_idx + agent->item] = 1.0f;
    obs_idx += env->num_resources;
  }
}

void c_reset(ConvertCircle *env) {
  for (int i = 0; i < env->num_agents; i++) {
    env->agents[i].x = env->width / 2.0f + random_float(-10.0f, 10.0f);
    env->agents[i].y = env->height / 2.0f + random_float(-10.0f, 10.0f);
    env->agents[i].item = rand() % env->num_resources;
    env->agents[i].episode_length = 0;
  }
  float angle;
  float delta_angle = 2.0f * PI / env->num_factories;
  for (int i = 0; i < env->num_factories; i++) {
    if (env->equidistant) {
      angle = i * delta_angle;
    } else {
      angle = random_float(0, 2.0f * PI);
    }
    env->factories[i].x = env->width / 2.0f + env->radius * cosf(angle);
    env->factories[i].y = env->height / 2.0f + env->radius * sinf(angle);
    env->factories[i].item = i % env->num_resources;
    env->factories[i].heading = (rand() % 360) * PI / 180.0f;
  }
  compute_observations(env);
}

float clip(float val, float min, float max) {
  if (val < min) {
    return min;
  } else if (val > max) {
    return max;
  }
  return val;
}

void c_step(ConvertCircle *env) {
  for (int i = 0; i < env->num_agents; i++) {
    env->terminals[i] = 0;
    env->rewards[i] = 0;
    Agent *agent = &env->agents[i];
    agent->episode_length += 1;

    agent->heading += ((float)env->actions[2 * i] - 4.0f) / 12.0f;
    agent->heading = clip(agent->heading, 0, 2 * PI);

    agent->speed += 1.0f * ((float)env->actions[2 * i + 1] - 2.0f);
    agent->speed = clip(agent->speed, -20.0f, 20.0f);

    agent->x += agent->speed * cosf(agent->heading);
    agent->x = clip(agent->x, 16, env->width - 16);

    agent->y += agent->speed * sinf(agent->heading);
    agent->y = clip(agent->y, 16, env->height - 16);

    if (rand() % env->num_agents == 0) {
      env->agents[i].x = env->width / 2.0f + random_float(-10.0f, 10.0f);
      env->agents[i].y = env->height / 2.0f + random_float(-10.0f, 10.0f);
    }

    for (int f = 0; f < env->num_factories; f++) {
      Factory *factory = &env->factories[f];
      float dx = (factory->x - agent->x);
      float dy = (factory->y - agent->y);
      float dist = sqrt(dx * dx + dy * dy);
      if (dist > 32) {
        continue;
      }
      if (factory->item == agent->item) {
        agent->item = (agent->item + 1) % env->num_resources;
        env->log.perf += 1.0f;
        env->log.score += 1.0f;
        env->log.episode_length += agent->episode_length;
        env->log.n++;
        env->rewards[i] = 1.0f;
        agent->episode_length = 0;
      }
    }
  }
  for (int f = 0; f < env->num_factories; f++) {
    Factory *factory = &env->factories[f];
    factory->x += 0.0f * cosf(factory->heading);
    factory->y += 0.0f * sinf(factory->heading);

    float factory_x = clip(factory->x, 16, env->width - 16);
    float factory_y = clip(factory->y, 16, env->height - 16);

    if (factory_x != factory->x || factory_y != factory->y) {
      factory->heading = (rand() % 360) * PI / 180.0f;
      factory->x = factory_x;
      factory->y = factory_y;
    }
  }
  compute_observations(env);
}

void c_render(ConvertCircle *env) {
  if (env->client == NULL) {
    InitWindow(env->width, env->height, "PufferLib ConvertCircle");
    SetTargetFPS(30);
    env->client = (Client *)calloc(1, sizeof(Client));
    env->client->sprites = LoadTexture("resources/shared/puffers.png");
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  for (int f = 0; f < env->num_factories; f++) {
    Factory *factory = &env->factories[f];
    DrawTexturePro(env->client->sprites,
                   (Rectangle){
                       64 * factory->item,
                       512,
                       64,
                       64,
                   },
                   (Rectangle){factory->x - 32, factory->y - 32, 64, 64},
                   (Vector2){0, 0}, 0, WHITE);
  }

  for (int i = 0; i < env->num_agents; i++) {
    Agent *agent = &env->agents[i];
    float heading = agent->heading;
    int y = 576;
    if (heading < PI / 2 || heading > 3 * PI / 2) {
      y += 32;
    }
    DrawTexturePro(env->client->sprites,
                   (Rectangle){
                       32 * agent->item,
                       y,
                       32,
                       32,
                   },
                   (Rectangle){agent->x - 16, agent->y - 16, 32, 32},
                   (Vector2){0, 0}, 0, WHITE);
  }

  EndDrawing();
}

void c_close(ConvertCircle *env) {
  free(env->agents);
  free(env->factories);
  if (env->client != NULL) {
    Client *client = env->client;
    UnloadTexture(client->sprites);
    CloseWindow();
    free(client);
  }
}
