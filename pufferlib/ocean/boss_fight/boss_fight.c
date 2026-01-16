#include "boss_fight.h"
#include "raylib.h"

int main() {
  int num_obs = 13;
  int num_actions = 1;
  int num_agents = 1;

  BossFight env = {};
  env.observations = (float *)calloc(num_obs, sizeof(unsigned char));
  env.actions = (float *)calloc(num_actions, sizeof(int));
  env.rewards = (float *)calloc(num_agents, sizeof(float));
  env.terminals = (unsigned char *)calloc(num_agents, sizeof(unsigned char));

  // Always call reset and render first
  c_reset(&env);
  c_render(&env);

  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      if (IsKeyDown(KEY_W))
        env.actions[0] = 1;
      else if (IsKeyDown(KEY_S))
        env.actions[0] = 2;
      else if (IsKeyDown(KEY_A))
        env.actions[0] = 3;
      else if (IsKeyDown(KEY_D))
        env.actions[0] = 4;
      else if (IsKeyDown(KEY_SPACE))
        env.actions[0] = 5;
      else if (IsKeyDown(KEY_J))
        env.actions[0] = 6;
      else
        env.actions[0] = 0;
    } else {
      env.actions[0] = rand() % 7;
    }
    c_step(&env);
    c_render(&env);
  }
  free(env.observations);
  free(env.actions);
  free(env.rewards);
  free(env.terminals);
  c_close(&env);
}
