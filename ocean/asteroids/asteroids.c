#include "asteroids.h"

int main() {
  Asteroids env = {.size = 500, .frameskip = 1};
  env.observations = (float *)calloc(4 + 2 * 50, sizeof(float));
  env.actions = (int *)calloc(1, sizeof(int));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (unsigned char *)calloc(1, sizeof(unsigned char));

  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
        env.actions[0] = 0;
      } else if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env.actions[0] = 1;
      } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env.actions[0] = 2;
      } else if (IsKeyDown(KEY_SPACE)) {
        env.actions[0] = 3;
      } else {
        env.actions[0] = -1;
      }
    } else {
      env.actions[0] = rand() % 4;
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
