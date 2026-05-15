#include "bitflip.h"

int main() {
  BitFlip env = {.size = 5};
  env.observations =
      (unsigned char *)calloc(env.size * 3, sizeof(unsigned char));
  env.actions = (int *)calloc(1, sizeof(int));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (unsigned char *)calloc(1, sizeof(unsigned char));

  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.actions[0] = NOOP;
      if (IsKeyDown(KEY_LEFT))
        env.actions[0] = LEFT;
      if (IsKeyDown(KEY_RIGHT))
        env.actions[0] = RIGHT;
      if (IsKeyDown(KEY_UP))
        env.actions[0] = FLIP;
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
