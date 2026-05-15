#include "cubed.h"

int main() {
  Cubed env = {.size = 11};
  env.observations = (unsigned char *)calloc(env.size * env.size * env.size,
                                             sizeof(unsigned char));
  env.actions = (int *)calloc(1, sizeof(int));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (unsigned char *)calloc(1, sizeof(unsigned char));

  c_reset(&env);
  c_render(&env);

  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.actions[0] = 0;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.actions[0] = UP;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.actions[0] = DOWN;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.actions[0] = LEFT;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.actions[0] = RIGHT;
      if (IsKeyDown(KEY_Q))
        env.actions[0] = FRONT;
      if (IsKeyDown(KEY_E))
        env.actions[0] = BACK;
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