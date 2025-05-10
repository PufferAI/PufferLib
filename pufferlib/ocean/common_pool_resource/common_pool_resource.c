#include "cpr.h"
#include <raylib.h>
#include <unistd.h>

int main() {
  int width = 32;
  int height = 32;

  int render_cell_size = 32;

  CCpr env = {
      .num_agents = 8,
      .width = width,
      .height = height,
      .vision = 3,
      .reward_food = 1.0f,
      .interactive_food_reward = 5.0f,
      .food_base_spawn_rate = 2e-3,
  };
  allocate_ccpr(&env);
  c_reset(&env);

  Renderer *renderer = init_renderer(render_cell_size, width, height);
  while (!WindowShouldClose()) {

    c_render(renderer, &env);
    int st = 0;
    // User can take control of the first puffer
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      st = 1;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_Z))
        env.actions[0] = 0;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.actions[0] = 1;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_Q))
        env.actions[0] = 2;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.actions[0] = 3;

      printf("Getting user input %d\n", env.actions[0]);
      sleep(2);
    }
    for (int i = st; i < env.num_agents; i++) {
      env.actions[i] = rand() % 5;
      // printf("Agent %d gets actions %d\n", i, env->actions[i]);
    }
    c_step(&env);
  }
  close_renderer(renderer);
  free_CCpr(&env);

  return 0;
}
