#include "convert_circle.h"
#include "puffernet.h"
#include <stdlib.h>
#include <time.h>

int main() {
  ConvertCircle env = {
      .width = 1920,
      .height = 1080,
      .num_agents = 128,
      .num_factories = 16,
      .num_resources = 6,
      .equidistant = 1,
      .radius = 400,
  };
  srand(time(NULL));
  init(&env);

  int num_obs = 2 * env.num_resources + 4 + env.num_resources;
  env.observations = calloc(env.num_agents * num_obs, sizeof(float));
  env.actions = calloc(2 * env.num_agents, sizeof(int));
  env.rewards = calloc(env.num_agents, sizeof(float));
  env.terminals = calloc(env.num_agents, sizeof(unsigned char));

  Weights *weights =
      load_weights("resources/convert/convert_weights.bin", 137743);
  int logit_sizes[2] = {9, 5};
  LinearLSTM *net =
      make_linearlstm(weights, env.num_agents, num_obs, logit_sizes, 2);

  c_reset(&env);
  c_render(&env);

  while (!WindowShouldClose()) {
    for (int i = 0; i < env.num_agents; i++) {
      env.actions[2 * i] = rand() % 9;
      env.actions[2 * i + 1] = rand() % 5;
    }

    forward_linearlstm(net, env.observations, env.actions);
    compute_observations(&env);
    c_step(&env);
    c_render(&env);
  }

  free_linearlstm(net);
  free(env.observations);
  free(env.actions);
  free(env.rewards);
  free(env.terminals);
  c_close(&env);
}
