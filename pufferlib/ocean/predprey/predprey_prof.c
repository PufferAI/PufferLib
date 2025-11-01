#include <raylib.h>
#include <unistd.h>
#include <stdlib.h>
#include "predprey.h"

int main() {
  PredPrey env = {
      .num_agents = 4,
      .width = 32,
      .height = 32,
      .vision = 3,
      .reward_food = 0.0f,
      .food_base_spawn_rate = 1e-1,
  };
  allocate_cenv(&env);
  c_reset(&env);

  long i = 0;
  while (true) { 

    for (int i = 0; i < env.num_agents; i++) {
        env.actions[i] = rand() % 7; 
    }
    
    c_step(&env);

    i++;
    if (i > 10000000) {
      printf("breaking");
      break;
    }
  }
  free_CEnv(&env);

  return 0;
}
