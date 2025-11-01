#include <raylib.h>
#include <unistd.h>
#include <stdlib.h>
// #include "puffernet.h"
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
  c_render(&env);

  // Weights* weights = load_weights("resources/cpr/cpr_weights.bin", 139270);
  // int logit_sizes[] = {5};
  // LinearLSTM* net = make_linearlstm(weights, env.num_agents, 49, logit_sizes, 1);
  while (!WindowShouldClose()) {

    for (int i = 0; i < env.num_agents; i++) {
        env.actions[i] = rand() % 7; 
    }
    
    // User can take control of the first puffer
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      sleep(1);
      env.actions[0] = NO_MOVE;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.actions[0] = UP;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.actions[0] = DOWN;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.actions[0] = LEFT;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.actions[0] = RIGHT;
      if (IsKeyDown(KEY_C))
        env.actions[0] = INTERACT;
      if (IsKeyDown(KEY_E))
        env.actions[0] = EAT;

      printf("Getting user input %d\n", env.actions[0]);
    } else {
        // for (int i = 0; i < env.num_agents*49; i++) {
        //     net->obs[i] = env.observations[i];
        // }
        // forward_linearlstm(net, net->obs, env.actions);
    }

    c_step(&env);
    c_render(&env);

  }
  free_CEnv(&env);

  return 0;
}


////////////////
// For profile
////////////////
// #include <raylib.h>
// #include <unistd.h>
// #include <stdlib.h>
// #include "predprey.h"

// int main() {
//   PredPrey env = {
//       .num_agents = 4,
//       .width = 32,
//       .height = 32,
//       .vision = 3,
//       .reward_food = 0.0f,
//       .food_base_spawn_rate = 1e-1,
//   };
//   allocate_cenv(&env);
//   c_reset(&env);

//   long i = 0;
//   while (true) { 

//     for (int i = 0; i < env.num_agents; i++) {
//         env.actions[i] = rand() % 7; 
//     }
    
//     c_step(&env);

//     i++;
//     if (i > 10000000) {
//       printf("breaking");
//       break;
//     }
//   }
//   free_CEnv(&env);

//   return 0;
// }
