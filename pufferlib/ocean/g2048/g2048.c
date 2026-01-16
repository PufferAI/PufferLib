#include "g2048.h"
#include "g2048_net.h"

#define OBS_DIM 289
#define HIDDEN_DIM 512

// Set NO_RENDER to true to run evals without the render
#define NO_RENDER false
#define NUM_EVAL_RUNS 200

int main() {
    srand(time(NULL));
    Game env = {
        .can_go_over_65536 = true,
        .reward_scaler = 0.0,
        .endgame_env_prob = 0.0,
        .scaffolding_ratio = 0.0,
        .use_heuristic_rewards = false,
        .snake_reward_weight = 0.0,
    };
    init(&env);

    unsigned char observations[OBS_DIM] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    Weights* weights = load_weights("resources/g2048/g2048_weights.bin", 3713541);
    G2048Net* net = make_g2048net(weights, OBS_DIM, HIDDEN_DIM);
    c_reset(&env);
    if (!NO_RENDER) c_render(&env);
    printf("Starting...\n");
    
    clock_t start_time = clock();

    // Main game loop
    int trial = 1;
    int frame = 0;
    int action = -1;
    while (NO_RENDER || !WindowShouldClose()) {
        if (!NO_RENDER) c_render(&env); // Render at the start of the loop
        frame++;
        
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            action = -1;
            if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) action = UP;
            else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) action = DOWN;
            else if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) action = LEFT;
            else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) action = RIGHT;
            env.actions[0] = action - 1;
        } else if (frame % 1 != 0) {
            continue;
        } else {
            action = 1;
            forward_g2048net(net, env.observations, env.actions);
        }

        if (action > 0) {
            step_without_reset(&env);
        }

        if (env.terminals[0] == 1) { 
            clock_t end_time = clock();
            double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Trial: %d, Ticks: %d, Max Tile: %d, Merge Score: %d, Time: %.2fs\n",
                trial++, env.tick, 1 << env.max_tile, env.score, time_taken);
            
            if (!NO_RENDER) {
                // Reached the 65536 tile, so full stop. Savor the moment!
                if (env.max_tile >= 16) WaitTime(100000);
                WaitTime(10);
            }

            c_reset(&env);
            if (!NO_RENDER) c_render(&env);
            start_time = clock();
            frame = 0;
        }

        if (!NO_RENDER && IsKeyDown(KEY_LEFT_SHIFT) && action > 0) {
            // Don't need to be super reactive
            WaitTime(0.1);
        }

        if (NO_RENDER && trial > NUM_EVAL_RUNS) break;
    }

    free_g2048net(net);
    free(weights);
    c_close(&env);
    printf("Finished %d trials.\n", NUM_EVAL_RUNS);
    return 0;
}
