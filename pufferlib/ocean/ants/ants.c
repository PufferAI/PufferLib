// for local testing of c code,build with: 
// bash scripts/build_ocean.sh ants local

#include <time.h>
#include "ants.h"
#include "puffernet.h"

int demo() {
    // Initialize environment with proper parameters - FOLLOWING SNAKE PATTERN
    AntsEnv env = {
        .num_ants = 32,
        .width = WINDOW_WIDTH,
        .height = WINDOW_HEIGHT,
        .reward_food = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death = -1.0f,
        .cell_size = 1,
    };
    
    // Allocate memory - CRITICAL: USING PROPER ALLOCATION PATTERN
    allocate_ants_env(&env);
    c_reset(&env);

    // Load trained weights if available
    Weights* weights = NULL;
    LinearLSTM* net = NULL;
    FILE* f = fopen("resources/ants_weights.bin", "rb");
    if (f) {
        fclose(f);
        weights = load_weights("resources/ants_weights.bin", 266501);
        if (weights) {
            int logit_sizes[1] = {4};
            net = make_linearlstm(weights, env.num_ants, env.obs_size, logit_sizes, 4);
        }
    }
    
    printf("Environment initialized. Starting render loop...\n");
    printf("Ants: %d, Observation size: %d\n", env.num_ants, env.obs_size);
    if (!net) {
        printf("No trained weights found. Running with random actions.\n");
    }
    
    // Initialize rendering client
    env.client = make_client(1, env.width, env.height);
    
    // Main loop - FOLLOWING SNAKE PATTERN
    while (!WindowShouldClose()) {
        // User can take control with shift key
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Control first ant of colony 1 for demo
            env.actions[0] = ACTION_MOVE_FORWARD;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACTION_TURN_LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_TURN_RIGHT;
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = ACTION_DROP_PHEROMONE;
            
            // Rest of ants act via scripted behaviors
            for (int i = 1; i < env.num_ants; i++) {
                Ant* ant = &env.ants[i];
                if (ant->has_food) {
                    // If ant has food, return to colony
                    Colony* colony = &env.colonies[ant->colony_id];
                    float angle_to_colony = get_angle(ant->position, colony->position);
                    float angle_diff = wrap_angle(angle_to_colony - ant->direction);
                    
                    // Turn towards colony
                    if (angle_diff > 0.1) {
                        env.actions[i] = ACTION_TURN_RIGHT;
                    } else if (angle_diff < -0.1) {
                        env.actions[i] = ACTION_TURN_LEFT;
                    } else {
                        env.actions[i] = ACTION_MOVE_FORWARD;
                    }
                } else {
                    // If ant doesn't have food, seek nearest food source
                    float closest_food_dist_sq = env.width * env.width;
                    Vector2D closest_food_pos = {0, 0};
                    bool found_food = false;
                    
                    for (int j = 0; j < env.num_food_sources; j++) {
                        if (env.food_sources[j].amount > 0) {
                            float dist_sq = distance_squared(ant->position, env.food_sources[j].position);
                            if (dist_sq < closest_food_dist_sq && is_in_vision(ant->position, env.food_sources[j].position)) {
                                closest_food_dist_sq = dist_sq;
                                closest_food_pos = env.food_sources[j].position;
                                found_food = true;
                            }
                        }
                    }
                    
                    if (found_food) {
                        // Turn towards food
                        float angle_to_food = get_angle(ant->position, closest_food_pos);
                        float angle_diff = wrap_angle(angle_to_food - ant->direction);
                        
                        if (angle_diff > 0.1) {
                            env.actions[i] = ACTION_TURN_RIGHT;
                        } else if (angle_diff < -0.1) {
                            env.actions[i] = ACTION_TURN_LEFT;
                        } else {
                            env.actions[i] = ACTION_MOVE_FORWARD;
                        }
                    } else {
                        // If no food in sight, move forward and occasionally turn
                        env.actions[i] = (rand() % 100 < 5) ? (rand() % 2 ? ACTION_TURN_LEFT : ACTION_TURN_RIGHT) : ACTION_MOVE_FORWARD;
                    }
                }
            }
        } else if (net) {
            // Use neural network for all ants
            forward_linearlstm(net, env.observations, env.actions);
        } else {
            // All ants act randomly
            for (int i = 0; i < env.num_ants; i++) {
                env.actions[i] = rand() % 4;
            }
        }
        
        c_step(&env);
        c_render(&env);
        
        // Print stats periodically
        if (env.tick % 1000 == 0 && env.log.n > 0) {
            printf("Tick %d: Episodes completed: %.0f, Avg score: %.2f, Avg return: %.2f\n",
                   env.tick, env.log.n, env.log.score / env.log.n, env.log.episode_return / env.log.n);
        }
    }
    
    printf("Closing environment...\n");
    
    // Clean up - PROPER CLEANUP FOLLOWING SNAKE PATTERN
    if (net) {
        free_linearlstm(net);
    }
    if (weights) {
        free(weights);
    }
    close_client(env.client);
    free_ants_env(&env);
    
    return 0;
}

void test_performance(float test_time) {
    // Performance test environment
    AntsEnv env = {
        .num_ants = 2048,
        .width = 1280,
        .height = 720,
        .reward_food = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death = -1.0f,
        .cell_size = 1,
    };
    
    allocate_ants_env(&env);
    c_reset(&env);
    
    int start = time(NULL);
    int steps = 0;
    
    while (time(NULL) - start < test_time) {
        // Random actions for performance test
        for (int i = 0; i < env.num_ants; i++) {
            env.actions[i] = rand() % 4;
        }
        
        c_step(&env);
        steps++;
        
        // Print intermediate stats
        if (steps % 1000 == 0 && env.log.n > 0) {
            printf("Step %d: Episodes: %.0f, Avg performance: %.4f\n",
                   steps, env.log.n, env.log.perf / env.log.n);
        }
    }
    
    int end = time(NULL);
    float sps = (float)env.num_ants * steps / (end - start);
    printf("Ant Colony Environment SPS: %.0f\n", sps);
    printf("Total ant steps: %.0f\n", sps);
    printf("Episodes completed: %.0f\n", env.log.n);
    if (env.log.n > 0) {
        printf("Average score: %.2f\n", env.log.score / env.log.n);
        printf("Average performance: %.4f\n", env.log.perf / env.log.n);
    }
    
    // Clean up
    free_ants_env(&env);
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    printf("Ant Colony Environment Demo\n");
    printf("Controls:\n");
    printf("- Hold SHIFT to control the first ant\n");
    printf("- A/D or LEFT/RIGHT to turn\n");
    printf("- SPACE to drop pheromone\n");
    printf("- ESC to exit\n\n");
    
    demo();
    
    // Uncomment for performance testing
    // printf("\nRunning performance test...\n");
    // test_performance(10);
    
    return 0;
}