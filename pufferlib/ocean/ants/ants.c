// for local testing of c code,build with: 
// bash scripts/build_ocean.sh ants local

#include <time.h>
#include "ants.h"
#include "puffernet.h"

// Function to visualize ant 1's observations
void render_ant_observations(AntsEnv* env, int ant_id) {
    if (ant_id >= env->num_ants) return;
    
    // Get ant 1's observations
    float* obs = &env->observations[ant_id * env->obs_size];
    Ant* ant = &env->ants[ant_id];
    
    // Define UI panel position
    int panel_x = 20;
    int panel_y = 100;
    int panel_width = 300;
    int panel_height = 200;
    
    // Draw semi-transparent background panel
    DrawRectangle(panel_x - 10, panel_y - 10, panel_width + 20, panel_height + 20, 
                  (Color){0, 0, 0, 180});
    DrawRectangleLines(panel_x - 10, panel_y - 10, panel_width + 20, panel_height + 20, RAYWHITE);
    
    // Title
    DrawText(TextFormat("ANT %d OBSERVATIONS", ant_id), panel_x, panel_y, 16, YELLOW);
    
    int y_offset = panel_y + 25;
    int line_height = 18;
    
    // Display each observation with description
    DrawText(TextFormat("Position X: %.3f", obs[0]), panel_x, y_offset, 14, RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Position Y: %.3f", obs[1]), panel_x, y_offset, 14, RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Direction: %.3f", obs[2]), panel_x, y_offset, 14, RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Has Food: %s", obs[3] > 0.5f ? "YES" : "NO"), 
             panel_x, y_offset, 14, obs[3] > 0.5f ? GREEN : RED);
    y_offset += line_height;
    
    DrawText(TextFormat("Colony Dir: %.3f", obs[4]), panel_x, y_offset, 14, RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Colony Dist: %.3f", obs[5]), panel_x, y_offset, 14, RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Food Dir: %.3f", obs[6]), panel_x, y_offset, 14, 
             obs[6] < 0 ? GRAY : RAYWHITE);
    y_offset += line_height;
    
    DrawText(TextFormat("Food Dist: %.3f", obs[7]), panel_x, y_offset, 14, 
             obs[7] < 0 ? GRAY : RAYWHITE);
    
    // Visual indicators on the ant
    Vector2D ant_pos = ant->position;
    
    // Highlight the selected ant
    DrawCircleLines(ant_pos.x, ant_pos.y, ANT_SIZE + 3, YELLOW);
    DrawCircleLines(ant_pos.x, ant_pos.y, ANT_SIZE + 5, YELLOW);
    
    // Draw direction to colony (if valid)
    if (obs[4] >= 0) {
        float colony_angle = (obs[4] * 2 * M_PI) - M_PI;
        float line_length = 40.0f;
        Vector2D colony_end = {
            ant_pos.x + line_length * cos(colony_angle),
            ant_pos.y + line_length * sin(colony_angle)
        };
        DrawLineEx((Vector2){ant_pos.x, ant_pos.y}, (Vector2){colony_end.x, colony_end.y}, 3, BLUE);
        DrawText("COLONY", colony_end.x + 5, colony_end.y - 10, 12, BLUE);
    }
    
    // Draw direction to food (if visible)
    if (obs[6] >= 0) {
        float food_angle = (obs[6] * 2 * M_PI) - M_PI;
        float line_length = 30.0f;
        Vector2D food_end = {
            ant_pos.x + line_length * cos(food_angle),
            ant_pos.y + line_length * sin(food_angle)
        };
        DrawLineEx((Vector2){ant_pos.x, ant_pos.y}, (Vector2){food_end.x, food_end.y}, 2, GREEN);
        DrawText("FOOD", food_end.x + 5, food_end.y - 10, 12, GREEN);
    }
    
    // Draw current direction
    float current_angle = (obs[2] * 2 * M_PI) - M_PI;
    float dir_length = 25.0f;
    Vector2D dir_end = {
        ant_pos.x + dir_length * cos(current_angle),
        ant_pos.y + dir_length * sin(current_angle)
    };
    DrawLineEx((Vector2){ant_pos.x, ant_pos.y}, (Vector2){dir_end.x, dir_end.y}, 4, YELLOW);
}

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
    
    // Track key states for single-press detection
    bool left_pressed = false;
    bool right_pressed = false;
    bool space_pressed = false;
    
    // Main loop - FOLLOWING SNAKE PATTERN
    while (!WindowShouldClose()) {
        // User can take control with shift key
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Control first ant of colony 1 for demo
            env.actions[0] = ACTION_MOVE_FORWARD;
            
            // Handle left turn
            if ((IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) && !left_pressed) {
                env.actions[0] = ACTION_TURN_LEFT;
                left_pressed = true;
            } else if (!IsKeyDown(KEY_LEFT) && !IsKeyDown(KEY_A)) {
                left_pressed = false;
            }
            
            // Handle right turn
            if ((IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) && !right_pressed) {
                env.actions[0] = ACTION_TURN_RIGHT;
                right_pressed = true;
            } else if (!IsKeyDown(KEY_RIGHT) && !IsKeyDown(KEY_D)) {
                right_pressed = false;
            }
            
            // Handle pheromone drop
            if (IsKeyDown(KEY_SPACE) && !space_pressed) {
                env.actions[0] = ACTION_DROP_PHEROMONE;
                space_pressed = true;
            } else if (!IsKeyDown(KEY_SPACE)) {
                space_pressed = false;
            }
            
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
        
        // Visualize ant observations when shift is pressed
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            render_ant_observations(&env, 0);
        }
        
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
    printf("- Hold SHIFT to control the first ant AND view ant 1's observations\n");
    printf("- A/D or LEFT/RIGHT to turn\n");
    printf("- SPACE to drop pheromone\n");
    printf("- ESC to exit\n\n");
    
    demo();
    
    // Uncomment for performance testing
    // printf("\nRunning performance test...\n");
    // test_performance(10);
    
    return 0;
}