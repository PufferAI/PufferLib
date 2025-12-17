/* Ants: Pure C demo file for testing the environment.
 * Build it with:
 *   bash scripts/build_ocean.sh ants local (debug)
 *   bash scripts/build_ocean.sh ants fast
 *
 * Following the Target env pattern for consistency.
 */
#include <stdio.h>
#include "ants.h"

int main() {
    int num_ants = 64;
    int num_obs = 10;  // Observation space with pheromones and neighbor awareness

    AntsEnv env = {
        .width = 1280,
        .height = 720,
        .num_ants = num_ants,
        .reward_food_pickup = 0.1f,
        .reward_delivery = 10.0f
    };

    init(&env);

    // Allocate buffers manually (normally passed from Python)
    env.observations = calloc(env.num_ants * num_obs, sizeof(float));
    env.actions = calloc(env.num_ants, sizeof(int));
    env.rewards = calloc(env.num_ants, sizeof(float));
    env.terminals = calloc(env.num_ants, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    printf("Ant Colony Demo with Pheromones, Vision & Neighbor Awareness\n");
    printf("=============================================================\n");
    printf("Controls:\n");
    printf("  [V]   - Toggle vision cone visualization\n");
    printf("  [ESC] - Exit\n\n");
    printf("Features:\n");
    printf("  - Very limited vision: 100px range, 30° narrow beam\n");
    printf("  - Neighbor awareness: ants can see nearby colony members\n");
    printf("  - Automatic pheromone trails when carrying food\n");
    printf("  - Pheromone evaporation (1000 step lifetime)\n");
    printf("  - Simple heuristic AI: seek food -> return to colony\n\n");

    // Main loop - exit with ESC or close window
    while (!WindowShouldClose()) {
        // Simple demo AI: seek food when empty, return when full
        for (int i = 0; i < env.num_ants; i++) {
            Ant* ant = &env.ants[i];

            // Simple heuristic AI
            if (ant->has_food) {
                // Return to colony
                Colony* colony = &env.colonies[ant->colony_id];
                float angle_to_colony = get_angle(ant->position, colony->position);
                float angle_diff = wrap_angle(angle_to_colony - ant->direction);

                if (angle_diff > M_PI / 8) {
                    env.actions[i] = ACTION_TURN_RIGHT;
                } else if (angle_diff < -M_PI / 8) {
                    env.actions[i] = ACTION_TURN_LEFT;
                } else {
                    env.actions[i] = ACTION_MOVE_FORWARD;
                }
            } else {
                // Seek nearest food
                float closest_dist_sq = env.width * env.width;
                Vector2D closest_food = {0, 0};
                bool found = false;

                for (int f = 0; f < env.num_food_sources; f++) {
                    if (env.food_sources[f].amount > 0) {
                        float dist_sq = distance_squared(ant->position, env.food_sources[f].position);
                        if (dist_sq < closest_dist_sq) {
                            closest_dist_sq = dist_sq;
                            closest_food = env.food_sources[f].position;
                            found = true;
                        }
                    }
                }

                if (found) {
                    float angle_to_food = get_angle(ant->position, closest_food);
                    float angle_diff = wrap_angle(angle_to_food - ant->direction);

                    if (angle_diff > M_PI / 8) {
                        env.actions[i] = ACTION_TURN_RIGHT;
                    } else if (angle_diff < -M_PI / 8) {
                        env.actions[i] = ACTION_TURN_LEFT;
                    } else {
                        env.actions[i] = ACTION_MOVE_FORWARD;
                    }
                } else {
                    // No food visible, just move forward
                    env.actions[i] = ACTION_MOVE_FORWARD;
                }
            }
        }

        c_step(&env);
        c_render(&env);

        // Print stats every 60 frames
        if (env.tick % 60 == 0) {
            printf("Tick: %d | Colony 1: %d | Colony 2: %d | Episodes: %.0f | Avg Score: %.2f\n",
                   env.tick,
                   env.colonies[0].food_collected,
                   env.colonies[1].food_collected,
                   env.log.n,
                   env.log.n > 0 ? env.log.score / env.log.n : 0.0f);
        }
    }

    // Cleanup
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);

    return 0;
}
