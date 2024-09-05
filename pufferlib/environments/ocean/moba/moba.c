#include "moba.h"

int main() {
    int num_agents = 10;
    int num_creeps = 100;
    int num_neutrals = 72;
    int num_towers = 24;
    int vision_range = 5;
    float agent_speed = 1.0;
    bool discretize = true;
    float reward_death = -1.0;
    float reward_xp = 0.006;
    float reward_distance = 0.05;
    float reward_tower = 3.0;

    MOBA* env = init_moba(num_agents, num_creeps, num_neutrals, num_towers, vision_range,
        agent_speed, discretize, reward_death, reward_xp, reward_distance, reward_tower);

    GameRenderer* renderer = init_game_renderer(32, 41, 23);

    reset(env);
    int frame = 0;
    while (!WindowShouldClose()) {
        if (frame % 12 == 0) {
            step(env);
        }
        render_game(renderer, env, frame);
        frame = (frame + 1) % 12;
    }
    free_moba(env);
    close_game_renderer(renderer);
    return 0;
}
