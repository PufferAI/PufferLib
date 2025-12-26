#include <time.h>
#include "overcooked.h"
#include "puffernet.h"

int main() {
    int num_agents = 2;
    int num_obs = 39;

    Weights* weights = load_weights("resources/overcooked/puffer_overcooked_weights.bin", 137743);
    int logit_sizes[] = {6};
    LinearLSTM* net = make_linearlstm(weights, num_agents, num_obs, logit_sizes, 1);

    Overcooked env = {
        .width = 5,
        .height = 5,
        .num_agents = num_agents,
        .grid_size = 100,
        .rewards_config = {
            .dish_served_whole_team = 1.0f,
            .dish_served_agent = 0.0f,
            .pot_started = 0.15f,
            .ingredient_added = 0.15f,
            .ingredient_picked = 0.05f,
            .soup_plated = 0.20f,
            .wrong_dish_served = 0.0f,
            .step_penalty = 0.0f
        },
        .observation_size = num_obs
    };

    env.observations = (float*)calloc(num_obs * num_agents, sizeof(float));
    env.actions = (int*)calloc(num_agents, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));

    init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        forward_linearlstm(net, env.observations, env.actions);
        c_step(&env);
        c_render(&env);
    }

    free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);

    return 0;
}
