#include "convert.h"
#include "puffernet.h"

int main() {
    Convert env = {
        .width = 1920,
        .height = 1080,
        .num_agents = 1024,
        .num_factories = 32,
        .num_resources = 8,
    };
    init(&env);

    int num_obs = 2*env.num_resources + 4 + env.num_resources;
    env.observations = calloc(env.num_agents*num_obs, sizeof(float));
    env.actions = calloc(2*env.num_agents, sizeof(int));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    Weights* weights = load_weights("resources/convert/convert_weights.bin", 137743);
    int logit_sizes[2] = {9, 5};
    LinearLSTM* net = make_linearlstm(weights, env.num_agents, num_obs, logit_sizes, 2);

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            env.actions[2*i] = rand() % 9;
            env.actions[2*i + 1] = rand() % 5;
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

