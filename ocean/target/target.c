#include "target.h"
#include "puffernet.h"

void demo() {
    // encoder(128x28=3584) + decoder(15x128=1920) + 4x mingru(384x128=49152) = 202112
    Weights* weights = load_weights("resources/target/target_weights.bin");

    int logit_sizes[2] = {9, 5};
    PufferNet* net = make_puffernet(weights, 8, 28, 128, 4, logit_sizes, 2);

    float observations[8*28] = {0};
    float actions[8*2] = {0};
    float rewards[8] = {0};
    float terminals[8] = {0};
    Agent agents[8] = {0};
    Goal goals[4] = {0};

    Target env = {
        .width = 952,
        .height = 592,
        .num_agents = 8,
        .num_goals = 4,
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .agents = agents,
        .goals = goals,
    };

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        forward_puffernet(net, env.observations, env.actions);
        c_step(&env);
        c_render(&env);
    }
    free_puffernet(net);
    free(weights);
}

int main() {
    demo();
}
