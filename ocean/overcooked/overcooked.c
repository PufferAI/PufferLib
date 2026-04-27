#include <time.h>
#include "overcooked.h"
#include "puffernet.h"

int main(int argc, char** argv) {
    LayoutType layout_id = LAYOUT_CRAMPED_ROOM;
    if (argc > 1) {
        layout_id = get_layout_by_name(argv[1]);
    }

    int num_agents = 2;
    int num_obs = 43;

    // Select weights file and size based on layout
    const char* weights_file;
    int weights_size;
    if (layout_id == LAYOUT_ASYMMETRIC_ADVANTAGES) {
        weights_file = "resources/overcooked/puffer_overcooked_weights_aa.bin";
        weights_size = 138631;
    } else if (layout_id == LAYOUT_FORCED_COORDINATION) {
        weights_file = "resources/overcooked/puffer_overcooked_weights_fc.bin";
        weights_size = 138631;
    } else if (layout_id == LAYOUT_COORDINATION_RING) {
        weights_file = "resources/overcooked/puffer_overcooked_weights_cor.bin";
        weights_size = 138631;
    } else if (layout_id == LAYOUT_COUNTER_CIRCUIT) {
        weights_file = "resources/overcooked/puffer_overcooked_weights_cc.bin";
        weights_size = 138631;
    } else {
        weights_file = "resources/overcooked/puffer_overcooked_weights_cr.bin";
        weights_size = 138631;
    }

    // Weights* weights = load_weights(weights_file);
    // int logit_sizes[] = {6};
    // LinearLSTM* net = make_linearlstm(weights, num_agents, num_obs, logit_sizes, 1);

    Overcooked env = {
        .layout_id = layout_id,
        .num_agents = num_agents,
        .grid_size = 100,
        .rewards_config = {
            .dish_served_whole_team = 1.0f,
            .dish_served_agent = 0.0f,
            .pot_started = 0.15f,
            .ingredient_added = 0.15f,
            .ingredient_picked = 0.05f,
            .plate_picked = 0.05f,
            .soup_plated = 0.20f,
            .wrong_dish_served = 0.0f,
            .step_penalty = 0.0f
        },
        .observation_size = num_obs
    };

    env.observations = (float*)calloc(num_obs * num_agents, sizeof(float));
    env.actions = (float*)calloc(num_agents, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (float*)calloc(num_agents, sizeof(float));

    init(&env);
    c_reset(&env);
    c_render(&env);

    srand(time(NULL));

    while (!WindowShouldClose()) {
        // forward_linearlstm(net, env.observations, env.actions);
        for (int i = 0; i < num_agents; i++) {
            env.actions[i] = rand() % 6;
        }
        c_step(&env);
        c_render(&env);
    }

    // free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);

    return 0;
}
