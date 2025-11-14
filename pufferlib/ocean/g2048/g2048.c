#include "g2048.h"
#include "puffernet.h"

// Network with hidden size 256. Should go to puffernet
LinearLSTM* make_linearlstm_256(Weights* weights, int num_agents, int input_dim, int logit_sizes[], int num_actions) {
    LinearLSTM* net = calloc(1, sizeof(LinearLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents*input_dim, sizeof(float));
    int hidden_dim = 256;
    net->encoder = make_linear(weights, num_agents, input_dim, hidden_dim);
    net->gelu1 = make_gelu(num_agents, hidden_dim);
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
    }
    net->actor = make_linear(weights, num_agents, hidden_dim, atn_sum);
    net->value_fn = make_linear(weights, num_agents, hidden_dim, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_dim, hidden_dim);
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, num_actions);
    return net;
}

int main() {
    srand(time(NULL));
    Game env;
    unsigned char observations[SIZE * SIZE] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    Weights* weights = load_weights("resources/g2048/g2048_weights.bin", 531973);
    int logit_sizes[1] = {4};
    LinearLSTM* net = make_linearlstm_256(weights, 1, 16, logit_sizes, 1);
    c_reset(&env);
    c_render(&env);

    // Main game loop
    int frame = 0;
    int action = -1;
    while (!WindowShouldClose()) {
        c_render(&env);
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
            for (int i = 0; i < 16; i++) {
                net->obs[i] = env.observations[i];
            }
            forward_linearlstm(net, net->obs, env.actions);
        }

        if (action > 0) {
            c_step(&env);
        }

        if (IsKeyDown(KEY_LEFT_SHIFT) && action > 0) {
            // Don't need to be super reactive
            WaitTime(0.1);
        }        
    }

    free_linearlstm(net);
    c_close(&env);
    printf("Game Over! Final Max Tile: %d\n", env.score);
    return 0;
}
