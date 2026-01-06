#include <time.h>
#include "tetris.h"
#include "puffernet.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

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


void demo() {
    Tetris env = {
        .n_rows = 20,
        .n_cols = 10,
        .use_deck_obs = true,
        .n_noise_obs = 0,
        .n_init_garbage = 0,
    };
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    Weights* weights = load_weights("resources/tetris/tetris_weights.bin", 588552);
    int logit_sizes[1] = {7};
    LinearLSTM* net = make_linearlstm_256(weights, 1, 234, logit_sizes, 1);

    // State tracking for single-press actions to avoid using IsKeyPressed
    // because IsKeyPressed doesn't work well in web browsers
    static bool rotate_key_was_down = false;
    static bool hard_drop_key_was_down = false;
    static bool swap_key_was_down = false;

    int frame = 0;
    env.actions[0] = 0;
    while (!WindowShouldClose()) {
        bool process_logic = true;
        frame++;

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (frame % 3 != 0) {
                // This effectively slows down the client by 3x
                process_logic = false;
            } else {
                // Use KeyDown for left, right, down to allow continuous input
                // Though, IsKeyDown can overshoot ...
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    env.actions[0] = 1;
                } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    env.actions[0] = 2;
                } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    env.actions[0] = 4; // Soft drop
                }
                // Manual state tracking for single-press actions, mutually exclusive
                else if ((IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) && !rotate_key_was_down) {
                    env.actions[0] = 3; // Rotate
                } else if (IsKeyDown(KEY_SPACE) && !hard_drop_key_was_down) {
                    env.actions[0] = 5; // Hard drop
                } else if (IsKeyDown(KEY_C) && !swap_key_was_down) {
                    env.actions[0] = 6; // Swap
                }
            }
        } else {
            forward_linearlstm(net, env.observations, env.actions);
        }

        if (process_logic) {
            // Update key state flags after processing actions for the frame
            rotate_key_was_down = IsKeyDown(KEY_UP) || IsKeyDown(KEY_W);
            hard_drop_key_was_down = IsKeyDown(KEY_SPACE);
            swap_key_was_down = IsKeyDown(KEY_C);

            c_step(&env);

            env.actions[0] = 0;
        }

        c_render(&env);
    }

    free_linearlstm(net);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
