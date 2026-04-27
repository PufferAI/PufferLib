#include <time.h>
#include "tetris.h"
#include "puffernet.h"

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

    Weights* weights = load_weights("resources/tetris/tetris_weights.bin");
    int logit_sizes[1] = {7};
    PufferNet* net = make_puffernet(weights, 1, 234, 512, 2, logit_sizes, 1);

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
                process_logic = false;
            } else {
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    env.actions[0] = 1;
                } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    env.actions[0] = 2;
                } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    env.actions[0] = 4;
                } else if ((IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) && !rotate_key_was_down) {
                    env.actions[0] = 3;
                } else if (IsKeyDown(KEY_SPACE) && !hard_drop_key_was_down) {
                    env.actions[0] = 5;
                } else if (IsKeyDown(KEY_C) && !swap_key_was_down) {
                    env.actions[0] = 6;
                }
            }
        } else {
            forward_puffernet(net, env.observations, env.actions);
        }

        if (process_logic) {
            rotate_key_was_down = IsKeyDown(KEY_UP) || IsKeyDown(KEY_W);
            hard_drop_key_was_down = IsKeyDown(KEY_SPACE);
            swap_key_was_down = IsKeyDown(KEY_C);
            c_step(&env);
            env.actions[0] = 0;
        }

        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
    return 0;
}
