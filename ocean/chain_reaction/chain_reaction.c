#include <time.h>
#include <unistd.h>

#include "chain_reaction.h"

static void demo(void) {
    ChainEnv env = {0};
    env.num_agents = 1;
    env.rows = 9;
    env.cols = 6;
    env.max_steps = 132;
    env.opponent_policy = HEURISTIC_OPPONENT;
    env.win_reward = 1.0f;
    env.loss_reward = -1.0f;
    env.invalid_move_reward = -1.0f;
    env.rng = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    init(&env);

    float observation_buf[OBS_SIZE] = {0};
    float action_buf[1] = {0};
    float reward_buf[1] = {0};
    float terminal_buf[1] = {0};
    unsigned char action_mask_buf[MAX_ACTIONS] = {0};
    env.obs_ptr[0] = observation_buf;
    env.action_ptr[0] = action_buf;
    env.reward_ptr[0] = reward_buf;
    env.terminal_ptr[0] = terminal_buf;
    env.action_mask_ptr[0] = action_mask_buf;
    env.client = make_client();
    c_reset(&env);

    while (true) {
        layout_board(&env);

        bool allow_input = env.client->animation_clock
            >= env.client->animation_total - 1.0e-4f;
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            if (env.end_game) {
                c_reset(&env);
            } else if (allow_input) {
                Vector2 mouse = GetMousePosition();
                float board_w = env.client->cell_size * env.cols;
                float board_h = env.client->cell_size * env.rows;
                bool on_board = mouse.x >= env.client->board_x
                    && mouse.x < env.client->board_x + board_w
                    && mouse.y >= env.client->board_y
                    && mouse.y < env.client->board_y + board_h;
                if (on_board) {
                    int col = (int)((mouse.x - env.client->board_x) / env.client->cell_size);
                    int row = (int)((mouse.y - env.client->board_y) / env.client->cell_size);
                    int action = row * env.cols + col;
                    if (!is_legal_move(&env, action, RED_PLAYER)) {
                        env.client->invalid_hint_time = 1.9f;
                    } else {
                        clear_invalid_hint(env.client);
                        *env.action_ptr[0] = (float)action;
                        c_step(&env);
                    }
                }
            }
        }

        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }

        c_render(&env);
    }
}

int main(void) {
    demo();
    return 0;
}
