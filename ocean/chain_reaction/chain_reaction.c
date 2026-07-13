#include <time.h>

#include "chain_reaction.h"

static void demo(void) {
    ChainEnv env = {0};
    env.num_agents = 1;
    env.rows = 9;
    env.cols = 6;
    env.max_steps = 132;
    env.opponent_policy = CHAINENV_OPPONENT_HEURISTIC;
    env.win_reward = 1.0f;
    env.loss_reward = -1.0f;
    env.invalid_move_reward = -1.0f;
    env.rng = (unsigned int)time(NULL);
    init_chainenv(&env);

    float observation_buf[CHAINENV_OBS_SIZE] = {0};
    float action_buf[1] = {0};
    float reward_buf[1] = {0};
    float terminal_buf[1] = {0};
    unsigned char action_mask_buf[CHAINENV_MAX_ACTIONS] = {0};
    env.observations = observation_buf;
    env.actions = action_buf;
    env.rewards = reward_buf;
    env.terminals = terminal_buf;
    env.action_mask = action_mask_buf;
    env.obs_ptr[0] = observation_buf;
    env.action_ptr[0] = action_buf;
    env.reward_ptr[0] = reward_buf;
    env.terminal_ptr[0] = terminal_buf;
    env.action_mask_ptr[0] = action_mask_buf;
    env.client = make_client();
    c_reset(&env);

    while (true) {
        chainenv_layout_board(env.client, &env);

        bool allow_input = !chainenv_animation_active(&env);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            if (env.end_game) {
                c_reset(&env);
            } else if (allow_input) {
                int action = chainenv_pick_action(&env, env.client, GetMousePosition());
                if (action >= 0) {
                    if (!chainenv_is_legal_move(&env, action, CHAINENV_PLAYER_RED)) {
                        chainenv_show_invalid_hint(&env);
                    } else {
                        chainenv_clear_invalid_hint(&env);
                        env.actions[0] = (float)action;
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
