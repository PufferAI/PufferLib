#include <time.h>
#include "pathfinder.h"

static int read_manual_action(void) {
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) return PATHFINDER_ACT_NORTH;
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) return PATHFINDER_ACT_EAST;
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) return PATHFINDER_ACT_SOUTH;
    if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) return PATHFINDER_ACT_WEST;
    return -1;
}

int main(void) {
    Pathfinder env;
    memset(&env, 0, sizeof(env));

    float observations[PATHFINDER_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.player_mode = true;
    env.rng = (unsigned int)time(NULL);
    env.branch_prob = 0.35f;
    env.loop_prob = 0.10f;
    env.start_solution_len = 4;
    env.curriculum_enabled = 1;
    env.max_steps = 128;
    env.step_penalty = -0.001f;
    env.new_wall_penalty = 0.0f;
    env.known_wall_death_penalty = -1.0f;
    env.repeat_move_death_penalty = -1.0f;
    env.new_cell_reward = 0.01f;
    env.revisit_penalty = -0.01f;
    env.impossible_penalty = -1.0f;
    env.goal_reward = 1.0f;

    init(&env);
    c_reset(&env);

    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }

        int action = read_manual_action();
        if (action >= 0) {
            actions[0] = (float)action;
            c_step(&env);
        } else if (IsKeyPressed(KEY_SPACE)) {
            actions[0] = (float)(pathfinder_rand(&env) % PATHFINDER_NUM_ACTIONS);
            c_step(&env);
        }

        c_render(&env);
    }

    c_close(&env);
    return 0;
}
