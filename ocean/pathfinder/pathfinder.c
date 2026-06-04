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
    env.rng = (unsigned int)time(NULL);
    env.branch_prob = 0.35f;
    env.loop_prob = 0.10f;
    env.extra_entry_prob = 0.0f;
    env.min_solution_len = 1;
    env.max_solution_len = 4;
    env.max_steps = 128;

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
