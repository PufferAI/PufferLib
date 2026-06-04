#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../pathfinder.h"

static void setup_env(Pathfinder* env, float* obs, float* actions,
        float* rewards, float* terminals) {
    memset(env, 0, sizeof(*env));
    env->observations = obs;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
    env->num_agents = 1;
    env->branch_prob = 0.35f;
    env->loop_prob = 0.10f;
    env->extra_entry_prob = 0.0f;
    env->min_solution_len = 1;
    env->max_solution_len = 0;
    env->max_steps = 128;
    env->rng = 7;
    init(env);
}

static void test_constants(void) {
    assert(PATHFINDER_ROWS == 6);
    assert(PATHFINDER_COLS == 6);
    assert(PATHFINDER_NUM_WALLS == 84);
    assert(PATHFINDER_OBS_SIZE == 86);
    assert(PATHFINDER_NUM_ACTIONS == 4);
    assert(PATHFINDER_RENDER_WIDTH > PATHFINDER_RENDER_BOARD_X + PATHFINDER_RENDER_BOARD_SIZE);
    assert(PATHFINDER_RENDER_HEIGHT > PATHFINDER_RENDER_BOARD_Y + PATHFINDER_RENDER_BOARD_SIZE);
}

static void test_reset_initializes_a1_and_unknown_walls(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);

    c_reset(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(fabsf(obs[PATHFINDER_NUM_WALLS] - 0.0f) < 1e-6f);
    assert(fabsf(obs[PATHFINDER_NUM_WALLS + 1] - 0.0f) < 1e-6f);
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        assert(fabsf(obs[i] + 1.0f) < 1e-6f);
    }
}

static void test_generated_mazes_connect_a1_to_goal(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);

    for (int i = 0; i < 100; i++) {
        c_reset(&env);
        assert(pathfinder_has_path_to_goal(&env.state));
        assert(env.state.shortest_path_len >= 0);
    }
}

static void test_open_edge_reveals_and_moves(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    c_reset(&env);

    memset(env.state.true_walls, 1, sizeof(env.state.true_walls));
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        env.state.known_walls[i] = PATHFINDER_UNKNOWN;
    }
    env.state.agent_row = 0;
    env.state.agent_col = 0;
    env.state.goal_row = 5;
    env.state.goal_col = 5;
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env.state.true_walls[east_wall] = 0;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 1);
    assert(fabsf(obs[east_wall] - 0.0f) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_blocked_edge_reveals_and_stays(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    c_reset(&env);

    memset(env.state.true_walls, 1, sizeof(env.state.true_walls));
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        env.state.known_walls[i] = PATHFINDER_UNKNOWN;
    }
    env.state.agent_row = 0;
    env.state.agent_col = 0;
    env.state.goal_row = 5;
    env.state.goal_col = 5;
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env.state.true_walls[east_wall] = 1;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(fabsf(obs[east_wall] - 1.0f) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_known_wall_repeat_terminates_with_penalty(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    c_reset(&env);

    memset(env.state.true_walls, 1, sizeof(env.state.true_walls));
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        env.state.known_walls[i] = PATHFINDER_UNKNOWN;
    }
    env.state.agent_row = 0;
    env.state.agent_col = 0;
    env.state.goal_row = 5;
    env.state.goal_col = 5;
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env.state.true_walls[east_wall] = 1;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(fabsf(rewards[0] - PATHFINDER_STEP_PENALTY) < 1e-6f);
    assert(terminals[0] == 0.0f);

    c_step(&env);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_KNOWN_WALL_PENALTY)) < 1e-6f);
    assert(terminals[0] == 1.0f);
    assert(env.log.n >= 1.0f);
    assert(env.log.success == 0.0f);
}

static void test_revisiting_previously_left_square_has_penalty(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    c_reset(&env);

    memset(env.state.true_walls, 1, sizeof(env.state.true_walls));
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        env.state.known_walls[i] = PATHFINDER_UNKNOWN;
    }
    env.state.agent_row = 0;
    env.state.agent_col = 0;
    env.state.goal_row = 5;
    env.state.goal_col = 5;
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env.state.true_walls[east_wall] = 0;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(fabsf(rewards[0] - PATHFINDER_STEP_PENALTY) < 1e-6f);

    actions[0] = PATHFINDER_ACT_WEST;
    c_step(&env);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_REVISIT_PENALTY)) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_max_solution_len_limits_curriculum_distance(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.max_solution_len = 2;

    for (int i = 0; i < 100; i++) {
        c_reset(&env);
        assert(pathfinder_has_path_to_goal(&env.state));
        assert(env.state.shortest_path_len >= 1);
        assert(env.state.shortest_path_len <= 2);
    }
}

static void test_curriculum_graduates_to_longer_mazes(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.branch_prob = 0.0f;
    env.loop_prob = 0.0f;
    env.max_solution_len = 4;
    c_reset(&env);

    assert(pathfinder_curriculum_max_solution_len(&env) == 4);
    for (int i = 0; i < PATHFINDER_CURRICULUM_WINDOW; i++) {
        env.state.success = 1;
        env.state.tick = 1;
        env.state.shortest_path_len = 1;
        env.state.agent_path_len = 1;
        add_log(&env);
    }
    assert(pathfinder_curriculum_max_solution_len(&env) == 5);

    bool saw_longer_than_four = false;
    for (int i = 0; i < 100; i++) {
        c_reset(&env);
        assert(env.state.shortest_path_len >= 3);
        assert(env.state.shortest_path_len <= 5);
        if (env.state.shortest_path_len > 4) {
            saw_longer_than_four = true;
        }
    }
    assert(saw_longer_than_four);
}

static void test_reaching_goal_terminates(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    c_reset(&env);

    memset(env.state.true_walls, 1, sizeof(env.state.true_walls));
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        env.state.known_walls[i] = PATHFINDER_UNKNOWN;
    }
    env.state.agent_row = 0;
    env.state.agent_col = 0;
    env.state.goal_row = 0;
    env.state.goal_col = 1;
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env.state.true_walls[east_wall] = 0;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(rewards[0] > 0.9f);
    assert(env.log.success >= 1.0f);
    assert(env.log.n >= 1.0f);
}

int main(void) {
    test_constants();
    test_reset_initializes_a1_and_unknown_walls();
    test_generated_mazes_connect_a1_to_goal();
    test_open_edge_reveals_and_moves();
    test_blocked_edge_reveals_and_stays();
    test_known_wall_repeat_terminates_with_penalty();
    test_revisiting_previously_left_square_has_penalty();
    test_max_solution_len_limits_curriculum_distance();
    test_curriculum_graduates_to_longer_mazes();
    test_reaching_goal_terminates();
    printf("pathfinder core tests passed\n");
    return 0;
}
