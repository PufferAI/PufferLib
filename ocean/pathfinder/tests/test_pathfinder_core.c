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

static void setup_manual_state(Pathfinder* env, int goal_row, int goal_col) {
    State* s = &env->state;
    memset(s, 0, sizeof(*s));
    memset(s->true_walls, 1, sizeof(s->true_walls));
    pathfinder_reset_known(s);
    s->agent_row = 0;
    s->agent_col = 0;
    s->goal_row = goal_row;
    s->goal_col = goal_col;
    pathfinder_mark_visited(s, 0, 0);
}

static void open_manual_edge(Pathfinder* env, int row, int col, int next_row, int next_col) {
    pathfinder_open_edge(&env->state, row, col, next_row, next_col);
    env->state.shortest_path_len = pathfinder_shortest_path(&env->state);
    refresh_state(env);
}

static void assert_wall_observations_unknown(Pathfinder* env, float* obs) {
    assert(env->state.known_wall_count == 0);
    assert(env->state.known_open_count == 0);
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        assert(fabsf(env->state.known_walls[i] - PATHFINDER_UNKNOWN) < 1e-6f);
        assert(fabsf(obs[i] - PATHFINDER_UNKNOWN) < 1e-6f);
    }
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

static void test_wall_indexing_and_action_deltas(void) {
    assert(pathfinder_v_wall(0, 0) == 0);
    assert(pathfinder_v_wall(5, 6) == PATHFINDER_VERTICAL_WALLS - 1);
    assert(pathfinder_h_wall(0, 0) == PATHFINDER_VERTICAL_WALLS);
    assert(pathfinder_h_wall(6, 5) == PATHFINDER_NUM_WALLS - 1);
    assert(pathfinder_wall_between(0, 0, 0, 1) == pathfinder_v_wall(0, 1));
    assert(pathfinder_wall_between(0, 1, 0, 0) == pathfinder_v_wall(0, 1));
    assert(pathfinder_wall_between(0, 0, 1, 0) == pathfinder_h_wall(1, 0));
    assert(pathfinder_wall_between(1, 0, 0, 0) == pathfinder_h_wall(1, 0));
    assert(pathfinder_wall_between(0, 0, 1, 1) == -1);

    int d_row;
    int d_col;
    pathfinder_action_delta(PATHFINDER_ACT_NORTH, &d_row, &d_col);
    assert(d_row == -1 && d_col == 0);
    pathfinder_action_delta(PATHFINDER_ACT_EAST, &d_row, &d_col);
    assert(d_row == 0 && d_col == 1);
    pathfinder_action_delta(PATHFINDER_ACT_SOUTH, &d_row, &d_col);
    assert(d_row == 1 && d_col == 0);
    pathfinder_action_delta(PATHFINDER_ACT_WEST, &d_row, &d_col);
    assert(d_row == 0 && d_col == -1);
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
    assert(env.state.visited[0][0] == 1);
    assert(env.state.visited_count == 1);
    assert(fabsf(obs[PATHFINDER_NUM_WALLS] - 0.0f) < 1e-6f);
    assert(fabsf(obs[PATHFINDER_NUM_WALLS + 1] - 0.0f) < 1e-6f);
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        assert(fabsf(obs[i] + 1.0f) < 1e-6f);
    }
}

static void test_action_mask_allows_unknown_edges_on_reset(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[PATHFINDER_NUM_ACTIONS] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.action_mask = action_mask;

    c_reset(&env);

    for (int i = 0; i < PATHFINDER_NUM_ACTIONS; i++) {
        assert(action_mask[i] == 1);
    }
}

static void test_action_mask_blocks_known_wall_but_forced_hit_still_dies(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[PATHFINDER_NUM_ACTIONS] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.action_mask = action_mask;
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);

    for (int i = 0; i < PATHFINDER_NUM_ACTIONS; i++) {
        assert(action_mask[i] == 1);
    }

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 0.0f);
    assert(action_mask[PATHFINDER_ACT_EAST] == 0);
    assert(action_mask[PATHFINDER_ACT_NORTH] == 1);
    assert(action_mask[PATHFINDER_ACT_SOUTH] == 1);
    assert(action_mask[PATHFINDER_ACT_WEST] == 1);

    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(env.log.known_wall_deaths == 1.0f);
}

static void test_known_wall_death_restarts_same_map_with_unknown_wall_memory(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[PATHFINDER_NUM_ACTIONS] = {0};
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    setup_env(&env, obs, actions, rewards, terminals);
    env.action_mask = action_mask;
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);

    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    memcpy(true_walls, env.state.true_walls, sizeof(true_walls));

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 0.0f);
    assert(fabsf(obs[east_wall] - PATHFINDER_WALL) < 1e-6f);
    assert(action_mask[PATHFINDER_ACT_EAST] == 0);

    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(env.log.n == 1.0f);
    assert(env.log.success == 0.0f);
    assert(env.log.known_wall_deaths == 1.0f);
    assert(env.state.tick == 0);
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(env.state.goal_row == 5);
    assert(env.state.goal_col == 5);
    assert(env.state.visited_count == 1);
    assert(env.state.visited[0][0] == 1);
    assert(memcmp(env.state.true_walls, true_walls, sizeof(true_walls)) == 0);
    assert_wall_observations_unknown(&env, obs);
    assert(action_mask[PATHFINDER_ACT_EAST] == 1);
}

static void test_position_observation_updates_after_move(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(fabsf(obs[PATHFINDER_NUM_WALLS] - 0.2f) < 1e-6f);
    assert(fabsf(obs[PATHFINDER_NUM_WALLS + 1] - 0.0f) < 1e-6f);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD)) < 1e-6f);
    assert(env.state.visited[0][1] == 1);
    assert(env.state.visited_count == 2);
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

static void test_generated_shortest_path_matches_goal_distance(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.max_solution_len = PATHFINDER_MAX_SOLUTION_LEN;

    for (int i = 0; i < 200; i++) {
        c_reset(&env);
        assert(env.state.shortest_path_len == pathfinder_shortest_path(&env.state));
    }
}

static void test_open_edge_reveals_and_moves(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    open_manual_edge(&env, 0, 0, 0, 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 1);
    assert(fabsf(obs[east_wall] - 0.0f) < 1e-6f);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD)) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_blocked_edge_reveals_and_stays(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(fabsf(obs[east_wall] - 1.0f) < 1e-6f);
    assert(fabsf(rewards[0] - PATHFINDER_STEP_PENALTY) < 1e-6f);
    assert(terminals[0] == 0.0f);
    assert(env.state.known_wall_death == 0);
}

static void test_known_wall_repeat_terminates_with_penalty(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(fabsf(rewards[0] - PATHFINDER_STEP_PENALTY) < 1e-6f);
    assert(terminals[0] == 0.0f);

    c_step(&env);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_KNOWN_WALL_PENALTY +
            PATHFINDER_KNOWN_WALL_DEATH_PENALTY)) < 1e-6f);
    assert(terminals[0] == 1.0f);
    assert(env.log.n >= 1.0f);
    assert(env.log.success == 0.0f);
    assert(env.log.known_wall_deaths == 1.0f);
}

static void test_invalid_action_penalizes_without_terminating(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);

    actions[0] = 99;
    c_step(&env);

    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_IMPOSSIBLE_PENALTY)) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_boundary_action_reveals_wall(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_NORTH;
    c_step(&env);

    int north_wall = pathfinder_wall_between(0, 0, -1, 0);
    assert(north_wall == pathfinder_h_wall(0, 0));
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(fabsf(obs[north_wall] - PATHFINDER_WALL) < 1e-6f);
    assert(fabsf(rewards[0] - PATHFINDER_STEP_PENALTY) < 1e-6f);
    assert(terminals[0] == 0.0f);
}

static void test_revisiting_previously_left_square_has_penalty(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD)) < 1e-6f);

    actions[0] = PATHFINDER_ACT_WEST;
    c_step(&env);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_REVISIT_PENALTY)) < 1e-6f);
    assert(terminals[0] == 0.0f);
    assert(env.state.revisit_count == 1);
}

static void test_repeating_directed_move_dies_and_restarts_same_map_blind(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    memcpy(true_walls, env.state.true_walls, sizeof(true_walls));

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(terminals[0] == 0.0f);
    assert(env.state.agent_col == 1);
    assert(fabsf(obs[east_wall] - PATHFINDER_OPEN) < 1e-6f);

    actions[0] = PATHFINDER_ACT_WEST;
    c_step(&env);
    assert(terminals[0] == 0.0f);
    assert(env.state.agent_col == 0);
    assert(env.state.revisit_count == 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(rewards[0] <= -1.0f);
    assert(env.log.n == 1.0f);
    assert(env.log.success == 0.0f);
    assert(env.log.repeat_move_deaths == 1.0f);
    assert(env.state.tick == 0);
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(env.state.goal_row == 5);
    assert(env.state.goal_col == 5);
    assert(memcmp(env.state.true_walls, true_walls, sizeof(true_walls)) == 0);
    assert_wall_observations_unknown(&env, obs);
}

static void test_longer_backtrack_then_forward_move_is_allowed(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);
    open_manual_edge(&env, 0, 1, 0, 2);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(terminals[0] == 0.0f);
    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    assert(terminals[0] == 0.0f);
    actions[0] = PATHFINDER_ACT_WEST;
    c_step(&env);
    assert(terminals[0] == 0.0f);
    actions[0] = PATHFINDER_ACT_WEST;
    c_step(&env);
    assert(terminals[0] == 0.0f);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 0.0f);
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 1);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_REVISIT_PENALTY)) < 1e-6f);
}

static void test_known_open_edge_to_new_square_has_no_extra_penalty(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);
    open_manual_edge(&env, 0, 1, 0, 2);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    int second_wall = pathfinder_wall_between(0, 1, 0, 2);
    env.state.known_walls[second_wall] = PATHFINDER_OPEN;
    refresh_state(&env);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(env.state.agent_col == 2);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD)) < 1e-6f);
    assert(env.state.revisit_count == 0);
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
        assert(env.state.shortest_path_len == 2);
    }
}

static void test_success_generates_next_map_one_step_farther(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.branch_prob = 0.0f;
    env.loop_prob = 0.0f;
    env.max_solution_len = 1;
    setup_manual_state(&env, 0, 1);
    open_manual_edge(&env, 0, 0, 0, 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(env.log.success == 1.0f);
    assert(pathfinder_curriculum_max_solution_len(&env) == 2);
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(env.state.shortest_path_len == 2);
    assert(env.state.goal_row + env.state.goal_col == 2);
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        assert(fabsf(env.state.known_walls[i] - PATHFINDER_UNKNOWN) < 1e-6f);
    }
}

static void test_failure_retry_does_not_graduate_curriculum(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    setup_env(&env, obs, actions, rewards, terminals);
    env.max_solution_len = 4;
    setup_manual_state(&env, 5, 5);
    refresh_state(&env);
    memcpy(true_walls, env.state.true_walls, sizeof(true_walls));

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(env.curriculum_level == 0);
    assert(pathfinder_curriculum_max_solution_len(&env) == 4);
    assert(memcmp(env.state.true_walls, true_walls, sizeof(true_walls)) == 0);
}

static void test_curriculum_caps_at_board_max(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.max_solution_len = 4;
    env.curriculum_level = PATHFINDER_MAX_SOLUTION_LEN;

    assert(pathfinder_curriculum_max_solution_len(&env) == PATHFINDER_MAX_SOLUTION_LEN);
    assert(pathfinder_curriculum_min_solution_len(&env) == PATHFINDER_MAX_SOLUTION_LEN);
}

static void test_generation_can_reach_board_max_distance(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.branch_prob = 0.0f;
    env.loop_prob = 0.0f;
    env.max_solution_len = PATHFINDER_MAX_SOLUTION_LEN;

    for (int i = 0; i < 200; i++) {
        c_reset(&env);
        assert(env.state.shortest_path_len >= 1);
        assert(env.state.shortest_path_len <= PATHFINDER_MAX_SOLUTION_LEN);
        assert(pathfinder_has_path_to_goal(&env.state));
    }
}

static void test_timeout_restarts_same_map_with_unknown_wall_memory(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[PATHFINDER_NUM_ACTIONS] = {0};
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    setup_env(&env, obs, actions, rewards, terminals);
    env.action_mask = action_mask;
    env.max_steps = 1;
    setup_manual_state(&env, 5, 5);
    open_manual_edge(&env, 0, 0, 0, 1);
    memcpy(true_walls, env.state.true_walls, sizeof(true_walls));

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD)) < 1e-6f);
    assert(env.log.n == 1.0f);
    assert(env.log.success == 0.0f);
    assert(env.log.episode_length == 1.0f);
    assert(env.state.tick == 0);
    assert(env.state.agent_row == 0);
    assert(env.state.agent_col == 0);
    assert(env.state.goal_row == 5);
    assert(env.state.goal_col == 5);
    assert(env.state.visited_count == 1);
    assert(env.state.visited[0][0] == 1);
    assert(env.state.visited[0][1] == 0);
    assert(memcmp(env.state.true_walls, true_walls, sizeof(true_walls)) == 0);
    assert_wall_observations_unknown(&env, obs);
    assert(action_mask[PATHFINDER_ACT_EAST] == 1);
}

static void test_reaching_goal_terminates(void) {
    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE];
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    setup_manual_state(&env, 0, 1);
    open_manual_edge(&env, 0, 0, 0, 1);

    actions[0] = PATHFINDER_ACT_EAST;
    c_step(&env);

    assert(terminals[0] == 1.0f);
    assert(fabsf(rewards[0] -
        (PATHFINDER_STEP_PENALTY + PATHFINDER_NEW_CELL_REWARD +
            PATHFINDER_GOAL_REWARD)) < 1e-6f);
    assert(env.log.success >= 1.0f);
    assert(env.log.n >= 1.0f);
}

int main(void) {
    test_constants();
    test_wall_indexing_and_action_deltas();
    test_reset_initializes_a1_and_unknown_walls();
    test_action_mask_allows_unknown_edges_on_reset();
    test_action_mask_blocks_known_wall_but_forced_hit_still_dies();
    test_known_wall_death_restarts_same_map_with_unknown_wall_memory();
    test_position_observation_updates_after_move();
    test_generated_mazes_connect_a1_to_goal();
    test_generated_shortest_path_matches_goal_distance();
    test_open_edge_reveals_and_moves();
    test_blocked_edge_reveals_and_stays();
    test_known_wall_repeat_terminates_with_penalty();
    test_invalid_action_penalizes_without_terminating();
    test_boundary_action_reveals_wall();
    test_revisiting_previously_left_square_has_penalty();
    test_repeating_directed_move_dies_and_restarts_same_map_blind();
    test_longer_backtrack_then_forward_move_is_allowed();
    test_known_open_edge_to_new_square_has_no_extra_penalty();
    test_max_solution_len_limits_curriculum_distance();
    test_success_generates_next_map_one_step_farther();
    test_failure_retry_does_not_graduate_curriculum();
    test_curriculum_caps_at_board_max();
    test_generation_can_reach_board_max_distance();
    test_timeout_restarts_same_map_with_unknown_wall_memory();
    test_reaching_goal_terminates();
    printf("pathfinder core tests passed\n");
    return 0;
}
