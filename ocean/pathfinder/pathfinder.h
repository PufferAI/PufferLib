#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define PATHFINDER_ROWS 6
#define PATHFINDER_COLS 6
#define PATHFINDER_VERTICAL_WALLS (PATHFINDER_ROWS * (PATHFINDER_COLS + 1))
#define PATHFINDER_HORIZONTAL_WALLS ((PATHFINDER_ROWS + 1) * PATHFINDER_COLS)
#define PATHFINDER_NUM_WALLS (PATHFINDER_VERTICAL_WALLS + PATHFINDER_HORIZONTAL_WALLS)
#define PATHFINDER_OBS_SIZE (PATHFINDER_NUM_WALLS + 2)
#define PATHFINDER_NUM_ACTIONS 4

#define PATHFINDER_ACT_NORTH 0
#define PATHFINDER_ACT_EAST 1
#define PATHFINDER_ACT_SOUTH 2
#define PATHFINDER_ACT_WEST 3

#define PATHFINDER_UNKNOWN -1.0f
#define PATHFINDER_OPEN 0.0f
#define PATHFINDER_WALL 1.0f

#define PATHFINDER_STEP_PENALTY -0.001f
#define PATHFINDER_NEW_WALL_PENALTY 0.0f
#define PATHFINDER_KNOWN_WALL_PENALTY -0.01f
#define PATHFINDER_IMPOSSIBLE_PENALTY -0.01f
#define PATHFINDER_GOAL_REWARD 1.0f

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success;
    float wall_hits;
    float known_walls;
    float known_open_edges;
    float shortest_path_len;
    float agent_path_len;
    float n;
} Log;

typedef struct State {
    int tick;
    int agent_row;
    int agent_col;
    int goal_row;
    int goal_col;
    int shortest_path_len;
    int agent_path_len;
    int wall_hits;
    int known_wall_count;
    int known_open_count;
    int success;
    float episode_return;
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    float known_walls[PATHFINDER_NUM_WALLS];
} State;

typedef struct Pathfinder {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    float branch_prob;
    float loop_prob;
    float extra_entry_prob;
    int min_solution_len;
    int max_solution_len;
    int max_steps;
    State state;
} Pathfinder;

static inline int pathfinder_v_wall(int row, int edge_col) {
    return row * (PATHFINDER_COLS + 1) + edge_col;
}

static inline int pathfinder_h_wall(int edge_row, int col) {
    return PATHFINDER_VERTICAL_WALLS + edge_row * PATHFINDER_COLS + col;
}

static inline bool pathfinder_in_bounds(int row, int col) {
    return row >= 0 && row < PATHFINDER_ROWS && col >= 0 && col < PATHFINDER_COLS;
}

static inline unsigned int pathfinder_rand(Pathfinder* env) {
    env->rng = 1664525u * env->rng + 1013904223u;
    return env->rng;
}

static inline float pathfinder_rand_float(Pathfinder* env) {
    return (float)(pathfinder_rand(env) >> 8) / 16777216.0f;
}

static inline int pathfinder_wall_between(int row, int col, int next_row, int next_col) {
    if (next_row == row && next_col == col + 1) {
        return pathfinder_v_wall(row, col + 1);
    }
    if (next_row == row && next_col == col - 1) {
        return pathfinder_v_wall(row, col);
    }
    if (next_col == col && next_row == row + 1) {
        return pathfinder_h_wall(row + 1, col);
    }
    if (next_col == col && next_row == row - 1) {
        return pathfinder_h_wall(row, col);
    }
    return -1;
}

static inline void pathfinder_open_edge(State* s, int row, int col, int next_row, int next_col) {
    int wall = pathfinder_wall_between(row, col, next_row, next_col);
    if (wall >= 0) {
        s->true_walls[wall] = 0;
    }
}

static inline void pathfinder_action_delta(int action, int* d_row, int* d_col) {
    *d_row = 0;
    *d_col = 0;
    if (action == PATHFINDER_ACT_NORTH) {
        *d_row = -1;
    } else if (action == PATHFINDER_ACT_EAST) {
        *d_col = 1;
    } else if (action == PATHFINDER_ACT_SOUTH) {
        *d_row = 1;
    } else if (action == PATHFINDER_ACT_WEST) {
        *d_col = -1;
    }
}

static int pathfinder_shortest_path(const State* s) {
    int dist[PATHFINDER_ROWS][PATHFINDER_COLS];
    int queue[PATHFINDER_ROWS * PATHFINDER_COLS];
    memset(dist, -1, sizeof(dist));

    int head = 0;
    int tail = 0;
    dist[0][0] = 0;
    queue[tail++] = 0;

    static const int d_rows[4] = {-1, 0, 1, 0};
    static const int d_cols[4] = {0, 1, 0, -1};
    while (head < tail) {
        int cell = queue[head++];
        int row = cell / PATHFINDER_COLS;
        int col = cell % PATHFINDER_COLS;
        if (row == s->goal_row && col == s->goal_col) {
            return dist[row][col];
        }

        for (int action = 0; action < PATHFINDER_NUM_ACTIONS; action++) {
            int nr = row + d_rows[action];
            int nc = col + d_cols[action];
            if (!pathfinder_in_bounds(nr, nc)) {
                continue;
            }
            int wall = pathfinder_wall_between(row, col, nr, nc);
            if (wall < 0 || s->true_walls[wall]) {
                continue;
            }
            if (dist[nr][nc] >= 0) {
                continue;
            }
            dist[nr][nc] = dist[row][col] + 1;
            queue[tail++] = nr * PATHFINDER_COLS + nc;
        }
    }

    return -1;
}

static inline bool pathfinder_has_path_to_goal(const State* s) {
    return pathfinder_shortest_path(s) >= 0;
}

static void pathfinder_recount_known(State* s) {
    s->known_wall_count = 0;
    s->known_open_count = 0;
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        if (s->known_walls[i] == PATHFINDER_WALL) {
            s->known_wall_count++;
        } else if (s->known_walls[i] == PATHFINDER_OPEN) {
            s->known_open_count++;
        }
    }
}

static void pathfinder_update_observations(Pathfinder* env) {
    pathfinder_recount_known(&env->state);
    if (env->observations == NULL) {
        return;
    }

    memcpy(env->observations, env->state.known_walls,
        sizeof(float) * PATHFINDER_NUM_WALLS);
    env->observations[PATHFINDER_NUM_WALLS] =
        (float)env->state.agent_col / (float)(PATHFINDER_COLS - 1);
    env->observations[PATHFINDER_NUM_WALLS + 1] =
        (float)env->state.agent_row / (float)(PATHFINDER_ROWS - 1);
}

static void pathfinder_reset_known(State* s) {
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        s->known_walls[i] = PATHFINDER_UNKNOWN;
    }
    s->known_wall_count = 0;
    s->known_open_count = 0;
}

static void pathfinder_init_walls(State* s) {
    memset(s->true_walls, 1, sizeof(s->true_walls));
    pathfinder_reset_known(s);
    s->true_walls[pathfinder_v_wall(0, 0)] = 0;
}

static void pathfinder_carve_solution(Pathfinder* env) {
    State* s = &env->state;
    int row = 0;
    int col = 0;

    while (row != s->goal_row || col != s->goal_col) {
        bool can_row = row != s->goal_row;
        bool can_col = col != s->goal_col;
        bool step_row = can_row && (!can_col || (pathfinder_rand(env) & 1u));

        int next_row = row;
        int next_col = col;
        if (step_row) {
            next_row += (s->goal_row > row) ? 1 : -1;
        } else {
            next_col += (s->goal_col > col) ? 1 : -1;
        }

        pathfinder_open_edge(s, row, col, next_row, next_col);
        row = next_row;
        col = next_col;
    }
}

static void pathfinder_open_random_edges(Pathfinder* env) {
    State* s = &env->state;
    float open_prob = env->branch_prob + env->loop_prob;
    if (open_prob < 0.0f) open_prob = 0.0f;
    if (open_prob > 0.95f) open_prob = 0.95f;

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS - 1; col++) {
            if (pathfinder_rand_float(env) < open_prob) {
                pathfinder_open_edge(s, row, col, row, col + 1);
            }
        }
    }
    for (int row = 0; row < PATHFINDER_ROWS - 1; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            if (pathfinder_rand_float(env) < open_prob) {
                pathfinder_open_edge(s, row, col, row + 1, col);
            }
        }
    }
    for (int row = 1; row < PATHFINDER_ROWS; row++) {
        if (pathfinder_rand_float(env) < env->extra_entry_prob) {
            s->true_walls[pathfinder_v_wall(row, 0)] = 0;
        }
    }
}

static void pathfinder_generate_maze(Pathfinder* env) {
    State* s = &env->state;
    int min_solution_len = env->min_solution_len < 1 ? 1 : env->min_solution_len;
    int max_solution_len = env->max_solution_len;
    if (max_solution_len > 0 && max_solution_len < min_solution_len) {
        max_solution_len = min_solution_len;
    }

    for (int attempt = 0; attempt < 128; attempt++) {
        pathfinder_init_walls(s);

        do {
            s->goal_row = (int)(pathfinder_rand(env) % PATHFINDER_ROWS);
            s->goal_col = (int)(pathfinder_rand(env) % PATHFINDER_COLS);
        } while (s->goal_row == 0 && s->goal_col == 0);

        pathfinder_carve_solution(env);
        pathfinder_open_random_edges(env);
        s->shortest_path_len = pathfinder_shortest_path(s);
        if (s->shortest_path_len >= min_solution_len &&
                (max_solution_len <= 0 || s->shortest_path_len <= max_solution_len)) {
            return;
        }
    }

    pathfinder_init_walls(s);
    s->goal_row = 0;
    int fallback_len = min_solution_len;
    if (max_solution_len > 0 && fallback_len > max_solution_len) {
        fallback_len = max_solution_len;
    }
    s->goal_col = fallback_len < PATHFINDER_COLS ? fallback_len : PATHFINDER_COLS - 1;
    for (int col = 0; col < s->goal_col; col++) {
        pathfinder_open_edge(s, 0, col, 0, col + 1);
    }
    s->shortest_path_len = pathfinder_shortest_path(s);
}

void add_log(Pathfinder* env) {
    State* s = &env->state;
    float success = (float)s->success;
    float efficiency = 0.0f;
    if (s->success && s->agent_path_len > 0 && s->shortest_path_len > 0) {
        efficiency = (float)s->shortest_path_len / (float)s->agent_path_len;
        if (efficiency > 1.0f) {
            efficiency = 1.0f;
        }
    }

    env->log.perf += success;
    env->log.score += success * efficiency;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.success += success;
    env->log.wall_hits += (float)s->wall_hits;
    env->log.known_walls += (float)s->known_wall_count;
    env->log.known_open_edges += (float)s->known_open_count;
    env->log.shortest_path_len += (float)s->shortest_path_len;
    env->log.agent_path_len += (float)s->agent_path_len;
    env->log.n += 1.0f;
}

void refresh_state(Pathfinder* env) {
    pathfinder_update_observations(env);
}

void init(Pathfinder* env) {
    if (env->num_agents == 0) {
        env->num_agents = 1;
    }
    if (env->branch_prob == 0.0f) {
        env->branch_prob = 0.35f;
    }
    if (env->loop_prob == 0.0f) {
        env->loop_prob = 0.10f;
    }
    if (env->min_solution_len == 0) {
        env->min_solution_len = 1;
    }
    if (env->max_steps == 0) {
        env->max_steps = 128;
    }
}

void c_reset(Pathfinder* env) {
    State* s = &env->state;
    memset(s, 0, sizeof(*s));
    s->agent_row = 0;
    s->agent_col = 0;
    pathfinder_generate_maze(env);
    pathfinder_update_observations(env);
}

static void pathfinder_reveal_wall(Pathfinder* env, int wall) {
    State* s = &env->state;
    if (s->known_walls[wall] != PATHFINDER_UNKNOWN) {
        return;
    }
    s->known_walls[wall] = s->true_walls[wall] ? PATHFINDER_WALL : PATHFINDER_OPEN;
}

void c_step(Pathfinder* env) {
    State* s = &env->state;
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;
    s->tick++;

    float reward = PATHFINDER_STEP_PENALTY;
    int action = (int)env->actions[0];
    if (action < 0 || action >= PATHFINDER_NUM_ACTIONS) {
        reward += PATHFINDER_IMPOSSIBLE_PENALTY;
    } else {
        int d_row;
        int d_col;
        pathfinder_action_delta(action, &d_row, &d_col);
        int next_row = s->agent_row + d_row;
        int next_col = s->agent_col + d_col;
        int wall = pathfinder_wall_between(s->agent_row, s->agent_col, next_row, next_col);

        if (wall < 0) {
            reward += PATHFINDER_IMPOSSIBLE_PENALTY;
        } else {
            bool was_known = s->known_walls[wall] != PATHFINDER_UNKNOWN;
            pathfinder_reveal_wall(env, wall);

            if (s->true_walls[wall]) {
                s->wall_hits++;
                reward += was_known ? PATHFINDER_KNOWN_WALL_PENALTY : PATHFINDER_NEW_WALL_PENALTY;
            } else if (!pathfinder_in_bounds(next_row, next_col)) {
                reward += PATHFINDER_IMPOSSIBLE_PENALTY;
            } else {
                s->agent_row = next_row;
                s->agent_col = next_col;
                s->agent_path_len++;
                if (s->agent_row == s->goal_row && s->agent_col == s->goal_col) {
                    s->success = 1;
                    reward += PATHFINDER_GOAL_REWARD;
                    env->terminals[0] = 1.0f;
                }
            }
        }
    }

    if (s->tick >= env->max_steps && env->terminals[0] == 0.0f) {
        env->terminals[0] = 1.0f;
    }

    env->rewards[0] = reward;
    s->episode_return += reward;
    pathfinder_update_observations(env);

    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
    }
}

void c_close(Pathfinder* env) {
    (void)env;
}

void c_render(Pathfinder* env) {
    (void)env;
}
