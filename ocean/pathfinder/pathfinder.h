#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if !defined(PATHFINDER_NO_RENDER) && !defined(PUFFER_PERF_NO_RENDER)
#include "raylib.h"
#endif

#define PATHFINDER_ROWS 6
#define PATHFINDER_COLS 6
#define PATHFINDER_VERTICAL_WALLS (PATHFINDER_ROWS * (PATHFINDER_COLS + 1))
#define PATHFINDER_HORIZONTAL_WALLS ((PATHFINDER_ROWS + 1) * PATHFINDER_COLS)
#define PATHFINDER_NUM_WALLS (PATHFINDER_VERTICAL_WALLS + PATHFINDER_HORIZONTAL_WALLS)
#define PATHFINDER_OBS_SIZE (PATHFINDER_NUM_WALLS + 2)
#define PATHFINDER_NUM_ACTIONS 4
#define PATHFINDER_MAX_SOLUTION_LEN ((PATHFINDER_ROWS - 1) + (PATHFINDER_COLS - 1))

#define PATHFINDER_RENDER_TILE 72
#define PATHFINDER_RENDER_MARGIN 40
#define PATHFINDER_RENDER_BOARD_X PATHFINDER_RENDER_MARGIN
#define PATHFINDER_RENDER_BOARD_Y 92
#define PATHFINDER_RENDER_BOARD_SIZE (PATHFINDER_RENDER_TILE * PATHFINDER_COLS)
#define PATHFINDER_RENDER_PANEL_WIDTH 332
#define PATHFINDER_RENDER_WIDTH \
    (PATHFINDER_RENDER_BOARD_X + PATHFINDER_RENDER_BOARD_SIZE + \
        PATHFINDER_RENDER_PANEL_WIDTH + PATHFINDER_RENDER_MARGIN)
#define PATHFINDER_RENDER_HEIGHT \
    (PATHFINDER_RENDER_BOARD_Y + PATHFINDER_RENDER_BOARD_SIZE + 168)

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
#define PATHFINDER_KNOWN_WALL_DEATH_PENALTY -0.05f
#define PATHFINDER_REPEAT_MOVE_DEATH_PENALTY -1.0f
#define PATHFINDER_NEW_CELL_REWARD 0.01f
#define PATHFINDER_REVISIT_PENALTY -0.01f
#define PATHFINDER_IMPOSSIBLE_PENALTY -0.01f
#define PATHFINDER_GOAL_REWARD 1.0f

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success;
    float wall_hits;
    float revisits;
    float known_wall_deaths;
    float repeat_move_deaths;
    float known_walls;
    float known_open_edges;
    float shortest_path_len;
    float agent_path_len;
    float curriculum_level;
    float curriculum_max_solution_len;
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
    int revisit_count;
    int known_wall_death;
    int repeat_move_death;
    int visited_count;
    int known_wall_count;
    int known_open_count;
    int success;
    float episode_return;
    unsigned char visited[PATHFINDER_ROWS][PATHFINDER_COLS];
    unsigned char recent_rows[3];
    unsigned char recent_cols[3];
    int recent_count;
    unsigned char true_walls[PATHFINDER_NUM_WALLS];
    float known_walls[PATHFINDER_NUM_WALLS];
} State;

typedef struct PathfinderClient {
    bool show_truth;
} PathfinderClient;

typedef struct Pathfinder {
    PathfinderClient* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    float branch_prob;
    float loop_prob;
    float extra_entry_prob;
    int min_solution_len;
    int max_solution_len;
    int max_steps;
    int curriculum_level;
    int curriculum_episodes;
    State state;
} Pathfinder;

static inline int pathfinder_clamp_int(int value, int min_value, int max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static inline int pathfinder_curriculum_base_solution_len(const Pathfinder* env) {
    if (env->max_solution_len <= 0) {
        return PATHFINDER_MAX_SOLUTION_LEN;
    }
    return pathfinder_clamp_int(env->max_solution_len, 1, PATHFINDER_MAX_SOLUTION_LEN);
}

static inline int pathfinder_curriculum_max_solution_len(const Pathfinder* env) {
    int max_len = pathfinder_curriculum_base_solution_len(env) + env->curriculum_level;
    return pathfinder_clamp_int(max_len, 1, PATHFINDER_MAX_SOLUTION_LEN);
}

static inline int pathfinder_curriculum_min_solution_len(const Pathfinder* env) {
    int max_len = pathfinder_curriculum_max_solution_len(env);
    return pathfinder_clamp_int(max_len, 1, PATHFINDER_MAX_SOLUTION_LEN);
}

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

static inline bool pathfinder_rand_chance_u8(Pathfinder* env, int threshold,
        unsigned int* samples, int* remaining) {
    if (threshold <= 0) {
        return false;
    }
    if (threshold >= 256) {
        return true;
    }
    if (*remaining == 0) {
        *samples = pathfinder_rand(env);
        *remaining = 4;
    }
    unsigned int sample = *samples & 0xffu;
    *samples >>= 8;
    *remaining -= 1;
    return (int)sample < threshold;
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

static inline void pathfinder_mark_visited(State* s, int row, int col) {
    if (!pathfinder_in_bounds(row, col) || s->visited[row][col]) {
        return;
    }
    s->visited[row][col] = 1;
    s->visited_count++;
}

static inline void pathfinder_reset_move_history(State* s) {
    s->recent_rows[0] = (unsigned char)s->agent_row;
    s->recent_cols[0] = (unsigned char)s->agent_col;
    s->recent_count = 1;
}

static inline void pathfinder_ensure_move_history(State* s) {
    if (s->recent_count <= 0) {
        pathfinder_reset_move_history(s);
    }
}

static inline bool pathfinder_repeats_two_cell_cycle(
        const State* s, int next_row, int next_col) {
    return s->recent_count >= 3 &&
        s->recent_rows[0] == s->agent_row &&
        s->recent_cols[0] == s->agent_col &&
        s->recent_rows[1] == next_row &&
        s->recent_cols[1] == next_col;
}

static inline void pathfinder_record_successful_move(State* s) {
    if (s->recent_count < 3) {
        int idx = s->recent_count++;
        s->recent_rows[idx] = (unsigned char)s->agent_row;
        s->recent_cols[idx] = (unsigned char)s->agent_col;
        return;
    }

    s->recent_rows[0] = s->recent_rows[1];
    s->recent_cols[0] = s->recent_cols[1];
    s->recent_rows[1] = s->recent_rows[2];
    s->recent_cols[1] = s->recent_cols[2];
    s->recent_rows[2] = (unsigned char)s->agent_row;
    s->recent_cols[2] = (unsigned char)s->agent_col;
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

static void pathfinder_update_action_mask(Pathfinder* env) {
    if (env->action_mask == NULL) {
        return;
    }

    for (int action = 0; action < PATHFINDER_NUM_ACTIONS; action++) {
        int d_row;
        int d_col;
        pathfinder_action_delta(action, &d_row, &d_col);
        int next_row = env->state.agent_row + d_row;
        int next_col = env->state.agent_col + d_col;
        int wall = pathfinder_wall_between(
            env->state.agent_row, env->state.agent_col, next_row, next_col);
        env->action_mask[action] =
            (wall >= 0 && env->state.known_walls[wall] == PATHFINDER_WALL) ? 0 : 1;
    }
}

static void pathfinder_update_observations(Pathfinder* env) {
    pathfinder_recount_known(&env->state);
    pathfinder_update_action_mask(env);
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
    int open_threshold = (int)(open_prob * 256.0f);
    int entry_threshold = (int)(env->extra_entry_prob * 256.0f);
    unsigned int samples = 0;
    int remaining = 0;

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS - 1; col++) {
            if (pathfinder_rand_chance_u8(env, open_threshold, &samples, &remaining)) {
                s->true_walls[pathfinder_v_wall(row, col + 1)] = 0;
            }
        }
    }
    for (int row = 0; row < PATHFINDER_ROWS - 1; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            if (pathfinder_rand_chance_u8(env, open_threshold, &samples, &remaining)) {
                s->true_walls[pathfinder_h_wall(row + 1, col)] = 0;
            }
        }
    }
    for (int row = 1; row < PATHFINDER_ROWS; row++) {
        if (pathfinder_rand_chance_u8(env, entry_threshold, &samples, &remaining)) {
            s->true_walls[pathfinder_v_wall(row, 0)] = 0;
        }
    }
}

static void pathfinder_choose_goal_at_distance(Pathfinder* env, int distance) {
    State* s = &env->state;
    int candidates[PATHFINDER_ROWS * PATHFINDER_COLS];
    int count = 0;
    distance = pathfinder_clamp_int(distance, 1, PATHFINDER_MAX_SOLUTION_LEN);

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            if (row == 0 && col == 0) {
                continue;
            }
            if (row + col == distance) {
                candidates[count++] = row * PATHFINDER_COLS + col;
            }
        }
    }

    if (count == 0) {
        s->goal_row = PATHFINDER_ROWS - 1;
        s->goal_col = PATHFINDER_COLS - 1;
        return;
    }

    int cell = candidates[pathfinder_rand(env) % (unsigned int)count];
    s->goal_row = cell / PATHFINDER_COLS;
    s->goal_col = cell % PATHFINDER_COLS;
}

static void pathfinder_generate_maze(Pathfinder* env) {
    State* s = &env->state;
    int min_solution_len = pathfinder_curriculum_min_solution_len(env);
    int max_solution_len = pathfinder_curriculum_max_solution_len(env);
    int span = max_solution_len - min_solution_len + 1;
    int target_len = min_solution_len + (int)(pathfinder_rand(env) % (unsigned int)span);

    pathfinder_init_walls(s);
    pathfinder_choose_goal_at_distance(env, target_len);
    pathfinder_carve_solution(env);
    pathfinder_open_random_edges(env);
    s->shortest_path_len = s->goal_row + s->goal_col;
}

static void pathfinder_update_curriculum(Pathfinder* env, int success) {
    env->curriculum_episodes++;
    if (success && pathfinder_curriculum_max_solution_len(env) < PATHFINDER_MAX_SOLUTION_LEN) {
        env->curriculum_level++;
    }
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

    pathfinder_update_curriculum(env, s->success);

    env->log.perf += success;
    env->log.score += success * efficiency;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.success += success;
    env->log.wall_hits += (float)s->wall_hits;
    env->log.revisits += (float)s->revisit_count;
    env->log.known_wall_deaths += (float)s->known_wall_death;
    env->log.repeat_move_deaths += (float)s->repeat_move_death;
    env->log.known_walls += (float)s->known_wall_count;
    env->log.known_open_edges += (float)s->known_open_count;
    env->log.shortest_path_len += (float)s->shortest_path_len;
    env->log.agent_path_len += (float)s->agent_path_len;
    env->log.curriculum_level += (float)env->curriculum_level;
    env->log.curriculum_max_solution_len += (float)pathfinder_curriculum_max_solution_len(env);
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
    pathfinder_mark_visited(s, s->agent_row, s->agent_col);
    pathfinder_reset_move_history(s);
    pathfinder_update_observations(env);
}

static void pathfinder_reset_attempt(Pathfinder* env) {
    State* s = &env->state;
    s->tick = 0;
    s->agent_row = 0;
    s->agent_col = 0;
    s->agent_path_len = 0;
    s->wall_hits = 0;
    s->revisit_count = 0;
    s->known_wall_death = 0;
    s->repeat_move_death = 0;
    s->visited_count = 0;
    s->success = 0;
    s->episode_return = 0.0f;
    memset(s->visited, 0, sizeof(s->visited));
    pathfinder_reset_known(s);
    pathfinder_mark_visited(s, s->agent_row, s->agent_col);
    pathfinder_reset_move_history(s);
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
                if (was_known) {
                    reward += PATHFINDER_KNOWN_WALL_DEATH_PENALTY;
                    s->known_wall_death = 1;
                    env->terminals[0] = 1.0f;
                }
            } else if (!pathfinder_in_bounds(next_row, next_col)) {
                reward += PATHFINDER_IMPOSSIBLE_PENALTY;
            } else {
                pathfinder_ensure_move_history(s);
                if (pathfinder_repeats_two_cell_cycle(s, next_row, next_col)) {
                    reward += PATHFINDER_REPEAT_MOVE_DEATH_PENALTY;
                    s->repeat_move_death = 1;
                    env->terminals[0] = 1.0f;
                } else {
                    bool revisited = s->visited[next_row][next_col] != 0;
                    s->agent_row = next_row;
                    s->agent_col = next_col;
                    s->agent_path_len++;
                    pathfinder_record_successful_move(s);
                    if (revisited) {
                        s->revisit_count++;
                        reward += PATHFINDER_REVISIT_PENALTY;
                    } else {
                        pathfinder_mark_visited(s, next_row, next_col);
                        reward += PATHFINDER_NEW_CELL_REWARD;
                    }
                    if (s->agent_row == s->goal_row && s->agent_col == s->goal_col) {
                        s->success = 1;
                        reward += PATHFINDER_GOAL_REWARD;
                        env->terminals[0] = 1.0f;
                    }
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
        int solved = s->success;
        add_log(env);
        if (solved) {
            c_reset(env);
        } else {
            pathfinder_reset_attempt(env);
        }
    }
}

void c_close(Pathfinder* env) {
#if !defined(PATHFINDER_NO_RENDER) && !defined(PUFFER_PERF_NO_RENDER)
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
    free(env->client);
    env->client = NULL;
}

#if defined(PATHFINDER_NO_RENDER) || defined(PUFFER_PERF_NO_RENDER)
void c_render(Pathfinder* env) {
    (void)env;
}
#else
static const Color PATHFINDER_BG = {6, 24, 24, 255};
static const Color PATHFINDER_CELL_A = {16, 39, 42, 255};
static const Color PATHFINDER_CELL_B = {19, 46, 49, 255};
static const Color PATHFINDER_GRID = {54, 84, 86, 255};
static const Color PATHFINDER_TEXT = {235, 242, 240, 255};
static const Color PATHFINDER_MUTED = {145, 166, 164, 255};
static const Color PATHFINDER_TRUE_WALL = {88, 96, 99, 255};
static const Color PATHFINDER_UNKNOWN_EDGE = {42, 63, 65, 255};
static const Color PATHFINDER_KNOWN_WALL = {218, 59, 54, 255};
static const Color PATHFINDER_KNOWN_OPEN = {75, 196, 118, 255};
static const Color PATHFINDER_AGENT = {0, 187, 187, 255};
static const Color PATHFINDER_GOAL = {232, 184, 58, 255};
static const Color PATHFINDER_START = {118, 146, 150, 255};
static const Color PATHFINDER_VISITED = {0, 187, 187, 42};

static PathfinderClient* pathfinder_make_client(void) {
    PathfinderClient* client = (PathfinderClient*)calloc(1, sizeof(PathfinderClient));
    client->show_truth = true;
    InitWindow(PATHFINDER_RENDER_WIDTH, PATHFINDER_RENDER_HEIGHT, "PufferLib Pathfinder");
    SetTargetFPS(30);
    return client;
}

static inline int pathfinder_cell_x(int col) {
    return PATHFINDER_RENDER_BOARD_X + col * PATHFINDER_RENDER_TILE;
}

static inline int pathfinder_cell_y(int row) {
    return PATHFINDER_RENDER_BOARD_Y + row * PATHFINDER_RENDER_TILE;
}

static inline Vector2 pathfinder_cell_center(int row, int col) {
    return (Vector2){
        (float)(pathfinder_cell_x(col) + PATHFINDER_RENDER_TILE / 2),
        (float)(pathfinder_cell_y(row) + PATHFINDER_RENDER_TILE / 2)
    };
}

static const char* pathfinder_action_name(int action) {
    if (action == PATHFINDER_ACT_NORTH) return "north";
    if (action == PATHFINDER_ACT_EAST) return "east";
    if (action == PATHFINDER_ACT_SOUTH) return "south";
    if (action == PATHFINDER_ACT_WEST) return "west";
    return "invalid";
}

static void pathfinder_draw_centered_text(const char* text, int cx, int y,
        int font_size, Color color) {
    int width = MeasureText(text, font_size);
    DrawText(text, cx - width / 2, y, font_size, color);
}

static void pathfinder_draw_edge(Pathfinder* env, int wall, Vector2 start, Vector2 end) {
    State* s = &env->state;
    float known = s->known_walls[wall];

    DrawLineEx(start, end, 2.0f, PATHFINDER_UNKNOWN_EDGE);
    if (env->client->show_truth && s->true_walls[wall]) {
        DrawLineEx(start, end, 6.0f, PATHFINDER_TRUE_WALL);
    }

    if (known == PATHFINDER_WALL) {
        DrawLineEx(start, end, 8.0f, PATHFINDER_KNOWN_WALL);
    } else if (known == PATHFINDER_OPEN) {
        DrawLineEx(start, end, 4.0f, PATHFINDER_KNOWN_OPEN);
    }
}

static void pathfinder_draw_board(Pathfinder* env) {
    State* s = &env->state;

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            Color cell_color = ((row + col) & 1) ? PATHFINDER_CELL_A : PATHFINDER_CELL_B;
            DrawRectangle(pathfinder_cell_x(col), pathfinder_cell_y(row),
                PATHFINDER_RENDER_TILE - 1, PATHFINDER_RENDER_TILE - 1, cell_color);
            if (s->visited[row][col]) {
                DrawRectangle(pathfinder_cell_x(col) + 8, pathfinder_cell_y(row) + 8,
                    PATHFINDER_RENDER_TILE - 17, PATHFINDER_RENDER_TILE - 17, PATHFINDER_VISITED);
            }
        }
    }

    DrawRectangleLinesEx((Rectangle){
        (float)PATHFINDER_RENDER_BOARD_X,
        (float)PATHFINDER_RENDER_BOARD_Y,
        (float)PATHFINDER_RENDER_BOARD_SIZE,
        (float)PATHFINDER_RENDER_BOARD_SIZE
    }, 2.0f, PATHFINDER_GRID);

    for (int col = 0; col < PATHFINDER_COLS; col++) {
        char label[2] = {(char)('A' + col), '\0'};
        pathfinder_draw_centered_text(label,
            pathfinder_cell_x(col) + PATHFINDER_RENDER_TILE / 2,
            PATHFINDER_RENDER_BOARD_Y - 28, 20, PATHFINDER_TEXT);
    }
    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        DrawText(TextFormat("%i", row + 1),
            PATHFINDER_RENDER_BOARD_X - 28,
            pathfinder_cell_y(row) + PATHFINDER_RENDER_TILE / 2 - 10,
            20, PATHFINDER_TEXT);
    }

    DrawRectangleLinesEx((Rectangle){
        (float)pathfinder_cell_x(0) + 4.0f,
        (float)pathfinder_cell_y(0) + 4.0f,
        (float)PATHFINDER_RENDER_TILE - 9.0f,
        (float)PATHFINDER_RENDER_TILE - 9.0f
    }, 2.0f, PATHFINDER_START);
    pathfinder_draw_centered_text("A1", pathfinder_cell_x(0) + PATHFINDER_RENDER_TILE / 2,
        pathfinder_cell_y(0) + PATHFINDER_RENDER_TILE - 24, 16, PATHFINDER_MUTED);

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int edge_col = 0; edge_col <= PATHFINDER_COLS; edge_col++) {
            int wall = pathfinder_v_wall(row, edge_col);
            float x = (float)(PATHFINDER_RENDER_BOARD_X + edge_col * PATHFINDER_RENDER_TILE);
            float y0 = (float)(pathfinder_cell_y(row) + 7);
            float y1 = (float)(pathfinder_cell_y(row + 1) - 7);
            pathfinder_draw_edge(env, wall, (Vector2){x, y0}, (Vector2){x, y1});
        }
    }
    for (int edge_row = 0; edge_row <= PATHFINDER_ROWS; edge_row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            int wall = pathfinder_h_wall(edge_row, col);
            float x0 = (float)(pathfinder_cell_x(col) + 7);
            float x1 = (float)(pathfinder_cell_x(col + 1) - 7);
            float y = (float)(PATHFINDER_RENDER_BOARD_Y + edge_row * PATHFINDER_RENDER_TILE);
            pathfinder_draw_edge(env, wall, (Vector2){x0, y}, (Vector2){x1, y});
        }
    }

    if (env->client->show_truth) {
        Vector2 goal = pathfinder_cell_center(s->goal_row, s->goal_col);
        DrawCircleV(goal, 19.0f, PATHFINDER_GOAL);
        pathfinder_draw_centered_text("T", (int)goal.x, (int)goal.y - 10, 22, PATHFINDER_BG);
    }

    Vector2 agent = pathfinder_cell_center(s->agent_row, s->agent_col);
    DrawCircleV(agent, 21.0f, PATHFINDER_AGENT);
    DrawCircleLines((int)agent.x, (int)agent.y, 22.0f, PATHFINDER_TEXT);
    pathfinder_draw_centered_text("P", (int)agent.x, (int)agent.y - 11, 24, PATHFINDER_BG);
}

static void pathfinder_draw_panel(Pathfinder* env) {
    State* s = &env->state;
    int x = PATHFINDER_RENDER_BOARD_X + PATHFINDER_RENDER_BOARD_SIZE + 34;
    int y = PATHFINDER_RENDER_BOARD_Y;
    int action = env->actions == NULL ? -1 : (int)env->actions[0];
    float reward = env->rewards == NULL ? 0.0f : env->rewards[0];
    float terminal = env->terminals == NULL ? 0.0f : env->terminals[0];
    int unknown = PATHFINDER_NUM_WALLS - s->known_wall_count - s->known_open_count;

    DrawText("Pathfinder", x, y, 28, PATHFINDER_TEXT);
    y += 38;
    DrawText(env->client->show_truth ? "View: truth + observation" : "View: observation only",
        x, y, 18, env->client->show_truth ? PATHFINDER_GOAL : PATHFINDER_KNOWN_OPEN);
    y += 34;

    DrawText(TextFormat("Position: %c%i", 'A' + s->agent_col, s->agent_row + 1),
        x, y, 20, PATHFINDER_TEXT);
    y += 26;
    if (env->client->show_truth) {
        DrawText(TextFormat("Target: %c%i", 'A' + s->goal_col, s->goal_row + 1),
            x, y, 20, PATHFINDER_GOAL);
    } else {
        DrawText("Target: hidden", x, y, 20, PATHFINDER_MUTED);
    }
    y += 34;

    DrawText(TextFormat("Tick: %i / %i", s->tick, env->max_steps), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Action: %s", pathfinder_action_name(action)), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Reward: %.3f", reward), x, y, 18,
        reward >= 0.0f ? PATHFINDER_KNOWN_OPEN : PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Return: %.3f", s->episode_return), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Terminal: %.0f", terminal), x, y, 18, PATHFINDER_TEXT);
    y += 34;

    DrawText(TextFormat("Known walls: %i", s->known_wall_count), x, y, 18, PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Known open: %i", s->known_open_count), x, y, 18, PATHFINDER_KNOWN_OPEN);
    y += 24;
    DrawText(TextFormat("Unknown edges: %i", unknown), x, y, 18, PATHFINDER_MUTED);
    y += 34;

    DrawText(TextFormat("Wall hits: %i", s->wall_hits), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Revisits: %i", s->revisit_count), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Known-wall deaths: %.0f", env->log.known_wall_deaths),
        x, y, 18, PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Repeat-move deaths: %.0f", env->log.repeat_move_deaths),
        x, y, 18, PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Shortest path: %i", s->shortest_path_len), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Agent path: %i", s->agent_path_len), x, y, 18, PATHFINDER_TEXT);
    y += 34;

    DrawText(TextFormat("Episodes: %.0f", env->log.n), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Avg success: %.3f", env->log.n > 0.0f ?
        env->log.success / env->log.n : 0.0f), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Curriculum: %i / %i moves", env->curriculum_level,
        pathfinder_curriculum_max_solution_len(env)), x, y, 18, PATHFINDER_GOAL);

    DrawText("Arrows/WASD move  |  R reset", PATHFINDER_RENDER_BOARD_X,
        PATHFINDER_RENDER_HEIGHT - 30, 18, PATHFINDER_MUTED);
    DrawText("TAB view  |  SPACE random  |  ESC quit",
        PATHFINDER_RENDER_BOARD_X + 310, PATHFINDER_RENDER_HEIGHT - 30,
        18, PATHFINDER_MUTED);
}

void c_render(Pathfinder* env) {
    if (!IsWindowReady()) {
        env->client = pathfinder_make_client();
    } else if (env->client == NULL) {
        env->client = (PathfinderClient*)calloc(1, sizeof(PathfinderClient));
        env->client->show_truth = true;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        env->client->show_truth = !env->client->show_truth;
    }

    BeginDrawing();
    ClearBackground(PATHFINDER_BG);
    DrawText("Milton Bradley Pathfinder", PATHFINDER_RENDER_BOARD_X, 26, 30, PATHFINDER_TEXT);
    DrawText("Red = known wall, green = known open, gray = true hidden wall",
        PATHFINDER_RENDER_BOARD_X, 60, 18, PATHFINDER_MUTED);
    pathfinder_draw_board(env);
    pathfinder_draw_panel(env);
    EndDrawing();
}
#endif
