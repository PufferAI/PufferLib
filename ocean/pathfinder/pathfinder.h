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
#define PATHFINDER_MAX_PATH_CELLS (PATHFINDER_ROWS * PATHFINDER_COLS)

#define PATHFINDER_RENDER_MAX_SIDE ((PATHFINDER_ROWS > PATHFINDER_COLS) ? PATHFINDER_ROWS : PATHFINDER_COLS)
#define PATHFINDER_RENDER_TILE (432 / PATHFINDER_RENDER_MAX_SIDE)
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

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success;
    float wins;
    float wall_hits;
    float revisits;
    float known_wall_deaths;
    float repeat_move_deaths;
    float known_walls;
    float known_open_edges;
    float shortest_path_len;
    float agent_path_len;
    float curriculum_level;
    float curriculum_min_solution_len;
    float curriculum_max_solution_len;
    float curriculum_target_len;
    float curriculum_next_target_len;
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
    float step_penalty;
    float new_wall_penalty;
    float known_wall_penalty;
    float known_wall_death_penalty;
    float repeat_move_death_penalty;
    float new_cell_reward;
    float revisit_penalty;
    float impossible_penalty;
    float goal_reward;
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

static inline int curriculum_base_solution_len(const Pathfinder* env) {
    if (env->max_solution_len <= 0) {
        return PATHFINDER_MAX_SOLUTION_LEN;
    }
    return pathfinder_clamp_int(env->max_solution_len, 1, PATHFINDER_MAX_SOLUTION_LEN);
}

static inline int curriculum_max_solution_len(const Pathfinder* env) {
    int max_len = curriculum_base_solution_len(env) + env->curriculum_level;
    return pathfinder_clamp_int(max_len, 1, PATHFINDER_MAX_SOLUTION_LEN);
}

static inline int curriculum_min_solution_len(const Pathfinder* env) {
    int base_max_solution_len = curriculum_base_solution_len(env);
    int base_min_solution_len = pathfinder_clamp_int(
        env->min_solution_len <= 0 ? 1 : env->min_solution_len,
        1,
        base_max_solution_len
    );
    int max_solution_len = pathfinder_clamp_int(
        base_max_solution_len + env->curriculum_level,
        1,
        PATHFINDER_MAX_SOLUTION_LEN
    );
    int min_solution_len = base_min_solution_len + env->curriculum_level;
    return pathfinder_clamp_int(min_solution_len, 1, max_solution_len);
}

static inline int v_wall(int row, int edge_col) {
    return row * (PATHFINDER_COLS + 1) + edge_col;
}

static inline int h_wall(int edge_row, int col) {
    return PATHFINDER_VERTICAL_WALLS + edge_row * PATHFINDER_COLS + col;
}

static inline bool in_bounds(int row, int col) {
    return row >= 0 && row < PATHFINDER_ROWS && col >= 0 && col < PATHFINDER_COLS;
}

static inline unsigned int rand_u32(Pathfinder* env) {
    env->rng = 1664525u * env->rng + 1013904223u;
    return env->rng;
}

static inline float rand_float(Pathfinder* env) {
    return (float)(rand_u32(env) >> 8) / 16777216.0f;
}

static inline bool rand_chance_u8(Pathfinder* env, int threshold,
        unsigned int* samples, int* remaining) {
    if (threshold <= 0) {
        return false;
    }
    if (threshold >= 256) {
        return true;
    }
    if (*remaining == 0) {
        *samples = rand_u32(env);
        *remaining = 4;
    }
    unsigned int sample = *samples & 0xffu;
    *samples >>= 8;
    *remaining -= 1;
    return (int)sample < threshold;
}

static inline int wall_between(int row, int col, int next_row, int next_col) {
    if (next_row == row && next_col == col + 1) {
        return v_wall(row, col + 1);
    }
    if (next_row == row && next_col == col - 1) {
        return v_wall(row, col);
    }
    if (next_col == col && next_row == row + 1) {
        return h_wall(row + 1, col);
    }
    if (next_col == col && next_row == row - 1) {
        return h_wall(row, col);
    }
    return -1;
}

static inline void open_edge(State* s, int row, int col, int next_row, int next_col) {
    int wall = wall_between(row, col, next_row, next_col);
    if (wall >= 0) {
        s->true_walls[wall] = 0;
    }
}

static inline void mark_visited(State* s, int row, int col) {
    if (!in_bounds(row, col) || s->visited[row][col]) {
        return;
    }
    s->visited[row][col] = 1;
    s->visited_count++;
}

static inline void reset_move_history(State* s) {
    s->recent_rows[0] = (unsigned char)s->agent_row;
    s->recent_cols[0] = (unsigned char)s->agent_col;
    s->recent_count = 1;
}

static inline void ensure_move_history(State* s) {
    if (s->recent_count <= 0) {
        reset_move_history(s);
    }
}

static inline bool repeats_two_cell_cycle(
        const State* s, int next_row, int next_col) {
    return s->recent_count >= 3 &&
        s->recent_rows[0] == s->agent_row &&
        s->recent_cols[0] == s->agent_col &&
        s->recent_rows[1] == next_row &&
        s->recent_cols[1] == next_col;
}

static inline void record_successful_move(State* s) {
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

static inline void action_delta(int action, int* d_row, int* d_col) {
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

static int shortest_path(const State* s) {
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
            if (!in_bounds(nr, nc)) {
                continue;
            }
            int wall = wall_between(row, col, nr, nc);
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

static inline bool has_path_to_goal(const State* s) {
    return shortest_path(s) >= 0;
}

static void recount_known(State* s) {
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

static void update_action_mask(Pathfinder* env) {
    if (env->action_mask == NULL) {
        return;
    }

    for (int action = 0; action < PATHFINDER_NUM_ACTIONS; action++) {
        int d_row;
        int d_col;
        action_delta(action, &d_row, &d_col);
        int next_row = env->state.agent_row + d_row;
        int next_col = env->state.agent_col + d_col;
        int wall = wall_between(
            env->state.agent_row, env->state.agent_col, next_row, next_col);
        env->action_mask[action] =
            (wall >= 0 && env->state.known_walls[wall] == PATHFINDER_WALL) ? 0 : 1;
    }
}

static void update_observations(Pathfinder* env) {
    recount_known(&env->state);
    update_action_mask(env);
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

static void reset_known(State* s) {
    for (int i = 0; i < PATHFINDER_NUM_WALLS; i++) {
        s->known_walls[i] = PATHFINDER_UNKNOWN;
    }
    s->known_wall_count = 0;
    s->known_open_count = 0;
}

static void init_walls(State* s) {
    memset(s->true_walls, 1, sizeof(s->true_walls));
    reset_known(s);
    s->true_walls[v_wall(0, 0)] = 0;
}

static inline int pathfinder_rand(Pathfinder* env) {
    return (int)rand_u32(env);
}

static inline void pathfinder_reset_known(State* s) {
    reset_known(s);
}

static inline void pathfinder_mark_visited(State* s, int row, int col) {
    mark_visited(s, row, col);
}

static inline void pathfinder_open_edge(State* s, int row, int col, int next_row, int next_col) {
    open_edge(s, row, col, next_row, next_col);
}

static inline int pathfinder_shortest_path(const State* s) {
    return shortest_path(s);
}

static inline int pathfinder_v_wall(int row, int edge_col) {
    return v_wall(row, edge_col);
}

static inline int pathfinder_h_wall(int edge_row, int col) {
    return h_wall(edge_row, col);
}

static inline int pathfinder_wall_between(int row, int col, int next_row, int next_col) {
    return wall_between(row, col, next_row, next_col);
}

static inline void pathfinder_action_delta(int action, int* d_row, int* d_col) {
    action_delta(action, d_row, d_col);
}

static inline bool pathfinder_has_path_to_goal(const State* s) {
    return has_path_to_goal(s);
}

static inline int pathfinder_curriculum_max_solution_len(const Pathfinder* env) {
    return curriculum_max_solution_len(env);
}

static inline int pathfinder_curriculum_min_solution_len(const Pathfinder* env) {
    return curriculum_min_solution_len(env);
}

static bool carve_solution_recursive(
        Pathfinder* env, int row, int col, int remaining_steps,
        int step, int visited_count, int* path_rows, int* path_cols,
        unsigned char visited[PATHFINDER_ROWS][PATHFINDER_COLS]) {
    if (remaining_steps == 0) {
        return true;
    }

    int max_possible_steps = PATHFINDER_MAX_PATH_CELLS - visited_count;
    if (remaining_steps > max_possible_steps) {
        return false;
    }

    int next_rows[PATHFINDER_NUM_ACTIONS];
    int next_cols[PATHFINDER_NUM_ACTIONS];
    int option_count = 0;

    static const int d_rows[PATHFINDER_NUM_ACTIONS] = {-1, 0, 1, 0};
    static const int d_cols[PATHFINDER_NUM_ACTIONS] = {0, 1, 0, -1};
    for (int action = 0; action < PATHFINDER_NUM_ACTIONS; action++) {
        int nr = row + d_rows[action];
        int nc = col + d_cols[action];
        if (!in_bounds(nr, nc) || visited[nr][nc]) {
            continue;
        }
        next_rows[option_count] = nr;
        next_cols[option_count] = nc;
        option_count++;
    }

    if (option_count == 0) {
        return false;
    }

    for (int i = 0; i < option_count - 1; i++) {
        int swap_idx = i + (int)(rand_u32(env) % (unsigned int)(option_count - i));
        int tmp_row = next_rows[i];
        int tmp_col = next_cols[i];
        next_rows[i] = next_rows[swap_idx];
        next_cols[i] = next_cols[swap_idx];
        next_rows[swap_idx] = tmp_row;
        next_cols[swap_idx] = tmp_col;
    }

    for (int i = 0; i < option_count; i++) {
        int nr = next_rows[i];
        int nc = next_cols[i];

        visited[nr][nc] = 1;
        path_rows[step + 1] = nr;
        path_cols[step + 1] = nc;

        if (carve_solution_recursive(env, nr, nc, remaining_steps - 1, step + 1, visited_count + 1,
                path_rows, path_cols, visited)) {
            return true;
        }
        visited[nr][nc] = 0;
    }
    return false;
}

static bool carve_solution(Pathfinder* env, int target_len) {
    State* s = &env->state;
    if (target_len <= 0) {
        target_len = 1;
    }
    if (target_len > PATHFINDER_MAX_SOLUTION_LEN) {
        target_len = PATHFINDER_MAX_SOLUTION_LEN;
    }

    unsigned char visited[PATHFINDER_ROWS][PATHFINDER_COLS] = {0};
    int path_rows[PATHFINDER_MAX_PATH_CELLS];
    int path_cols[PATHFINDER_MAX_PATH_CELLS];

    visited[0][0] = 1;
    path_rows[0] = 0;
    path_cols[0] = 0;

    if (!carve_solution_recursive(env, 0, 0, target_len, 0, 1,
            path_rows, path_cols, visited)) {
        return false;
    }

    for (int step = 0; step < target_len; step++) {
        open_edge(s, path_rows[step], path_cols[step], path_rows[step + 1], path_cols[step + 1]);
    }

    s->goal_row = path_rows[target_len];
    s->goal_col = path_cols[target_len];
    return true;
}

static int biased_target_len(int min_len, int max_len, Pathfinder* env) {
    int span = max_len - min_len + 1;
    if (span <= 0) {
        return min_len;
    }
    int roll_a = (int)(rand_u32(env) % (unsigned int)span);
    int roll_b = (int)(rand_u32(env) % (unsigned int)span);
    int roll_c = (int)(rand_u32(env) % (unsigned int)span);
    int pick = roll_a;
    if (roll_b > pick) {
        pick = roll_b;
    }
    if (roll_c > pick) {
        pick = roll_c;
    }
    return min_len + pick;
}

static void open_random_edges(Pathfinder* env, int target_len) {
    State* s = &env->state;
    float open_prob = env->branch_prob + env->loop_prob;
    if (open_prob < 0.0f) open_prob = 0.0f;
    if (open_prob > 0.50f) open_prob = 0.50f;
    int open_threshold = (int)(open_prob * 256.0f);
    int entry_threshold = (int)(env->extra_entry_prob * 256.0f);
    unsigned int samples = 0;
    int remaining = 0;

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS - 1; col++) {
            if (rand_chance_u8(env, open_threshold, &samples, &remaining)) {
                int wall = v_wall(row, col + 1);
                if (s->true_walls[wall] == 0) {
                    continue;
                }
                s->true_walls[wall] = 0;
                if (shortest_path(s) < target_len) {
                    s->true_walls[wall] = 1;
                }
            }
        }
    }
    for (int row = 0; row < PATHFINDER_ROWS - 1; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            if (rand_chance_u8(env, open_threshold, &samples, &remaining)) {
                int wall = h_wall(row + 1, col);
                if (s->true_walls[wall] == 0) {
                    continue;
                }
                s->true_walls[wall] = 0;
                if (shortest_path(s) < target_len) {
                    s->true_walls[wall] = 1;
                }
            }
        }
    }
    for (int row = 1; row < PATHFINDER_ROWS; row++) {
        if (rand_chance_u8(env, entry_threshold, &samples, &remaining)) {
            int wall = v_wall(row, 0);
            if (s->true_walls[wall] == 0) {
                continue;
            }
            s->true_walls[wall] = 0;
            if (shortest_path(s) < target_len) {
                s->true_walls[wall] = 1;
            }
        }
    }
}

static void generate_maze(Pathfinder* env) {
    State* s = &env->state;
    int min_solution_len = curriculum_min_solution_len(env);
    int max_solution_len = curriculum_max_solution_len(env);
    int target_len = biased_target_len(min_solution_len, max_solution_len, env);

    init_walls(s);
    bool carved = carve_solution(env, target_len);
    if (!carved) {
        target_len = min_solution_len;
        carved = carve_solution(env, target_len);
    }

    if (!carved) {
        target_len = 1;
        carve_solution(env, target_len);
    }

    open_random_edges(env, target_len);
    s->shortest_path_len = shortest_path(s);
    if (s->shortest_path_len < 0) {
        s->shortest_path_len = target_len;
    }
}

static void update_curriculum(Pathfinder* env, int success) {
    env->curriculum_episodes++;
    if (success && curriculum_max_solution_len(env) < PATHFINDER_MAX_SOLUTION_LEN) {
        env->curriculum_level++;
    }
}

void add_log(Pathfinder* env) {
    State* s = &env->state;
    float success = (float)s->success;
    int current_curriculum_level = env->curriculum_level;
    int current_curriculum_min_solution_len = curriculum_min_solution_len(env);
    int current_curriculum_max_solution_len = curriculum_max_solution_len(env);
    int next_curriculum_max_solution_len = pathfinder_clamp_int(
        current_curriculum_max_solution_len + (s->success ? 1 : 0),
        1,
        PATHFINDER_MAX_SOLUTION_LEN
    );
    float efficiency = 0.0f;
    if (s->success && s->agent_path_len > 0 && s->shortest_path_len > 0) {
        efficiency = (float)s->shortest_path_len / (float)s->agent_path_len;
        if (efficiency > 1.0f) {
            efficiency = 1.0f;
        }
    }

    update_curriculum(env, s->success);

    env->log.perf += success;
    env->log.score += success * efficiency;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.success += success;
    env->log.wins += success;
    env->log.wall_hits += (float)s->wall_hits;
    env->log.revisits += (float)s->revisit_count;
    env->log.known_wall_deaths += (float)s->known_wall_death;
    env->log.repeat_move_deaths += (float)s->repeat_move_death;
    env->log.known_walls += (float)s->known_wall_count;
    env->log.known_open_edges += (float)s->known_open_count;
    env->log.shortest_path_len += (float)s->shortest_path_len;
    env->log.agent_path_len += (float)s->agent_path_len;
    env->log.curriculum_level += (float)current_curriculum_level;
    env->log.curriculum_min_solution_len += (float)current_curriculum_min_solution_len;
    env->log.curriculum_max_solution_len += (float)current_curriculum_max_solution_len;
    env->log.curriculum_target_len += (float)current_curriculum_max_solution_len;
    env->log.curriculum_next_target_len += (float)next_curriculum_max_solution_len;
    env->log.n += 1.0f;
}

void refresh_state(Pathfinder* env) {
    update_observations(env);
}

void init(Pathfinder* env) {
    if (env->num_agents == 0) {
        env->num_agents = 1;
    }
    if (env->branch_prob == 0.0f) {
        env->branch_prob = 0.10f;
    }
    if (env->loop_prob == 0.0f) {
        env->loop_prob = 0.03f;
    }
    if (env->min_solution_len == 0) {
        env->min_solution_len = 1;
    }
    env->step_penalty = env->step_penalty == 0.0f ? -0.001f : env->step_penalty;
    env->new_wall_penalty = env->new_wall_penalty == 0.0f ? 0.0f : env->new_wall_penalty;
    env->known_wall_penalty = env->known_wall_penalty == 0.0f ? -0.01f : env->known_wall_penalty;
    env->known_wall_death_penalty = env->known_wall_death_penalty == 0.0f ? -0.05f : env->known_wall_death_penalty;
    env->repeat_move_death_penalty = env->repeat_move_death_penalty == 0.0f ? -1.0f : env->repeat_move_death_penalty;
    env->new_cell_reward = env->new_cell_reward == 0.0f ? 0.01f : env->new_cell_reward;
    env->revisit_penalty = env->revisit_penalty == 0.0f ? -0.01f : env->revisit_penalty;
    env->impossible_penalty = env->impossible_penalty == 0.0f ? -1.0f : env->impossible_penalty;
    env->goal_reward = env->goal_reward == 0.0f ? 1.0f : env->goal_reward;
    if (env->max_steps == 0) {
        env->max_steps = 128;
    }
}

void c_reset(Pathfinder* env) {
    State* s = &env->state;
    memset(s, 0, sizeof(*s));
    s->agent_row = 0;
    s->agent_col = 0;
    generate_maze(env);
    mark_visited(s, s->agent_row, s->agent_col);
    reset_move_history(s);
    update_observations(env);
}

static void reset_attempt(Pathfinder* env) {
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
    reset_known(s);
    mark_visited(s, s->agent_row, s->agent_col);
    reset_move_history(s);
    update_observations(env);
}

static void reveal_wall(Pathfinder* env, int wall) {
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

    float reward = env->step_penalty;
    int action = (int)env->actions[0];
    if (action < 0 || action >= PATHFINDER_NUM_ACTIONS) {
        reward += env->impossible_penalty;
        env->terminals[0] = 1.0f;
    } else {
        int d_row;
        int d_col;
        action_delta(action, &d_row, &d_col);
        int next_row = s->agent_row + d_row;
        int next_col = s->agent_col + d_col;
        int wall = wall_between(s->agent_row, s->agent_col, next_row, next_col);

        if (wall < 0) {
            reward += env->impossible_penalty;
            env->terminals[0] = 1.0f;
        } else {
            bool was_known = s->known_walls[wall] != PATHFINDER_UNKNOWN;
            reveal_wall(env, wall);

            if (s->true_walls[wall]) {
                s->wall_hits++;
                reward += was_known ? env->known_wall_penalty : env->new_wall_penalty;
                if (was_known) {
                    reward += env->known_wall_death_penalty;
                    s->known_wall_death = 1;
                    env->terminals[0] = 1.0f;
                }
            } else if (!in_bounds(next_row, next_col)) {
                reward += env->impossible_penalty;
            } else {
                ensure_move_history(s);
                if (repeats_two_cell_cycle(s, next_row, next_col)) {
                    reward += env->repeat_move_death_penalty;
                    s->repeat_move_death = 1;
                    env->terminals[0] = 1.0f;
                } else {
                    bool revisited = s->visited[next_row][next_col] != 0;
                    s->agent_row = next_row;
                    s->agent_col = next_col;
                    s->agent_path_len++;
                    record_successful_move(s);
                    if (revisited) {
                        s->revisit_count++;
                        reward += env->revisit_penalty;
                    } else {
                        mark_visited(s, next_row, next_col);
                        reward += env->new_cell_reward;
                    }
                    if (s->agent_row == s->goal_row && s->agent_col == s->goal_col) {
                        s->success = 1;
                        reward += env->goal_reward;
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
    update_observations(env);

    if (env->terminals[0]) {
        int solved = s->success;
        add_log(env);
        if (solved) {
            c_reset(env);
        } else {
            reset_attempt(env);
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

static PathfinderClient* make_client(void) {
    PathfinderClient* client = (PathfinderClient*)calloc(1, sizeof(PathfinderClient));
    client->show_truth = true;
    InitWindow(PATHFINDER_RENDER_WIDTH, PATHFINDER_RENDER_HEIGHT, "PufferLib Pathfinder");
    SetTargetFPS(30);
    return client;
}

static inline int cell_x(int col) {
    return PATHFINDER_RENDER_BOARD_X + col * PATHFINDER_RENDER_TILE;
}

static inline int cell_y(int row) {
    return PATHFINDER_RENDER_BOARD_Y + row * PATHFINDER_RENDER_TILE;
}

static inline Vector2 cell_center(int row, int col) {
    return (Vector2){
        (float)(cell_x(col) + PATHFINDER_RENDER_TILE / 2),
        (float)(cell_y(row) + PATHFINDER_RENDER_TILE / 2)
    };
}

static const char* action_name(int action) {
    if (action == PATHFINDER_ACT_NORTH) return "north";
    if (action == PATHFINDER_ACT_EAST) return "east";
    if (action == PATHFINDER_ACT_SOUTH) return "south";
    if (action == PATHFINDER_ACT_WEST) return "west";
    return "invalid";
}

static void draw_centered_text(const char* text, int cx, int y,
        int font_size, Color color) {
    int width = MeasureText(text, font_size);
    DrawText(text, cx - width / 2, y, font_size, color);
}

static void draw_edge(Pathfinder* env, int wall, Vector2 start, Vector2 end) {
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

static void draw_board(Pathfinder* env) {
    State* s = &env->state;

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            Color cell_color = ((row + col) & 1) ? PATHFINDER_CELL_A : PATHFINDER_CELL_B;
            DrawRectangle(cell_x(col), cell_y(row),
                PATHFINDER_RENDER_TILE - 1, PATHFINDER_RENDER_TILE - 1, cell_color);
            if (s->visited[row][col]) {
                DrawRectangle(cell_x(col) + 8, cell_y(row) + 8,
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
        draw_centered_text(label,
            cell_x(col) + PATHFINDER_RENDER_TILE / 2,
            PATHFINDER_RENDER_BOARD_Y - 28, 20, PATHFINDER_TEXT);
    }
    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        DrawText(TextFormat("%i", row + 1),
            PATHFINDER_RENDER_BOARD_X - 28,
            cell_y(row) + PATHFINDER_RENDER_TILE / 2 - 10,
            20, PATHFINDER_TEXT);
    }

    DrawRectangleLinesEx((Rectangle){
        (float)cell_x(0) + 4.0f,
        (float)cell_y(0) + 4.0f,
        (float)PATHFINDER_RENDER_TILE - 9.0f,
        (float)PATHFINDER_RENDER_TILE - 9.0f
    }, 2.0f, PATHFINDER_START);
    draw_centered_text("A1", cell_x(0) + PATHFINDER_RENDER_TILE / 2,
        cell_y(0) + PATHFINDER_RENDER_TILE - 24, 16, PATHFINDER_MUTED);

    for (int row = 0; row < PATHFINDER_ROWS; row++) {
        for (int edge_col = 0; edge_col <= PATHFINDER_COLS; edge_col++) {
            int wall = v_wall(row, edge_col);
            float x = (float)(PATHFINDER_RENDER_BOARD_X + edge_col * PATHFINDER_RENDER_TILE);
            float y0 = (float)(cell_y(row) + 7);
            float y1 = (float)(cell_y(row + 1) - 7);
            draw_edge(env, wall, (Vector2){x, y0}, (Vector2){x, y1});
        }
    }
    for (int edge_row = 0; edge_row <= PATHFINDER_ROWS; edge_row++) {
        for (int col = 0; col < PATHFINDER_COLS; col++) {
            int wall = h_wall(edge_row, col);
            float x0 = (float)(cell_x(col) + 7);
            float x1 = (float)(cell_x(col + 1) - 7);
            float y = (float)(PATHFINDER_RENDER_BOARD_Y + edge_row * PATHFINDER_RENDER_TILE);
            draw_edge(env, wall, (Vector2){x0, y}, (Vector2){x1, y});
        }
    }

    if (env->client->show_truth) {
        Vector2 goal = cell_center(s->goal_row, s->goal_col);
        DrawCircleV(goal, 19.0f, PATHFINDER_GOAL);
        draw_centered_text("T", (int)goal.x, (int)goal.y - 10, 22, PATHFINDER_BG);
    }

    Vector2 agent = cell_center(s->agent_row, s->agent_col);
    DrawCircleV(agent, 21.0f, PATHFINDER_AGENT);
    DrawCircleLines((int)agent.x, (int)agent.y, 22.0f, PATHFINDER_TEXT);
    draw_centered_text("P", (int)agent.x, (int)agent.y - 11, 24, PATHFINDER_BG);
}

static void draw_panel(Pathfinder* env) {
    State* s = &env->state;
    int x = PATHFINDER_RENDER_BOARD_X + PATHFINDER_RENDER_BOARD_SIZE + 34;
    int y = PATHFINDER_RENDER_BOARD_Y;
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

    y += 10;

    DrawText(TextFormat("Wall deaths: %i", s->wall_hits), x, y, 18, PATHFINDER_TEXT);
    y += 24;
    DrawText(TextFormat("Known-wall deaths: %.0f", env->log.known_wall_deaths),
        x, y, 18, PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Repeat-move deaths: %.0f", env->log.repeat_move_deaths),
        x, y, 18, PATHFINDER_KNOWN_WALL);
    y += 24;
    DrawText(TextFormat("Wins: %.0f", env->log.wins),
        x, y, 18, PATHFINDER_KNOWN_OPEN);

    DrawText("Arrows/WASD move  |  R reset", PATHFINDER_RENDER_BOARD_X,
        PATHFINDER_RENDER_HEIGHT - 30, 18, PATHFINDER_MUTED);
    DrawText("TAB view  |  SPACE random  |  ESC quit",
        PATHFINDER_RENDER_BOARD_X + 310, PATHFINDER_RENDER_HEIGHT - 30,
        18, PATHFINDER_MUTED);
}

void c_render(Pathfinder* env) {
    if (!IsWindowReady()) {
        env->client = make_client();
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
    draw_board(env);
    draw_panel(env);
    EndDrawing();
}
#endif
