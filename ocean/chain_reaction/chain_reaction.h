/* Chain Reaction env core */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define CHAINENV_MAX_ROWS 9
#define CHAINENV_MAX_COLS 9
#define CHAINENV_MAX_CELLS (CHAINENV_MAX_ROWS * CHAINENV_MAX_COLS)
#define CHAINENV_MAX_ACTIONS (CHAINENV_MAX_CELLS + 1)
#define CHAINENV_NOOP_ACTION CHAINENV_MAX_CELLS
#define CHAINENV_MAX_SLOTS 2
#define CHAINENV_OBS_CHANNELS 4
#define CHAINENV_GLOBAL_OBS 5
#define CHAINENV_OBS_SIZE (CHAINENV_MAX_CELLS * CHAINENV_OBS_CHANNELS + CHAINENV_GLOBAL_OBS)
#define CHAINENV_MAX_RENDER_WAVES 256
#define CHAINENV_PLAYER_RED 1
#define CHAINENV_PLAYER_GREEN -1
#define CHAINENV_OPPONENT_RANDOM 0
#define CHAINENV_OPPONENT_HEURISTIC 1
#define CHAINENV_MAX_RENDER_TRANSFERS 768
#define CHAINENV_MAX_RENDER_BURSTS 256
#define CHAINENV_MAX_SNAPSHOTS (2 * CHAINENV_MAX_RENDER_WAVES + 8)
#define CHAINENV_BURST_DURATION 0.36f
#define CHAINENV_TRANSFER_DURATION 0.28f
#define CHAINENV_WAVE_DELAY 0.10f
#define CHAINENV_PLACEMENT_DURATION 0.11f
#define CHAINENV_TURN_GAP 0.08f
#define CHAINENV_RED_ORBIT_SPEED 0.90f
#define CHAINENV_GREEN_ORBIT_SPEED -0.78f

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float chain_bursts;
    float invalid_rate;
    float slot_0_score;
    float slot_1_score;
    float draw_rate;
    float n;
};

typedef struct OrbTransferAnim OrbTransferAnim;
struct OrbTransferAnim {
    int src_row;
    int src_col;
    int dst_row;
    int dst_col;
    int color;
    float delay;
    float duration;
};

typedef struct BurstAnim BurstAnim;
struct BurstAnim {
    int row;
    int col;
    int color;
    float delay;
    float duration;
};

typedef struct Client Client;
struct Client {
    int screen_width;
    int screen_height;
    float board_x;
    float board_y;
    float cell_size;
    float animation_clock;
    float animation_total;
    float invalid_hint_time;
    OrbTransferAnim transfers[CHAINENV_MAX_RENDER_TRANSFERS];
    BurstAnim bursts[CHAINENV_MAX_RENDER_BURSTS];
    int base_owner[CHAINENV_MAX_CELLS];
    int base_orbs[CHAINENV_MAX_CELLS];
    int snapshot_owner[CHAINENV_MAX_SNAPSHOTS][CHAINENV_MAX_CELLS];
    int snapshot_orbs[CHAINENV_MAX_SNAPSHOTS][CHAINENV_MAX_CELLS];
    float snapshot_time[CHAINENV_MAX_SNAPSHOTS];
    int snapshot_count;
    int transfer_count;
    int burst_count;
};

typedef struct ResolveStats ResolveStats;
struct ResolveStats {
    int bursts;
    int waves;
    int transfers;
    float end_time;
};

typedef struct ChainEnv ChainEnv;
struct ChainEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    Log log;
    Client* client;

    int rows;
    int cols;
    int max_steps;
    int opponent_policy;
    float win_reward;
    float loss_reward;
    float invalid_move_reward;

    int owner[CHAINENV_MAX_CELLS];
    int orbs[CHAINENV_MAX_CELLS];
    int move_count;
    int tick;
    int end_game;
    int turns_taken[2];
    int current_player;
    int player_for_slot[CHAINENV_MAX_SLOTS];
    int slot_for_player[CHAINENV_MAX_SLOTS];
    int last_player_action;
    int last_env_action;
    int last_chain_bursts;
    int last_chain_waves;
    int winner;
    unsigned int rng;
};

static inline int chainenv_slot(int row, int col) {
    return row * CHAINENV_MAX_COLS + col;
}

static inline int chainenv_player_index(int player) {
    return (player == CHAINENV_PLAYER_RED) ? 0 : 1;
}

static inline bool chainenv_is_selfplay(const ChainEnv* env) {
    return env->num_agents > 1;
}

static inline int chainenv_slot_for_player(const ChainEnv* env, int player) {
    return env->slot_for_player[chainenv_player_index(player)];
}

static inline int chainenv_player_for_slot(const ChainEnv* env, int slot) {
    return env->player_for_slot[slot];
}

static inline void chainenv_set_slot_players(ChainEnv* env, int slot0_player, int slot1_player) {
    env->player_for_slot[0] = slot0_player;
    env->player_for_slot[1] = slot1_player;
    env->slot_for_player[chainenv_player_index(slot0_player)] = 0;
    env->slot_for_player[chainenv_player_index(slot1_player)] = 1;
}

static inline void chainenv_copy_board(
        int* dst_owner, int* dst_orbs, const int* src_owner, const int* src_orbs) {
    memcpy(dst_owner, src_owner, sizeof(int) * CHAINENV_MAX_CELLS);
    memcpy(dst_orbs, src_orbs, sizeof(int) * CHAINENV_MAX_CELLS);
}

static inline bool chainenv_cell_active(const ChainEnv* env, int row, int col) {
    return row >= 0 && row < env->rows && col >= 0 && col < env->cols;
}

static inline bool chainenv_board_idx_active(const ChainEnv* env, int idx) {
    int row = idx / CHAINENV_MAX_COLS;
    int col = idx % CHAINENV_MAX_COLS;
    return chainenv_cell_active(env, row, col);
}

static inline int chainenv_action_to_board_idx(const ChainEnv* env, int action) {
    if (action < 0 || action >= env->rows * env->cols) {
        return -1;
    }

    int row = action / env->cols;
    int col = action % env->cols;
    return chainenv_slot(row, col);
}

static inline int chainenv_board_idx_to_action(const ChainEnv* env, int board_idx) {
    int row = board_idx / CHAINENV_MAX_COLS;
    int col = board_idx % CHAINENV_MAX_COLS;
    if (!chainenv_cell_active(env, row, col)) {
        return -1;
    }

    return row * env->cols + col;
}

static inline int chainenv_critical_mass(const ChainEnv* env, int row, int col) {
    int mass = 0;
    mass += chainenv_cell_active(env, row - 1, col) ? 1 : 0;
    mass += chainenv_cell_active(env, row + 1, col) ? 1 : 0;
    mass += chainenv_cell_active(env, row, col - 1) ? 1 : 0;
    mass += chainenv_cell_active(env, row, col + 1) ? 1 : 0;
    return mass;
}

static inline int chainenv_stable_capacity(const ChainEnv* env) {
    // Stable cells hold at most (critical_mass - 1) orbs, so this is the
    // largest turn count that can still admit a stable board on the chosen grid.
    return 3 * env->rows * env->cols - 2 * env->rows - 2 * env->cols;
}

static inline void chainenv_add_log(ChainEnv* env, int invalid) {
    float reward = env->rewards[0];
    env->log.perf += reward > 0.0f ? 1.0f : (reward == 0.0f ? 0.5f : 0.0f);
    env->log.score += reward;
    env->log.episode_return += reward;
    env->log.episode_length += (float)env->tick;
    env->log.chain_bursts += (float)env->last_chain_bursts;
    env->log.invalid_rate += invalid ? 1.0f : 0.0f;
    if (env->num_agents > 1) {
        float slot0_reward = env->rewards[0];
        float slot1_reward = env->rewards[1];
        if (fabsf(slot0_reward - slot1_reward) <= 1.0e-6f) {
            env->log.slot_0_score += 0.5f;
            env->log.slot_1_score += 0.5f;
            env->log.draw_rate += 1.0f;
        } else if (slot0_reward > slot1_reward) {
            env->log.slot_0_score += 1.0f;
        } else {
            env->log.slot_1_score += 1.0f;
        }
    }
    env->log.n += 1.0f;
}

static inline void chainenv_layout_board(Client* client, const ChainEnv* env) {
    float sidebar_w = fminf(250.0f, (float)client->screen_width * 0.26f);
    float gutter = fmaxf(30.0f, (float)client->screen_width * 0.04f);
    float board_area_x = sidebar_w + gutter;
    float board_area_w = fmaxf(260.0f, (float)client->screen_width - board_area_x - gutter);
    float available_h = (float)client->screen_height - 150.0f;
    float cell_size = fminf((board_area_w - 8.0f) / (float)env->cols, available_h / (float)env->rows);
    cell_size = fmaxf(cell_size, 48.0f);

    client->cell_size = cell_size;
    client->board_x = board_area_x + fmaxf(0.0f, (board_area_w - cell_size * env->cols) * 0.5f);
    client->board_y = ((float)client->screen_height - cell_size * env->rows) * 0.5f + 6.0f;
}

static inline void chainenv_reset_render_animations(ChainEnv* env) {
    if (env->client == NULL) {
        return;
    }

    env->client->animation_clock = 0.0f;
    env->client->animation_total = 0.0f;
    env->client->snapshot_count = 0;
    env->client->transfer_count = 0;
    env->client->burst_count = 0;
}

static inline void chainenv_clear_invalid_hint(ChainEnv* env) {
    if (env->client == NULL) {
        return;
    }

    env->client->invalid_hint_time = 0.0f;
}

static inline void chainenv_show_invalid_hint(ChainEnv* env) {
    if (env->client == NULL) {
        return;
    }

    env->client->invalid_hint_time = 1.9f;
}

static inline void chainenv_begin_animation(ChainEnv* env) {
    if (env->client == NULL) {
        return;
    }

    chainenv_reset_render_animations(env);
    chainenv_copy_board(
        env->client->base_owner,
        env->client->base_orbs,
        env->owner,
        env->orbs);
}

static inline void chainenv_record_snapshot(ChainEnv* env, float reveal_time) {
    if (env->client == NULL) {
        return;
    }

    Client* client = env->client;
    int snapshot_idx = client->snapshot_count;
    if (snapshot_idx >= CHAINENV_MAX_SNAPSHOTS) {
        snapshot_idx = CHAINENV_MAX_SNAPSHOTS - 1;
    } else {
        client->snapshot_count += 1;
    }

    chainenv_copy_board(
        client->snapshot_owner[snapshot_idx],
        client->snapshot_orbs[snapshot_idx],
        env->owner,
        env->orbs);
    client->snapshot_time[snapshot_idx] = reveal_time;

    if (reveal_time > client->animation_total) {
        client->animation_total = reveal_time;
    }
}

static inline void chainenv_record_burst(ChainEnv* env, int row, int col, int color, float delay) {
    if (env->client == NULL) {
        return;
    }

    Client* client = env->client;
    if (client->burst_count < CHAINENV_MAX_RENDER_BURSTS) {
        client->bursts[client->burst_count++] = (BurstAnim) {
            .row = row,
            .col = col,
            .color = color,
            .delay = delay,
            .duration = CHAINENV_BURST_DURATION,
        };
    }

    float end_time = delay + CHAINENV_BURST_DURATION;
    if (end_time > client->animation_total) {
        client->animation_total = end_time;
    }
}

static inline void chainenv_record_transfer(
        ChainEnv* env, int src_row, int src_col, int dst_row, int dst_col, int color, float delay) {
    if (env->client == NULL) {
        return;
    }

    Client* client = env->client;
    if (client->transfer_count < CHAINENV_MAX_RENDER_TRANSFERS) {
        client->transfers[client->transfer_count++] = (OrbTransferAnim) {
            .src_row = src_row,
            .src_col = src_col,
            .dst_row = dst_row,
            .dst_col = dst_col,
            .color = color,
            .delay = delay,
            .duration = CHAINENV_TRANSFER_DURATION,
        };
    }

    float end_time = delay + CHAINENV_TRANSFER_DURATION;
    if (end_time > client->animation_total) {
        client->animation_total = end_time;
    }
}

static inline void chainenv_recount(const ChainEnv* env, int* red_total, int* green_total) {
    int red = 0;
    int green = 0;
    for (int row = 0; row < env->rows; row++) {
        for (int col = 0; col < env->cols; col++) {
            int idx = chainenv_slot(row, col);
            if (env->owner[idx] == CHAINENV_PLAYER_RED) {
                red += env->orbs[idx];
            } else if (env->owner[idx] == CHAINENV_PLAYER_GREEN) {
                green += env->orbs[idx];
            }
        }
    }

    *red_total = red;
    *green_total = green;
}

static inline void chainenv_zero_outputs(ChainEnv* env) {
    for (int slot = 0; slot < env->num_agents; slot++) {
        env->rewards[slot] = 0.0f;
        env->terminals[slot] = 0.0f;
    }
}

static inline bool chainenv_is_legal_move(const ChainEnv* env, int action, int player);
static inline int chainenv_check_winner(const ChainEnv* env);

static inline void chainenv_compute_observations(ChainEnv* env) {
    int red_total = 0;
    int green_total = 0;
    chainenv_recount(env, &red_total, &green_total);
    memset(env->observations, 0, env->num_agents * CHAINENV_OBS_SIZE * sizeof(float));

    float area = (float)(env->rows * env->cols);
    for (int slot = 0; slot < env->num_agents; slot++) {
        int player = chainenv_player_for_slot(env, slot);
        float* obs = env->observations + slot * CHAINENV_OBS_SIZE;

        for (int row = 0; row < env->rows; row++) {
            for (int col = 0; col < env->cols; col++) {
                int idx = chainenv_slot(row, col);
                int obs_idx = idx * CHAINENV_OBS_CHANNELS;
                obs[obs_idx + 0] = 1.0f;
                obs[obs_idx + 1] = chainenv_critical_mass(env, row, col) / 4.0f;

                if (env->owner[idx] == player) {
                    obs[obs_idx + 2] = env->orbs[idx] / 4.0f;
                } else if (env->owner[idx] == -player) {
                    obs[obs_idx + 3] = env->orbs[idx] / 4.0f;
                }
            }
        }

        int self_total = player == CHAINENV_PLAYER_RED ? red_total : green_total;
        int opp_total = player == CHAINENV_PLAYER_RED ? green_total : red_total;
        int base = CHAINENV_MAX_CELLS * CHAINENV_OBS_CHANNELS;
        obs[base + 0] = env->rows / (float)CHAINENV_MAX_ROWS;
        obs[base + 1] = env->cols / (float)CHAINENV_MAX_COLS;
        obs[base + 2] = area > 0.0f ? self_total / (area * 4.0f) : 0.0f;
        obs[base + 3] = area > 0.0f ? opp_total / (area * 4.0f) : 0.0f;
        obs[base + 4] = env->current_player == player ? 1.0f : 0.0f;
    }
}

static inline void chainenv_update_action_mask(ChainEnv* env) {
    if (env->action_mask == NULL) {
        return;
    }

    memset(env->action_mask, 0, env->num_agents * CHAINENV_MAX_ACTIONS * sizeof(unsigned char));
    if (env->end_game) {
        for (int slot = 0; slot < env->num_agents; slot++) {
            env->action_mask[slot * CHAINENV_MAX_ACTIONS + CHAINENV_NOOP_ACTION] = 1;
        }
        return;
    }

    if (!chainenv_is_selfplay(env)) {
        for (int action = 0; action < env->rows * env->cols; action++) {
            int idx = chainenv_action_to_board_idx(env, action);
            if (idx < 0) {
                continue;
            }

            if (env->owner[idx] == 0 || env->owner[idx] == CHAINENV_PLAYER_RED) {
                env->action_mask[action] = 1;
            }
        }
        return;
    }

    int active_slot = chainenv_slot_for_player(env, env->current_player);
    for (int slot = 0; slot < env->num_agents; slot++) {
        unsigned char* mask = env->action_mask + slot * CHAINENV_MAX_ACTIONS;
        if (slot != active_slot) {
            mask[CHAINENV_NOOP_ACTION] = 1;
            continue;
        }

        for (int action = 0; action < env->rows * env->cols; action++) {
            if (chainenv_is_legal_move(env, action, env->current_player)) {
                mask[action] = 1;
            }
        }
    }
}

static inline bool chainenv_is_legal_move(const ChainEnv* env, int action, int player) {
    int idx = chainenv_action_to_board_idx(env, action);
    if (idx < 0) {
        return false;
    }

    return env->owner[idx] == 0 || env->owner[idx] == player;
}

static inline void chainenv_apply_single_orb(ChainEnv* env, int board_idx, int player) {
    if (env->owner[board_idx] != player) {
        env->owner[board_idx] = player;
    }
    env->orbs[board_idx] += 1;
}

static inline ResolveStats chainenv_resolve_chain(ChainEnv* env, int player, float start_delay) {
    ResolveStats stats = {0};
    stats.end_time = start_delay;
    int recorded_waves = 0;
    bool truncated_render = false;

    for (int wave = 0;; wave++) {
        int unstable[CHAINENV_MAX_CELLS];
        int unstable_count = 0;

        for (int r = 0; r < env->rows; r++) {
            for (int c = 0; c < env->cols; c++) {
                int idx = chainenv_slot(r, c);
                if (env->owner[idx] != player) {
                    continue;
                }

                int critical = chainenv_critical_mass(env, r, c);
                if (env->orbs[idx] >= critical) {
                    unstable[unstable_count++] = idx;
                }
            }
        }

        if (unstable_count == 0) {
            break;
        }

        stats.waves += 1;
        bool record_wave = wave < CHAINENV_MAX_RENDER_WAVES;
        float delay = start_delay + recorded_waves * CHAINENV_WAVE_DELAY;
        for (int i = 0; i < unstable_count; i++) {
            int idx = unstable[i];
            int burst_row = idx / CHAINENV_MAX_COLS;
            int burst_col = idx % CHAINENV_MAX_COLS;
            int critical = chainenv_critical_mass(env, burst_row, burst_col);

            env->orbs[idx] -= critical;
            if (env->orbs[idx] <= 0) {
                env->orbs[idx] = 0;
                env->owner[idx] = 0;
            } else {
                env->owner[idx] = player;
            }

            stats.bursts += 1;
            if (record_wave) {
                chainenv_record_burst(env, burst_row, burst_col, player, delay);
            }
        }

        for (int i = 0; i < unstable_count; i++) {
            int idx = unstable[i];
            int src_row = idx / CHAINENV_MAX_COLS;
            int src_col = idx % CHAINENV_MAX_COLS;
            static const int dr[4] = {-1, 1, 0, 0};
            static const int dc[4] = {0, 0, -1, 1};

            for (int d = 0; d < 4; d++) {
                int nr = src_row + dr[d];
                int nc = src_col + dc[d];
                if (!chainenv_cell_active(env, nr, nc)) {
                    continue;
                }

                int nidx = chainenv_slot(nr, nc);
                chainenv_apply_single_orb(env, nidx, player);
                stats.transfers += 1;
                if (record_wave) {
                    chainenv_record_transfer(env, src_row, src_col, nr, nc, player, delay);
                }
            }
        }

        if (record_wave) {
            float reveal_time = delay + fmaxf(CHAINENV_BURST_DURATION, CHAINENV_TRANSFER_DURATION);
            chainenv_record_snapshot(env, reveal_time);
            stats.end_time = reveal_time;
            recorded_waves += 1;
        } else {
            truncated_render = true;
        }

        // Match the mobile game: once a player has been wiped out after both
        // sides have taken at least one turn, the move ends immediately instead
        // of continuing a single-color avalanche.
        if (chainenv_check_winner(env) != 0) {
            break;
        }
    }

    if (truncated_render) {
        float final_time = start_delay
            + recorded_waves * CHAINENV_WAVE_DELAY
            + fmaxf(CHAINENV_BURST_DURATION, CHAINENV_TRANSFER_DURATION);
        chainenv_record_snapshot(env, final_time);
        stats.end_time = final_time;
    }

    return stats;
}

static inline bool chainenv_player_eliminated(const ChainEnv* env, int player) {
    for (int row = 0; row < env->rows; row++) {
        for (int col = 0; col < env->cols; col++) {
            int idx = chainenv_slot(row, col);
            if (env->owner[idx] == player && env->orbs[idx] > 0) {
                return false;
            }
        }
    }

    return true;
}

static inline int chainenv_check_winner(const ChainEnv* env) {
    if (env->turns_taken[0] == 0 || env->turns_taken[1] == 0) {
        return 0;
    }

    bool red_out = chainenv_player_eliminated(env, CHAINENV_PLAYER_RED);
    bool green_out = chainenv_player_eliminated(env, CHAINENV_PLAYER_GREEN);
    if (!red_out && green_out) {
        return CHAINENV_PLAYER_RED;
    }
    if (red_out && !green_out) {
        return CHAINENV_PLAYER_GREEN;
    }

    return 0;
}

static inline int chainenv_timeout_winner(const ChainEnv* env) {
    int red_total = 0;
    int green_total = 0;
    chainenv_recount(env, &red_total, &green_total);

    if (red_total > green_total) {
        return CHAINENV_PLAYER_RED;
    }
    if (green_total > red_total) {
        return CHAINENV_PLAYER_GREEN;
    }
    return 0;
}

static inline float chainenv_move_score(const ChainEnv* env, int action, int player) {
    int idx = chainenv_action_to_board_idx(env, action);
    if (idx < 0) {
        return -1.0e9f;
    }

    int row = idx / CHAINENV_MAX_COLS;
    int col = idx % CHAINENV_MAX_COLS;
    int critical = chainenv_critical_mass(env, row, col);
    int existing = env->orbs[idx];
    int owner = env->owner[idx];
    float score = 0.0f;
    int friendly_neighbors = 0;
    int enemy_neighbors = 0;

    score += owner == player ? 3.1f : 1.0f;
    score += (5 - critical) * 0.35f;
    score += (existing + 1 >= critical) ? 4.0f : (1.2f / (float)(critical - existing));

    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    for (int d = 0; d < 4; d++) {
        int nr = row + dr[d];
        int nc = col + dc[d];
        if (!chainenv_cell_active(env, nr, nc)) {
            continue;
        }

        int nidx = chainenv_slot(nr, nc);
        int nowner = env->owner[nidx];
        int norbs = env->orbs[nidx];
        int ncritical = chainenv_critical_mass(env, nr, nc);

        if (nowner == player) {
            friendly_neighbors += 1;
            score += norbs == ncritical - 1 ? 1.1f : 0.32f * norbs;
        } else if (nowner == -player) {
            enemy_neighbors += 1;
            score += norbs == ncritical - 1 ? 2.2f : 0.15f * norbs;
            if (owner == 0 && env->move_count < 10) {
                score -= 0.35f;
            }
        }
    }

    if (owner == 0) {
        if (env->move_count < 8) {
            score += 0.18f * friendly_neighbors;
            score -= 0.20f * enemy_neighbors;
        } else {
            score += 0.10f * friendly_neighbors;
        }
    }

    return score;
}

static inline int chainenv_choose_env_action(ChainEnv* env) {
    int legal_actions[CHAINENV_MAX_CELLS];
    int legal_count = 0;
    for (int action = 0; action < env->rows * env->cols; action++) {
        int idx = chainenv_action_to_board_idx(env, action);
        if (idx < 0) {
            continue;
        }

        if (env->owner[idx] == 0 || env->owner[idx] == CHAINENV_PLAYER_GREEN) {
            legal_actions[legal_count++] = action;
        }
    }

    if (legal_count == 0) {
        return -1;
    }

    if (env->opponent_policy == CHAINENV_OPPONENT_RANDOM) {
        return legal_actions[rand_r(&env->rng) % legal_count];
    }

    float scores[CHAINENV_MAX_CELLS];
    float best_score = -1.0e9f;
    int best_action = legal_actions[0];
    for (int i = 0; i < legal_count; i++) {
        int action = legal_actions[i];
        float score = chainenv_move_score(env, action, CHAINENV_PLAYER_GREEN);
        scores[i] = score;
        if (score > best_score + 1.0e-6f) {
            best_score = score;
            best_action = action;
        }
    }

    float margin = env->move_count < 8 ? 0.95f : 0.55f;
    float threshold = best_score - margin;
    float total_weight = 0.0f;
    int candidate_count = 0;
    int candidates[CHAINENV_MAX_CELLS];
    float weights[CHAINENV_MAX_CELLS];

    for (int i = 0; i < legal_count; i++) {
        if (scores[i] + 1.0e-6f < threshold) {
            continue;
        }

        float weight = 0.15f + (scores[i] - threshold);
        candidates[candidate_count] = legal_actions[i];
        weights[candidate_count] = weight;
        total_weight += weight;
        candidate_count += 1;
    }

    if (candidate_count == 0 || total_weight <= 1.0e-6f) {
        return best_action;
    }

    float sample = ((float)(rand_r(&env->rng) % 100000) / 100000.0f) * total_weight;
    for (int i = 0; i < candidate_count; i++) {
        sample -= weights[i];
        if (sample <= 0.0f) {
            return candidates[i];
        }
    }

    return candidates[candidate_count - 1];
}

static inline float chainenv_reward_for_player(
        const ChainEnv* env, int player, int winner, int invalid_player) {
    if (winner == 0) {
        return 0.0f;
    }
    if (player == winner) {
        return env->win_reward;
    }
    if (invalid_player != 0 && player == invalid_player) {
        return env->invalid_move_reward;
    }
    return env->loss_reward;
}

static inline void chainenv_finish_game(ChainEnv* env, int winner, int invalid_player) {
    if (env->num_agents == 1) {
        env->rewards[0] = chainenv_reward_for_player(
            env, CHAINENV_PLAYER_RED, winner, invalid_player);
        env->terminals[0] = 1.0f;
    } else {
        for (int slot = 0; slot < env->num_agents; slot++) {
            int player = chainenv_player_for_slot(env, slot);
            env->rewards[slot] = chainenv_reward_for_player(
                env, player, winner, invalid_player);
            env->terminals[slot] = 1.0f;
        }
    }

    env->end_game = 1;
    env->winner = winner;
    chainenv_update_action_mask(env);
    chainenv_add_log(env, invalid_player != 0);
}

static inline void chainenv_reset_slot_mapping(ChainEnv* env) {
    if (chainenv_is_selfplay(env) && (rand_r(&env->rng) & 1) != 0) {
        chainenv_set_slot_players(env, CHAINENV_PLAYER_GREEN, CHAINENV_PLAYER_RED);
        return;
    }

    chainenv_set_slot_players(env, CHAINENV_PLAYER_RED, CHAINENV_PLAYER_GREEN);
}

static inline ResolveStats chainenv_execute_turn(
        ChainEnv* env, int player, int action, float start_delay) {
    ResolveStats stats = {0};
    int board_idx = chainenv_action_to_board_idx(env, action);
    if (board_idx < 0) {
        return stats;
    }

    env->turns_taken[chainenv_player_index(player)] += 1;
    env->move_count += 1;
    if (player == CHAINENV_PLAYER_RED) {
        env->last_player_action = action;
    } else {
        env->last_env_action = action;
    }

    chainenv_apply_single_orb(env, board_idx, player);
    chainenv_record_snapshot(env, start_delay + CHAINENV_PLACEMENT_DURATION);
    return chainenv_resolve_chain(env, player, start_delay + CHAINENV_PLACEMENT_DURATION);
}

static inline void chainenv_maybe_open_scripted_duel(ChainEnv* env) {
    if (chainenv_is_selfplay(env)) {
        return;
    }

    env->current_player = CHAINENV_PLAYER_RED;
    if ((rand_r(&env->rng) & 1u) == 0u) {
        return;
    }

    int env_action = chainenv_choose_env_action(env);
    if (env_action < 0) {
        return;
    }

    chainenv_begin_animation(env);
    ResolveStats env_stats = chainenv_execute_turn(env, CHAINENV_PLAYER_GREEN, env_action, 0.0f);
    env->last_chain_bursts = env_stats.bursts;
    env->last_chain_waves = env_stats.waves;
    env->current_player = CHAINENV_PLAYER_RED;
}

static inline void init_chainenv(ChainEnv* env) {
    if (env->num_agents < 1) env->num_agents = 1;
    if (env->num_agents > CHAINENV_MAX_SLOTS) env->num_agents = CHAINENV_MAX_SLOTS;
    if (env->max_steps < 1) env->max_steps = 1;
    int stable_capacity = chainenv_stable_capacity(env);
    if (env->max_steps > stable_capacity) env->max_steps = stable_capacity;
    env->tick = 0;
    env->move_count = 0;
    env->end_game = 0;
    env->turns_taken[0] = 0;
    env->turns_taken[1] = 0;
    env->current_player = CHAINENV_PLAYER_RED;
    env->last_player_action = -1;
    env->last_env_action = -1;
    env->last_chain_bursts = 0;
    env->last_chain_waves = 0;
    env->winner = 0;
    chainenv_set_slot_players(env, CHAINENV_PLAYER_RED, CHAINENV_PLAYER_GREEN);
    chainenv_reset_render_animations(env);
}

static inline void c_close(ChainEnv* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

static inline void c_reset(ChainEnv* env) {
    memset(env->owner, 0, sizeof(env->owner));
    memset(env->orbs, 0, sizeof(env->orbs));
    env->tick = 0;
    env->move_count = 0;
    env->end_game = 0;
    env->turns_taken[0] = 0;
    env->turns_taken[1] = 0;
    env->current_player = CHAINENV_PLAYER_RED;
    env->last_player_action = -1;
    env->last_env_action = -1;
    env->last_chain_bursts = 0;
    env->last_chain_waves = 0;
    env->winner = 0;
    chainenv_reset_slot_mapping(env);
    chainenv_zero_outputs(env);
    chainenv_reset_render_animations(env);
    chainenv_clear_invalid_hint(env);
    chainenv_maybe_open_scripted_duel(env);
    chainenv_compute_observations(env);
    chainenv_update_action_mask(env);
}

static inline void c_step(ChainEnv* env) {
    env->tick += 1;
    chainenv_zero_outputs(env);

    if (env->end_game) {
        c_reset(env);
        return;
    }

    env->last_chain_bursts = 0;
    env->last_chain_waves = 0;

    if (chainenv_is_selfplay(env)) {
        int active_slot = chainenv_slot_for_player(env, env->current_player);
        int action = (int)env->actions[active_slot];
        if (env->current_player == CHAINENV_PLAYER_RED) {
            env->last_player_action = action;
        } else {
            env->last_env_action = action;
        }

        if (action == CHAINENV_NOOP_ACTION ||
                !chainenv_is_legal_move(env, action, env->current_player)) {
            chainenv_compute_observations(env);
            chainenv_finish_game(env, -env->current_player, env->current_player);
            return;
        }

        chainenv_begin_animation(env);
        ResolveStats stats = chainenv_execute_turn(env, env->current_player, action, 0.0f);
        env->last_chain_bursts = stats.bursts;
        env->last_chain_waves = stats.waves;

        int winner = chainenv_check_winner(env);
        if (winner == 0 && env->move_count >= env->max_steps) {
            winner = chainenv_timeout_winner(env);
        }

        if (winner != 0 || env->move_count >= env->max_steps) {
            chainenv_compute_observations(env);
            chainenv_finish_game(env, winner, 0);
            return;
        }

        env->current_player = -env->current_player;
        chainenv_compute_observations(env);
        chainenv_update_action_mask(env);
        return;
    }

    int action = (int)env->actions[0];
    env->last_env_action = -1;
    if (!chainenv_is_legal_move(env, action, CHAINENV_PLAYER_RED)) {
        env->last_player_action = action;
        chainenv_compute_observations(env);
        chainenv_finish_game(env, CHAINENV_PLAYER_GREEN, CHAINENV_PLAYER_RED);
        return;
    }

    chainenv_begin_animation(env);
    ResolveStats player_stats = chainenv_execute_turn(env, CHAINENV_PLAYER_RED, action, 0.0f);
    env->last_chain_bursts = player_stats.bursts;
    env->last_chain_waves = player_stats.waves;

    int winner = chainenv_check_winner(env);
    if (winner == 0 && env->move_count >= env->max_steps) {
        winner = chainenv_timeout_winner(env);
    }
    if (winner != 0 || env->move_count >= env->max_steps) {
        chainenv_compute_observations(env);
        chainenv_finish_game(env, winner, 0);
        return;
    }

    int env_action = chainenv_choose_env_action(env);
    if (env_action < 0) {
        chainenv_compute_observations(env);
        chainenv_finish_game(env, CHAINENV_PLAYER_RED, 0);
        return;
    }

    float env_start = player_stats.end_time + CHAINENV_TURN_GAP;
    ResolveStats env_stats = chainenv_execute_turn(
        env, CHAINENV_PLAYER_GREEN, env_action, env_start);
    env->last_chain_bursts += env_stats.bursts;
    env->last_chain_waves += env_stats.waves;

    winner = chainenv_check_winner(env);
    if (winner == 0 && env->move_count >= env->max_steps) {
        winner = chainenv_timeout_winner(env);
    }
    env->current_player = CHAINENV_PLAYER_RED;
    chainenv_compute_observations(env);
    if (winner != 0 || env->move_count >= env->max_steps) {
        chainenv_finish_game(env, winner, 0);
        return;
    }

    chainenv_update_action_mask(env);
}

static inline Client* make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->screen_width = 980;
    client->screen_height = 860;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(client->screen_width, client->screen_height, "PufferLib Chain Reaction");
    SetTargetFPS(60);
    return client;
}

static inline Color chainenv_player_color(int player) {
    return player == CHAINENV_PLAYER_RED
        ? (Color){255, 80, 88, 255}
        : (Color){77, 231, 125, 255};
}

static inline Color chainenv_dim_color(Color color, float factor) {
    return (Color) {
        (unsigned char)fminf(255.0f, color.r * factor),
        (unsigned char)fminf(255.0f, color.g * factor),
        (unsigned char)fminf(255.0f, color.b * factor),
        color.a,
    };
}

static inline float chainenv_ease_out(float t) {
    float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
}

static inline void chainenv_draw_glossy_orb(Vector2 center, float radius, Color color, float alpha) {
    Color shadow = (Color){0, 0, 0, (unsigned char)(100.0f * alpha)};
    Color base = color;
    base.a = (unsigned char)(255.0f * alpha);
    Color dark = chainenv_dim_color(color, 0.60f);
    dark.a = (unsigned char)(220.0f * alpha);
    Color highlight = (Color){255, 255, 255, (unsigned char)(180.0f * alpha)};

    DrawCircleV((Vector2){center.x + radius * 0.14f, center.y + radius * 0.18f}, radius * 1.05f, shadow);
    DrawCircleV(center, radius, dark);
    DrawCircleV((Vector2){center.x - radius * 0.08f, center.y - radius * 0.08f}, radius * 0.88f, base);
    DrawCircleV((Vector2){center.x - radius * 0.30f, center.y - radius * 0.34f}, radius * 0.32f, highlight);
}

static inline void chainenv_draw_stack(
        Client* client, int row, int col, int owner, int count, float time) {
    if (owner == 0 || count <= 0) {
        return;
    }

    float cx = client->board_x + (col + 0.5f) * client->cell_size;
    float cy = client->board_y + (row + 0.5f) * client->cell_size;
    float orbit = client->cell_size * 0.12f;
    float radius = client->cell_size * 0.18f;
    Color color = chainenv_player_color(owner);

    if (count == 1) {
        chainenv_draw_glossy_orb((Vector2){cx, cy}, radius, color, 1.0f);
        return;
    }

    float speed = owner == CHAINENV_PLAYER_RED ? CHAINENV_RED_ORBIT_SPEED : CHAINENV_GREEN_ORBIT_SPEED;
    for (int i = 0; i < count; i++) {
        float angle = time * speed + ((2.0f * PI) * i) / (float)count;
        float ox = cosf(angle) * orbit;
        float oy = sinf(angle) * orbit * 0.65f;
        chainenv_draw_glossy_orb((Vector2){cx + ox, cy + oy}, radius * 0.92f, color, 1.0f);
    }
}

static inline void chainenv_get_display_board(
        const ChainEnv* env, const Client* client, int* display_owner, int* display_orbs) {
    chainenv_copy_board(display_owner, display_orbs, env->owner, env->orbs);
    if (client == NULL || client->snapshot_count == 0) {
        return;
    }

    chainenv_copy_board(display_owner, display_orbs, client->base_owner, client->base_orbs);
    for (int i = 0; i < client->snapshot_count; i++) {
        if (client->snapshot_time[i] > client->animation_clock + 1.0e-4f) {
            break;
        }

        chainenv_copy_board(
            display_owner,
            display_orbs,
            client->snapshot_owner[i],
            client->snapshot_orbs[i]);
    }
}

static inline void chainenv_draw_board(ChainEnv* env, Client* client) {
    Color bg = (Color){7, 20, 25, 255};
    Color frame = (Color){16, 47, 57, 255};
    Color inner = (Color){18, 69, 84, 255};
    Color grid_line = (Color){111, 214, 232, 55};
    int display_owner[CHAINENV_MAX_CELLS];
    int display_orbs[CHAINENV_MAX_CELLS];

    DrawRectangleGradientV(0, 0, client->screen_width, client->screen_height,
        (Color){4, 12, 18, 255}, bg);

    float board_w = client->cell_size * env->cols;
    float board_h = client->cell_size * env->rows;
    DrawRectangleRounded(
        (Rectangle){client->board_x - 22, client->board_y - 22, board_w + 44, board_h + 44},
        0.08f, 24, frame);
    DrawRectangleRounded(
        (Rectangle){client->board_x - 8, client->board_y - 8, board_w + 16, board_h + 16},
        0.05f, 24, inner);

    float time = (float)GetTime();
    chainenv_get_display_board(env, client, display_owner, display_orbs);
    for (int row = 0; row < env->rows; row++) {
        for (int col = 0; col < env->cols; col++) {
            float x = client->board_x + col * client->cell_size;
            float y = client->board_y + row * client->cell_size;
            Rectangle cell = {x, y, client->cell_size, client->cell_size};
            DrawRectangleRounded(cell, 0.14f, 12, (Color){14, 44, 55, 255});
            DrawRectangleRounded(
                (Rectangle){x + 3, y + 3, client->cell_size - 6, client->cell_size - 6},
                0.14f, 12, (Color){11, 33, 42, 255});
            DrawRectangleLinesEx(
                (Rectangle){x + 1, y + 1, client->cell_size - 2, client->cell_size - 2},
                1.0f, grid_line);

            int idx = chainenv_slot(row, col);
            int critical = chainenv_critical_mass(env, row, col);
            char label[8];
            snprintf(label, sizeof(label), "%d", critical);
            DrawText(
                label,
                (int)(x + client->cell_size - 16),
                (int)(y + 8),
                (int)(client->cell_size * 0.16f),
                (Color){176, 228, 233, 135});

            chainenv_draw_stack(client, row, col, display_owner[idx], display_orbs[idx], time);
        }
    }
}

static inline void chainenv_draw_effects(ChainEnv* env, Client* client) {
    float t = client->animation_clock;
    for (int i = 0; i < client->burst_count; i++) {
        BurstAnim* burst = &client->bursts[i];
        float local = t - burst->delay;
        if (local < 0.0f || local > burst->duration) {
            continue;
        }

        float p = local / burst->duration;
        float eased = chainenv_ease_out(p);
        float cx = client->board_x + (burst->col + 0.5f) * client->cell_size;
        float cy = client->board_y + (burst->row + 0.5f) * client->cell_size;
        Color color = chainenv_player_color(burst->color);
        color.a = (unsigned char)(150.0f * (1.0f - p));
        DrawRing((Vector2){cx, cy},
            client->cell_size * (0.10f + eased * 0.02f),
            client->cell_size * (0.24f + eased * 0.30f),
            0.0f, 360.0f, 32, color);
    }

    for (int i = 0; i < client->transfer_count; i++) {
        OrbTransferAnim* anim = &client->transfers[i];
        float local = t - anim->delay;
        if (local < 0.0f || local > anim->duration) {
            continue;
        }

        float p = chainenv_ease_out(local / anim->duration);
        float src_x = client->board_x + (anim->src_col + 0.5f) * client->cell_size;
        float src_y = client->board_y + (anim->src_row + 0.5f) * client->cell_size;
        float dst_x = client->board_x + (anim->dst_col + 0.5f) * client->cell_size;
        float dst_y = client->board_y + (anim->dst_row + 0.5f) * client->cell_size;
        Vector2 pos = {
            src_x + (dst_x - src_x) * p,
            src_y + (dst_y - src_y) * p,
        };
        float arc = sinf(p * PI) * client->cell_size * 0.07f;
        pos.y -= arc;
        chainenv_draw_glossy_orb(pos, client->cell_size * 0.10f, chainenv_player_color(anim->color), 0.95f);
    }
}

static inline void chainenv_draw_hud(ChainEnv* env, Client* client) {
    int red_total = 0;
    int green_total = 0;
    chainenv_recount(env, &red_total, &green_total);
    const char* mode_label = chainenv_is_selfplay(env) ? "Self-play" : "Scripted duel";
    const char* red_label = chainenv_is_selfplay(env) ? "Red" : "Red (you)";
    const char* green_label = chainenv_is_selfplay(env) ? "Green" : "Green (scripted)";
    float sidebar_x = 32.0f;
    float sidebar_y = 26.0f;
    float header_h = 124.0f;
    float stats_y = sidebar_y + header_h + 8.0f;
    int stat_label_x = (int)(sidebar_x + 20.0f);
    int stat_value_x = (int)(sidebar_x + 20.0f);
    int stat_y = (int)(stats_y + 12.0f);
    int stat_gap = 42;

    DrawText("Chain Reaction", (int)(sidebar_x + 20.0f), (int)(sidebar_y + 18.0f), 33,
        (Color){236, 248, 250, 255});
    DrawText(red_label, (int)(sidebar_x + 22.0f), (int)(sidebar_y + 64.0f), 22,
        chainenv_player_color(CHAINENV_PLAYER_RED));
    DrawText(green_label, (int)(sidebar_x + 22.0f), (int)(sidebar_y + 92.0f), 22,
        chainenv_player_color(CHAINENV_PLAYER_GREEN));

    DrawText("Mode", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText(mode_label, stat_value_x, stat_y + 16, 23, (Color){236, 248, 250, 255});
    stat_y += stat_gap;

    char board_text[32];
    snprintf(board_text, sizeof(board_text), "%dx%d", env->rows, env->cols);
    DrawText("Board", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText(board_text, stat_value_x, stat_y + 16, 23, (Color){236, 248, 250, 255});
    stat_y += stat_gap;

    char red_text[32];
    char green_text[32];
    char burst_text[32];
    snprintf(red_text, sizeof(red_text), "%d", red_total);
    snprintf(green_text, sizeof(green_text), "%d", green_total);
    snprintf(burst_text, sizeof(burst_text), "%d", env->last_chain_bursts);

    DrawText("Red Orbs", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText(red_text, stat_value_x, stat_y + 16, 23, chainenv_player_color(CHAINENV_PLAYER_RED));
    stat_y += stat_gap;

    DrawText("Green Orbs", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText(green_text, stat_value_x, stat_y + 16, 23, chainenv_player_color(CHAINENV_PLAYER_GREEN));
    stat_y += stat_gap;

    DrawText("Last Chain Bursts", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText(burst_text, stat_value_x, stat_y + 16, 23, (Color){236, 248, 250, 255});
    stat_y += stat_gap;

    DrawText("Cell Labels", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
    DrawText("Critical mass", stat_value_x, stat_y + 16, 22, (Color){210, 233, 239, 255});
    stat_y += stat_gap;

    if (!chainenv_is_selfplay(env) && client->invalid_hint_time > 0.0f) {
        DrawText("Invalid Move", stat_label_x, stat_y, 15, (Color){236, 183, 120, 240});
        DrawText("Place on empty or red cells", stat_value_x, stat_y + 16, 22,
            (Color){255, 216, 164, 255});
    }

    if (!env->end_game) {
        return;
    }

    const char* result = "Draw";
    Color result_color = (Color){235, 235, 235, 255};
    if (env->winner == CHAINENV_PLAYER_RED) {
        result = "Red Wins";
        result_color = chainenv_player_color(CHAINENV_PLAYER_RED);
    } else if (env->winner == CHAINENV_PLAYER_GREEN) {
        result = "Green Wins";
        result_color = chainenv_player_color(CHAINENV_PLAYER_GREEN);
    }

    float board_w = client->cell_size * env->cols;
    float board_h = client->cell_size * env->rows;
    int center_x = (int)(client->board_x + board_w * 0.5f);
    int center_y = (int)(client->board_y + board_h * 0.5f);
    int font_size = 52;
    int msg_width = MeasureText(result, font_size);
    DrawText(result,
        center_x - msg_width / 2,
        center_y - 48,
        font_size, result_color);
    const char* reset_text = "Click anywhere to reset";
    int reset_width = MeasureText(reset_text, 24);
    DrawText(reset_text,
        center_x - reset_width / 2,
        center_y + 18,
        24, (Color){240, 240, 240, 230});
}

static inline int chainenv_pick_action(const ChainEnv* env, const Client* client, Vector2 mouse) {
    float board_w = client->cell_size * env->cols;
    float board_h = client->cell_size * env->rows;
    if (mouse.x < client->board_x || mouse.x >= client->board_x + board_w ||
            mouse.y < client->board_y || mouse.y >= client->board_y + board_h) {
        return -1;
    }

    int col = (int)((mouse.x - client->board_x) / client->cell_size);
    int row = (int)((mouse.y - client->board_y) / client->cell_size);
    if (!chainenv_cell_active(env, row, col)) {
        return -1;
    }

    return row * env->cols + col;
}

static inline bool chainenv_animation_active(const ChainEnv* env) {
    if (env->client == NULL) {
        return false;
    }

    return env->client->animation_clock < env->client->animation_total - 1.0e-4f;
}

static inline void c_render(ChainEnv* env) {
    if (env->client == NULL) {
        env->client = make_client();
    }

    if (IsWindowReady()) {
        env->client->screen_width = GetScreenWidth();
        env->client->screen_height = GetScreenHeight();
    }

    if (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    chainenv_layout_board(env->client, env);
    float dt = GetFrameTime();
    env->client->invalid_hint_time = fmaxf(0.0f, env->client->invalid_hint_time - dt);
    env->client->animation_clock = fminf(
        env->client->animation_total,
        env->client->animation_clock + dt);

    BeginDrawing();
    chainenv_draw_board(env, env->client);
    chainenv_draw_effects(env, env->client);
    chainenv_draw_hud(env, env->client);
    EndDrawing();
}
