/* Chain Reaction env core */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define MAX_ROWS 9
#define MAX_COLS 9
#define MAX_CELLS (MAX_ROWS * MAX_COLS)
#define MAX_ACTIONS (MAX_CELLS + 1)
#define NOOP_ACTION MAX_CELLS
#define MAX_SLOTS 2
#define MAX_BANKS 8
#define OBS_CHANNELS 4
#define GLOBAL_OBS 5
#define OBS_SIZE (MAX_CELLS * OBS_CHANNELS + GLOBAL_OBS)
#define MAX_RENDER_WAVES 256
#define RED_PLAYER 1
#define GREEN_PLAYER -1
#define RANDOM_OPPONENT 0
#define HEURISTIC_OPPONENT 1
#define MAX_RENDER_TRANSFERS 768
#define MAX_RENDER_BURSTS 256
// Scripted mode animates both players' turns without resetting the client.
#define MAX_SNAPSHOTS (2 * (MAX_RENDER_WAVES + 2))
#define BURST_DURATION 0.36f
#define TRANSFER_DURATION 0.28f
#define WAVE_DELAY 0.10f
#define PLACEMENT_DURATION 0.11f
#define TURN_GAP 0.08f
#define RED_ORBIT_SPEED 0.90f
#define GREEN_ORBIT_SPEED -0.78f

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float chain_bursts;
    float invalid_rate;
    float hist_score;
    float hist_n;
    float hist_score_bank[MAX_BANKS];
    float hist_n_bank[MAX_BANKS];
    float slot_0_score;
    float slot_1_score;
    float draw_rate;
    float n;
} Log;

typedef struct {
    int src_row;
    int src_col;
    int dst_row;
    int dst_col;
    int color;
    float delay;
    float duration;
} OrbTransferAnim;

typedef struct {
    int row;
    int col;
    int color;
    float delay;
    float duration;
} BurstAnim;

typedef struct {
    int screen_width;
    int screen_height;
    float board_x;
    float board_y;
    float cell_size;
    float animation_clock;
    float animation_total;
    float invalid_hint_time;
    OrbTransferAnim transfers[MAX_RENDER_TRANSFERS];
    BurstAnim bursts[MAX_RENDER_BURSTS];
    int8_t base_owner[MAX_CELLS];
    uint16_t base_orbs[MAX_CELLS];
    int8_t snapshot_owner[MAX_SNAPSHOTS][MAX_CELLS];
    uint16_t snapshot_orbs[MAX_SNAPSHOTS][MAX_CELLS];
    float snapshot_time[MAX_SNAPSHOTS];
    int snapshot_count;
    int transfer_count;
    int burst_count;
} Client;

typedef struct {
    int bursts;
    int winner;
    float end_time;
} ResolveStats;

typedef struct {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    // Logical slots may be non-adjacent during historical self-play.
    float* obs_ptr[MAX_SLOTS];
    float* action_ptr[MAX_SLOTS];
    float* reward_ptr[MAX_SLOTS];
    float* terminal_ptr[MAX_SLOTS];
    unsigned char* action_mask_ptr[MAX_SLOTS];
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

    int8_t owner[MAX_CELLS];
    uint16_t orbs[MAX_CELLS];
    uint8_t action_cell[MAX_CELLS];
    uint8_t critical_mass[MAX_CELLS];
    uint8_t neighbor_count[MAX_CELLS];
    uint8_t neighbors[MAX_CELLS][4];
    int move_count;
    int tick;
    int end_game;
    int turns_taken[2];
    int current_player;
    int player_for_slot[MAX_SLOTS];
    int slot_for_player[MAX_SLOTS];
    int last_player_action;
    int last_env_action;
    int last_chain_bursts;
    int winner;
    // Selfplay-pool tagging. tag = 0 means pure selfplay; tag = 1..N means
    // slot 0 (primary) is playing frozen bank tag-1. boundary_reached is set
    // on historical episode end and cleared by pufferl_count_aligned after a
    // pending opponent swap has safely aligned all tagged envs.
    int tag;
    int boundary_reached;
    unsigned int rng;
} ChainEnv;

static int board_index(int row, int col) {
    return row * MAX_COLS + col;
}

static int player_index(int player) {
    return (player == RED_PLAYER) ? 0 : 1;
}

static bool is_selfplay(ChainEnv* env) {
    return env->num_agents > 1;
}

static int slot_for_player(ChainEnv* env, int player) {
    return env->slot_for_player[player_index(player)];
}

static int player_for_slot(ChainEnv* env, int slot) {
    return env->player_for_slot[slot];
}

static void set_slot_players(ChainEnv* env, int slot0_player, int slot1_player) {
    env->player_for_slot[0] = slot0_player;
    env->player_for_slot[1] = slot1_player;
    env->slot_for_player[player_index(slot0_player)] = 0;
    env->slot_for_player[player_index(slot1_player)] = 1;
}

// Rendering snapshots whole board states so chain waves can be replayed visually.
static void copy_render_board(
        int8_t* dst_owner, uint16_t* dst_orbs,
        int8_t* src_owner, uint16_t* src_orbs) {
    memcpy(dst_owner, src_owner, sizeof(int8_t) * MAX_CELLS);
    memcpy(dst_orbs, src_orbs, sizeof(uint16_t) * MAX_CELLS);
}

static void layout_board(ChainEnv* env) {
    Client* client = env->client;
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

static void reset_animations(Client* client) {
    client->animation_clock = 0.0f;
    client->animation_total = 0.0f;
    client->snapshot_count = 0;
    client->transfer_count = 0;
    client->burst_count = 0;
}

static void clear_invalid_hint(Client* client) {
    client->invalid_hint_time = 0.0f;
}

static void begin_animation(ChainEnv* env) {
    reset_animations(env->client);
    copy_render_board(
        env->client->base_owner,
        env->client->base_orbs,
        env->owner,
        env->orbs);
}

static void record_snapshot(ChainEnv* env, float reveal_time) {
    Client* client = env->client;
    int snapshot_idx = client->snapshot_count++;
    copy_render_board(
        client->snapshot_owner[snapshot_idx],
        client->snapshot_orbs[snapshot_idx],
        env->owner,
        env->orbs);
    client->snapshot_time[snapshot_idx] = reveal_time;

    if (reveal_time > client->animation_total) {
        client->animation_total = reveal_time;
    }
}

static void count_orbs(ChainEnv* env, int* red_total, int* green_total) {
    int red = 0;
    int green = 0;
    for (int row = 0; row < env->rows; row++) {
        for (int col = 0; col < env->cols; col++) {
            int idx = board_index(row, col);
            if (env->owner[idx] == RED_PLAYER) {
                red += env->orbs[idx];
            } else if (env->owner[idx] == GREEN_PLAYER) {
                green += env->orbs[idx];
            }
        }
    }

    *red_total = red;
    *green_total = green;
}

static void zero_outputs(ChainEnv* env) {
    for (int slot = 0; slot < env->num_agents; slot++) {
        *env->reward_ptr[slot] = 0.0f;
        *env->terminal_ptr[slot] = 0.0f;
    }
}

static bool is_legal_move(ChainEnv* env, int action, int player) {
    if (action < 0 || action >= env->rows * env->cols) {
        return false;
    }
    int owner = env->owner[env->action_cell[action]];
    return owner == 0 || owner == player;
}

static void compute_observations(ChainEnv* env) {
    int red_total = 0;
    int green_total = 0;
    count_orbs(env, &red_total, &green_total);

    float area = (float)(env->rows * env->cols);
    for (int slot = 0; slot < env->num_agents; slot++) {
        float* obs = env->obs_ptr[slot];
        memset(obs, 0, OBS_SIZE * sizeof(float));
        int player = player_for_slot(env, slot);

        for (int row = 0; row < env->rows; row++) {
            for (int col = 0; col < env->cols; col++) {
                int idx = board_index(row, col);
                int obs_idx = idx * OBS_CHANNELS;
                obs[obs_idx + 0] = 1.0f;
                obs[obs_idx + 1] = env->critical_mass[idx] / 4.0f;

                if (env->owner[idx] == player) {
                    obs[obs_idx + 2] = env->orbs[idx] / 4.0f;
                } else if (env->owner[idx] == -player) {
                    obs[obs_idx + 3] = env->orbs[idx] / 4.0f;
                }
            }
        }

        int self_total = player == RED_PLAYER ? red_total : green_total;
        int opp_total = player == RED_PLAYER ? green_total : red_total;
        int base = MAX_CELLS * OBS_CHANNELS;
        obs[base + 0] = env->rows / (float)MAX_ROWS;
        obs[base + 1] = env->cols / (float)MAX_COLS;
        obs[base + 2] = self_total / (area * 4.0f);
        obs[base + 3] = opp_total / (area * 4.0f);
        obs[base + 4] = env->current_player == player ? 1.0f : 0.0f;
    }
}

static void compute_action_masks(ChainEnv* env) {
    for (int slot = 0; slot < env->num_agents; slot++) {
        memset(env->action_mask_ptr[slot], 0, MAX_ACTIONS);
    }

    if (env->end_game) {
        for (int slot = 0; slot < env->num_agents; slot++) {
            env->action_mask_ptr[slot][NOOP_ACTION] = 1;
        }
        return;
    }

    if (!is_selfplay(env)) {
        unsigned char* mask = env->action_mask_ptr[0];
        for (int action = 0; action < env->rows * env->cols; action++) {
            int idx = env->action_cell[action];
            if (env->owner[idx] == 0 || env->owner[idx] == RED_PLAYER) {
                mask[action] = 1;
            }
        }
        return;
    }

    int active_slot = slot_for_player(env, env->current_player);
    for (int slot = 0; slot < env->num_agents; slot++) {
        unsigned char* mask = env->action_mask_ptr[slot];
        if (slot != active_slot) {
            mask[NOOP_ACTION] = 1;
            continue;
        }

        for (int action = 0; action < env->rows * env->cols; action++) {
            int owner = env->owner[env->action_cell[action]];
            if (owner == 0 || owner == env->current_player) {
                mask[action] = 1;
            }
        }
    }
}

static void place_orb(ChainEnv* env, int board_idx, int player) {
    env->owner[board_idx] = player;
    env->orbs[board_idx] += 1;
}

static int timeout_winner(ChainEnv* env) {
    int red_total = 0;
    int green_total = 0;
    count_orbs(env, &red_total, &green_total);

    if (red_total > green_total) {
        return RED_PLAYER;
    }
    if (green_total > red_total) {
        return GREEN_PLAYER;
    }
    return 0;
}

static int choose_opponent_action(ChainEnv* env) {
    int legal_actions[MAX_CELLS];
    int legal_count = 0;
    for (int action = 0; action < env->rows * env->cols; action++) {
        int idx = env->action_cell[action];
        if (env->owner[idx] == 0 || env->owner[idx] == GREEN_PLAYER) {
            legal_actions[legal_count++] = action;
        }
    }

    if (env->opponent_policy == RANDOM_OPPONENT) {
        return legal_actions[rand_r(&env->rng) % legal_count];
    }

    float scores[MAX_CELLS];
    float best_score = -1.0e9f;
    for (int i = 0; i < legal_count; i++) {
        int action = legal_actions[i];
        int idx = env->action_cell[action];
        int critical = env->critical_mass[idx];
        int existing = env->orbs[idx];
        int owner = env->owner[idx];
        float score = owner == GREEN_PLAYER ? 3.1f : 1.0f;
        int friendly_neighbors = 0;
        int enemy_neighbors = 0;

        score += (5 - critical) * 0.35f;
        score += existing + 1 >= critical ? 4.0f : 1.2f / (critical - existing);
        for (int n = 0; n < env->neighbor_count[idx]; n++) {
            int neighbor = env->neighbors[idx][n];
            int neighbor_owner = env->owner[neighbor];
            int neighbor_orbs = env->orbs[neighbor];
            int neighbor_mass = env->critical_mass[neighbor];
            if (neighbor_owner == GREEN_PLAYER) {
                friendly_neighbors += 1;
                score += neighbor_orbs == neighbor_mass - 1 ? 1.1f : 0.32f * neighbor_orbs;
            } else if (neighbor_owner == RED_PLAYER) {
                enemy_neighbors += 1;
                score += neighbor_orbs == neighbor_mass - 1 ? 2.2f : 0.15f * neighbor_orbs;
                if (owner == 0 && env->move_count < 10) {
                    score -= 0.35f;
                }
            }
        }
        if (owner == 0 && env->move_count < 8) {
            score += 0.18f * friendly_neighbors;
            score -= 0.20f * enemy_neighbors;
        } else if (owner == 0) {
            score += 0.10f * friendly_neighbors;
        }

        scores[i] = score;
        if (score > best_score) {
            best_score = score;
        }
    }

    float margin = env->move_count < 8 ? 0.95f : 0.55f;
    float threshold = best_score - margin;
    float total_weight = 0.0f;
    for (int i = 0; i < legal_count; i++) {
        if (scores[i] >= threshold) {
            total_weight += 0.15f + (scores[i] - threshold);
        }
    }

    float sample = ((float)(rand_r(&env->rng) % 100000) / 100000.0f) * total_weight;
    int last_candidate = legal_actions[0];
    for (int i = 0; i < legal_count; i++) {
        if (scores[i] < threshold) {
            continue;
        }
        last_candidate = legal_actions[i];
        sample -= 0.15f + (scores[i] - threshold);
        if (sample <= 0.0f) {
            return last_candidate;
        }
    }

    return last_candidate;
}

static void finish_game(ChainEnv* env, int winner, int invalid_player) {
    for (int slot = 0; slot < env->num_agents; slot++) {
        int player = player_for_slot(env, slot);
        float reward = env->loss_reward;
        if (winner == 0) {
            reward = 0.0f;
        } else if (player == winner) {
            reward = env->win_reward;
        } else if (player == invalid_player) {
            reward = env->invalid_move_reward;
        }
        *env->reward_ptr[slot] = reward;
        *env->terminal_ptr[slot] = 1.0f;
    }

    env->winner = winner;
    if (env->num_agents == 2 && env->tag > 0 && env->tag <= MAX_BANKS) {
        float primary_score = 0.0f;
        if (winner == 0) {
            primary_score = 0.5f;
        } else if (player_for_slot(env, 0) == winner) {
            primary_score = 1.0f;
        }
        int bank = env->tag - 1;
        env->log.hist_score_bank[bank] += primary_score;
        env->log.hist_n_bank[bank] += 1.0f;
        env->log.hist_score += primary_score;
        env->log.hist_n += 1.0f;
        env->boundary_reached = 1;
    }

    env->end_game = 1;
    compute_action_masks(env);

    float reward = *env->reward_ptr[0];
    env->log.perf += reward > 0.0f ? 1.0f : (reward == 0.0f ? 0.5f : 0.0f);
    env->log.score += reward;
    env->log.episode_return += reward;
    env->log.episode_length += (float)env->tick;
    env->log.chain_bursts += (float)env->last_chain_bursts;
    env->log.invalid_rate += invalid_player != 0 ? 1.0f : 0.0f;
    if (env->num_agents == 2) {
        if (winner == 0) {
            env->log.slot_0_score += 0.5f;
            env->log.slot_1_score += 0.5f;
            env->log.draw_rate += 1.0f;
        } else if (slot_for_player(env, winner) == 0) {
            env->log.slot_0_score += 1.0f;
        } else {
            env->log.slot_1_score += 1.0f;
        }
    }
    env->log.n += 1.0f;
}

static ResolveStats execute_turn(
        ChainEnv* env, int player, int action, float start_delay) {
    int board_idx = env->action_cell[action];
    env->turns_taken[player_index(player)] += 1;
    env->move_count += 1;
    if (player == RED_PLAYER) {
        env->last_player_action = action;
    } else {
        env->last_env_action = action;
    }

    place_orb(env, board_idx, player);
    if (env->client != NULL) {
        record_snapshot(env, start_delay + PLACEMENT_DURATION);
    }
    start_delay += PLACEMENT_DURATION;
    ResolveStats stats = {.end_time = start_delay};
    bool render = env->client != NULL;
    int recorded_waves = 0;
    bool truncated_render = false;

    for (int wave = 0;; wave++) {
        uint8_t unstable[MAX_CELLS];
        int unstable_count = 0;

        for (int action = 0; action < env->rows * env->cols; action++) {
            int idx = env->action_cell[action];
            if (env->owner[idx] == player && env->orbs[idx] >= env->critical_mass[idx]) {
                unstable[unstable_count++] = idx;
            }
        }

        if (unstable_count == 0) {
            break;
        }

        bool record_wave = render && wave < MAX_RENDER_WAVES;
        float delay = start_delay + recorded_waves * WAVE_DELAY;
        for (int i = 0; i < unstable_count; i++) {
            int idx = unstable[i];
            env->orbs[idx] -= env->critical_mass[idx];
            if (env->orbs[idx] == 0) {
                env->owner[idx] = 0;
            }

            stats.bursts += 1;
            if (record_wave) {
                Client* client = env->client;
                if (client->burst_count < MAX_RENDER_BURSTS) {
                    client->bursts[client->burst_count++] = (BurstAnim) {
                        .row = idx / MAX_COLS,
                        .col = idx % MAX_COLS,
                        .color = player,
                        .delay = delay,
                        .duration = BURST_DURATION,
                    };
                }
                float end_time = delay + BURST_DURATION;
                if (end_time > client->animation_total) {
                    client->animation_total = end_time;
                }
            }
        }

        for (int i = 0; i < unstable_count; i++) {
            int idx = unstable[i];
            for (int n = 0; n < env->neighbor_count[idx]; n++) {
                int neighbor = env->neighbors[idx][n];
                place_orb(env, neighbor, player);
                if (record_wave) {
                    Client* client = env->client;
                    if (client->transfer_count < MAX_RENDER_TRANSFERS) {
                        client->transfers[client->transfer_count++] = (OrbTransferAnim) {
                            .src_row = idx / MAX_COLS,
                            .src_col = idx % MAX_COLS,
                            .dst_row = neighbor / MAX_COLS,
                            .dst_col = neighbor % MAX_COLS,
                            .color = player,
                            .delay = delay,
                            .duration = TRANSFER_DURATION,
                        };
                    }
                    float end_time = delay + TRANSFER_DURATION;
                    if (end_time > client->animation_total) {
                        client->animation_total = end_time;
                    }
                }
            }
        }

        if (record_wave) {
            float reveal_time = delay + fmaxf(BURST_DURATION, TRANSFER_DURATION);
            record_snapshot(env, reveal_time);
            stats.end_time = reveal_time;
            recorded_waves += 1;
        } else if (render) {
            truncated_render = true;
        }

        // Match the mobile game: once a player has been wiped out after both
        // sides have taken at least one turn, the move ends immediately instead
        // of continuing a single-color avalanche.
        if (env->turns_taken[0] > 0 && env->turns_taken[1] > 0) {
            bool red_present = false;
            bool green_present = false;
            for (int action = 0; action < env->rows * env->cols; action++) {
                int owner = env->owner[env->action_cell[action]];
                red_present |= owner == RED_PLAYER;
                green_present |= owner == GREEN_PLAYER;
                if (red_present && green_present) {
                    break;
                }
            }
            if (red_present && !green_present) {
                stats.winner = RED_PLAYER;
            } else if (green_present && !red_present) {
                stats.winner = GREEN_PLAYER;
            }
            if (stats.winner != 0) {
                break;
            }
        }
    }

    if (truncated_render) {
        float final_time = start_delay
            + recorded_waves * WAVE_DELAY
            + fmaxf(BURST_DURATION, TRANSFER_DURATION);
        record_snapshot(env, final_time);
        stats.end_time = final_time;
    }

    return stats;
}

static void init(ChainEnv* env) {
    for (int action = 0; action < env->rows * env->cols; action++) {
        int row = action / env->cols;
        int col = action % env->cols;
        int idx = board_index(row, col);
        env->action_cell[action] = idx;

        uint8_t* neighbors = env->neighbors[idx];
        int count = 0;
        if (row > 0) {
            neighbors[count++] = board_index(row - 1, col);
        }
        if (row + 1 < env->rows) {
            neighbors[count++] = board_index(row + 1, col);
        }
        if (col > 0) {
            neighbors[count++] = board_index(row, col - 1);
        }
        if (col + 1 < env->cols) {
            neighbors[count++] = board_index(row, col + 1);
        }
        env->neighbor_count[idx] = count;
        env->critical_mass[idx] = count;
    }

    // Stable cells hold at most critical_mass - 1 orbs.
    int stable_capacity = 3 * env->rows * env->cols - 2 * env->rows - 2 * env->cols;
    if (env->max_steps > stable_capacity) {
        env->max_steps = stable_capacity;
    }
    env->tick = 0;
    env->move_count = 0;
    env->end_game = 0;
    env->turns_taken[0] = 0;
    env->turns_taken[1] = 0;
    env->current_player = RED_PLAYER;
    env->last_player_action = -1;
    env->last_env_action = -1;
    env->last_chain_bursts = 0;
    env->winner = 0;
    set_slot_players(env, RED_PLAYER, GREEN_PLAYER);
}

static void c_close(ChainEnv* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

static void c_reset(ChainEnv* env) {
    memset(env->owner, 0, sizeof(env->owner));
    memset(env->orbs, 0, sizeof(env->orbs));
    env->tick = 0;
    env->move_count = 0;
    env->end_game = 0;
    env->turns_taken[0] = 0;
    env->turns_taken[1] = 0;
    env->current_player = RED_PLAYER;
    env->last_player_action = -1;
    env->last_env_action = -1;
    env->last_chain_bursts = 0;
    env->winner = 0;
    if (is_selfplay(env) && (rand_r(&env->rng) & 1) != 0) {
        set_slot_players(env, GREEN_PLAYER, RED_PLAYER);
    } else {
        set_slot_players(env, RED_PLAYER, GREEN_PLAYER);
    }
    zero_outputs(env);
    if (env->client != NULL) {
        reset_animations(env->client);
        clear_invalid_hint(env->client);
    }

    if (!is_selfplay(env) && (rand_r(&env->rng) & 1u) != 0u) {
        int opponent_action = choose_opponent_action(env);
        if (env->client != NULL) {
            begin_animation(env);
        }
        ResolveStats stats = execute_turn(
            env, GREEN_PLAYER, opponent_action, 0.0f);
        env->last_chain_bursts = stats.bursts;
        env->current_player = RED_PLAYER;
    }

    compute_observations(env);
    compute_action_masks(env);
}

static void c_step(ChainEnv* env) {
    env->tick += 1;
    zero_outputs(env);

    if (env->end_game) {
        c_reset(env);
        return;
    }

    env->last_chain_bursts = 0;

    if (is_selfplay(env)) {
        int active_slot = slot_for_player(env, env->current_player);
        int action = (int)*env->action_ptr[active_slot];
        if (env->current_player == RED_PLAYER) {
            env->last_player_action = action;
        } else {
            env->last_env_action = action;
        }

        if (!is_legal_move(env, action, env->current_player)) {
            compute_observations(env);
            finish_game(env, -env->current_player, env->current_player);
            return;
        }

        if (env->client != NULL) {
            begin_animation(env);
        }
        ResolveStats stats = execute_turn(env, env->current_player, action, 0.0f);
        env->last_chain_bursts = stats.bursts;

        int winner = stats.winner;
        if (winner == 0 && env->move_count >= env->max_steps) {
            winner = timeout_winner(env);
        }

        if (winner != 0 || env->move_count >= env->max_steps) {
            compute_observations(env);
            finish_game(env, winner, 0);
            return;
        }

        env->current_player = -env->current_player;
        compute_observations(env);
        compute_action_masks(env);
        return;
    }

    int action = (int)*env->action_ptr[0];
    env->last_env_action = -1;
    if (!is_legal_move(env, action, RED_PLAYER)) {
        env->last_player_action = action;
        compute_observations(env);
        finish_game(env, GREEN_PLAYER, RED_PLAYER);
        return;
    }

    if (env->client != NULL) {
        begin_animation(env);
    }
    ResolveStats player_stats = execute_turn(env, RED_PLAYER, action, 0.0f);
    env->last_chain_bursts = player_stats.bursts;

    int winner = player_stats.winner;
    if (winner == 0 && env->move_count >= env->max_steps) {
        winner = timeout_winner(env);
    }
    if (winner != 0 || env->move_count >= env->max_steps) {
        compute_observations(env);
        finish_game(env, winner, 0);
        return;
    }

    int env_action = choose_opponent_action(env);
    float env_start = player_stats.end_time + TURN_GAP;
    ResolveStats env_stats = execute_turn(
        env, GREEN_PLAYER, env_action, env_start);
    env->last_chain_bursts += env_stats.bursts;

    winner = env_stats.winner;
    if (winner == 0 && env->move_count >= env->max_steps) {
        winner = timeout_winner(env);
    }
    env->current_player = RED_PLAYER;
    compute_observations(env);
    if (winner != 0 || env->move_count >= env->max_steps) {
        finish_game(env, winner, 0);
        return;
    }

    compute_action_masks(env);
}

static Client* make_client(void) {
    Client* client = calloc(1, sizeof(Client));
    client->screen_width = 980;
    client->screen_height = 860;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(client->screen_width, client->screen_height, "PufferLib Chain Reaction");
    SetTargetFPS(60);
    return client;
}

static Color player_color(int player) {
    return player == RED_PLAYER
        ? (Color){255, 80, 88, 255}
        : (Color){77, 231, 125, 255};
}

static float ease_out(float t) {
    float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
}

static void draw_orb(Vector2 center, float radius, Color color, float alpha) {
    Color shadow = (Color){0, 0, 0, (unsigned char)(100.0f * alpha)};
    Color base = color;
    base.a = (unsigned char)(255.0f * alpha);
    Color dark = (Color) {
        (unsigned char)(color.r * 0.60f),
        (unsigned char)(color.g * 0.60f),
        (unsigned char)(color.b * 0.60f),
        color.a,
    };
    dark.a = (unsigned char)(220.0f * alpha);
    Color highlight = (Color){255, 255, 255, (unsigned char)(180.0f * alpha)};

    DrawCircleV((Vector2){center.x + radius * 0.14f, center.y + radius * 0.18f}, radius * 1.05f, shadow);
    DrawCircleV(center, radius, dark);
    DrawCircleV((Vector2){center.x - radius * 0.08f, center.y - radius * 0.08f}, radius * 0.88f, base);
    DrawCircleV((Vector2){center.x - radius * 0.30f, center.y - radius * 0.34f}, radius * 0.32f, highlight);
}

static void c_render(ChainEnv* env) {
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

    layout_board(env);
    float dt = GetFrameTime();
    env->client->invalid_hint_time = fmaxf(0.0f, env->client->invalid_hint_time - dt);
    env->client->animation_clock = fminf(
        env->client->animation_total,
        env->client->animation_clock + dt);

    BeginDrawing();
    {
        Client* client = env->client;
        Color bg = (Color){7, 20, 25, 255};
        Color frame = (Color){16, 47, 57, 255};
        Color inner = (Color){18, 69, 84, 255};
        Color grid_line = (Color){111, 214, 232, 55};
        int8_t display_owner[MAX_CELLS];
        uint16_t display_orbs[MAX_CELLS];

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
        copy_render_board(display_owner, display_orbs, env->owner, env->orbs);
        if (client->snapshot_count > 0) {
            copy_render_board(display_owner, display_orbs, client->base_owner, client->base_orbs);
            for (int i = 0; i < client->snapshot_count; i++) {
                if (client->snapshot_time[i] > client->animation_clock + 1.0e-4f) {
                    break;
                }
                copy_render_board(
                    display_owner,
                    display_orbs,
                    client->snapshot_owner[i],
                    client->snapshot_orbs[i]);
            }
        }

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

                int idx = board_index(row, col);
                int critical = env->critical_mass[idx];
                char label[8];
                snprintf(label, sizeof(label), "%d", critical);
                DrawText(
                    label,
                    (int)(x + client->cell_size - 16),
                    (int)(y + 8),
                    (int)(client->cell_size * 0.16f),
                    (Color){176, 228, 233, 135});

                int owner = display_owner[idx];
                int count = display_orbs[idx];
                if (owner != 0) {
                    float cx = client->board_x + (col + 0.5f) * client->cell_size;
                    float cy = client->board_y + (row + 0.5f) * client->cell_size;
                    float radius = client->cell_size * 0.18f;
                    Color color = player_color(owner);
                    if (count == 1) {
                        draw_orb((Vector2){cx, cy}, radius, color, 1.0f);
                    } else {
                        float orbit = client->cell_size * 0.12f;
                        float speed = owner == RED_PLAYER
                            ? RED_ORBIT_SPEED
                            : GREEN_ORBIT_SPEED;
                        for (int i = 0; i < count; i++) {
                            float angle = time * speed + 2.0f * PI * i / count;
                            float ox = cosf(angle) * orbit;
                            float oy = sinf(angle) * orbit * 0.65f;
                            draw_orb(
                                (Vector2){cx + ox, cy + oy},
                                radius * 0.92f,
                                color,
                                1.0f);
                        }
                    }
                }
            }
        }
    }
    {
        Client* client = env->client;
        float t = client->animation_clock;
        for (int i = 0; i < client->burst_count; i++) {
            BurstAnim* burst = &client->bursts[i];
            float local = t - burst->delay;
            if (local < 0.0f || local > burst->duration) {
                continue;
            }

            float p = local / burst->duration;
            float eased = ease_out(p);
            float cx = client->board_x + (burst->col + 0.5f) * client->cell_size;
            float cy = client->board_y + (burst->row + 0.5f) * client->cell_size;
            Color color = player_color(burst->color);
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

            float p = ease_out(local / anim->duration);
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
            draw_orb(pos, client->cell_size * 0.10f, player_color(anim->color), 0.95f);
        }
    }
    {
        Client* client = env->client;
        int red_total = 0;
        int green_total = 0;
        count_orbs(env, &red_total, &green_total);
        const char* mode_label = is_selfplay(env) ? "Self-play" : "Scripted duel";
        const char* red_label = is_selfplay(env) ? "Red" : "Red (you)";
        const char* green_label = is_selfplay(env) ? "Green" : "Green (scripted)";
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
            player_color(RED_PLAYER));
        DrawText(green_label, (int)(sidebar_x + 22.0f), (int)(sidebar_y + 92.0f), 22,
            player_color(GREEN_PLAYER));

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
        DrawText(red_text, stat_value_x, stat_y + 16, 23, player_color(RED_PLAYER));
        stat_y += stat_gap;

        DrawText("Green Orbs", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
        DrawText(green_text, stat_value_x, stat_y + 16, 23, player_color(GREEN_PLAYER));
        stat_y += stat_gap;

        DrawText("Last Chain Bursts", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
        DrawText(burst_text, stat_value_x, stat_y + 16, 23, (Color){236, 248, 250, 255});
        stat_y += stat_gap;

        DrawText("Cell Labels", stat_label_x, stat_y, 15, (Color){151, 194, 205, 230});
        DrawText("Critical mass", stat_value_x, stat_y + 16, 22, (Color){210, 233, 239, 255});
        stat_y += stat_gap;

        if (!is_selfplay(env) && client->invalid_hint_time > 0.0f) {
            DrawText("Invalid Move", stat_label_x, stat_y, 15, (Color){236, 183, 120, 240});
            DrawText("Place on empty or red cells", stat_value_x, stat_y + 16, 22,
                (Color){255, 216, 164, 255});
        }

        if (env->end_game) {
            const char* result = "Draw";
            Color result_color = (Color){235, 235, 235, 255};
            if (env->winner == RED_PLAYER) {
                result = "Red Wins";
                result_color = player_color(RED_PLAYER);
            } else if (env->winner == GREEN_PLAYER) {
                result = "Green Wins";
                result_color = player_color(GREEN_PLAYER);
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
    }
    EndDrawing();
}
