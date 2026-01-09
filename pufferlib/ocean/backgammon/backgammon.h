/*
 * Backgammon: A two-player board game environment for PufferLib
 * 
*/

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define NUM_POINTS 24           // Number of points on the board
#define NUM_CHECKERS 15         // Each player starts with 15 checkers
#define MAX_CHECKERS_PER_POINT 15

// Players
#define WHITE 0
#define BLACK 1

// Movement directions (white moves negative, black moves positive)
#define WHITE_DIRECTION -1
#define BLACK_DIRECTION 1

#define BAR_POSITION 0          // Action from bar
#define BEAR_OFF_POSITION 25    // Action to bear off

// Action space: source (0-25) * 4 dice options = 104 actions
// source 0 = bar, 1-24 = points, actions encode which die to use
#define NUM_ACTIONS 104

// Observation space size
// 24 points + 2 bar + 2 bear-off + 4 dice + 1 current_player + 2 can_bear_off = 35
#define OBSERVATION_SIZE 35

#define MAX_STEPS 5000
#define DEFAULT_LOG_INTERVAL 128

// Opponent difficulty
// 0.0 = fully greedy, 1.0 = fully random
#define OPPONENT_RANDOM_PROB 1.0f


typedef struct Log {
    float episode_return;
    float episode_length;       // Steps in episode
    float win_rate;             // Fraction of games won by white
    float black_win_rate;       // Fraction of games won by black (opponent)
    float avg_moves_per_turn;
    float hit_rate;             // Rate of hitting opponent blots
    float checkers_home;        // Avg checkers in home board at episode end
    float checkers_off;         // Avg checkers borne off at episode end
    float n;                    // Number of episodes
} Log;


typedef struct Client {
    int width;
    int height;
    int point_width;
    int point_height;
    bool initialized;
    // Textures would go here for full rendering
} Client;

// ============================================================================
// CBackgammon struct - main environment state
// ============================================================================

typedef struct CBackgammon {
    // ========================================================================
    // PufferLib I/O - pointers to shared memory buffers
    // ========================================================================
    float* observations;        // Neural network input [OBSERVATION_SIZE]
    int* actions;               // Agent's chosen action [1]
    float* rewards;             // Reward signal [1]
    unsigned char* terminals;   // Episode done flag [1]
    
    // ========================================================================
    // Logging and rendering
    // ========================================================================
    Log log;
    Client* client;
    
    // ========================================================================
    // Board state
    // ========================================================================
    
    // Points 1-24 (index 0 unused for clarity, use indices 1-24)
    // Positive values = white checkers, Negative values = black checkers
    int8_t board[NUM_POINTS + 1];  // board[1] to board[24]
    
    // Bar - checkers that have been hit and must re-enter
    // bar[WHITE] = white checkers on bar, bar[BLACK] = black checkers on bar
    int8_t bar[2];
    
    // Borne off - checkers that have been removed from the board
    // off[WHITE] = white checkers borne off, off[BLACK] = black checkers borne off
    int8_t off[2];
    
    // ========================================================================
    // Dice state
    // ========================================================================
    
    int8_t dice[4];
    // Number of dice available (2 normally, 4 for doubles)
    int8_t num_dice;
    // Number of dice already used this turn
    int8_t dice_used;
    bool dice_available[4];
    
    // ========================================================================
    // Turn state
    // ========================================================================
    
    // Current player (WHITE or BLACK)
    int8_t current_player;
    
    // Whether the current player must move from the bar first
    bool must_enter_from_bar;
    
    // ========================================================================
    // Episode tracking
    // ========================================================================
    
    // Current step count within episode
    int tick;
    
    // Cumulative return for current episode
    float episode_return;
    
    // Statistics for current episode
    int moves_this_episode;
    int hits_this_episode;
    int turns_this_episode;
    
} CBackgammon;

// ============================================================================
// Function declarations (to be implemented in backgammon.h below)
// ============================================================================

// Core PufferLib interface
void init(CBackgammon* env);
void c_reset(CBackgammon* env);
void c_step(CBackgammon* env);
void c_render(CBackgammon* env);
void c_close(CBackgammon* env);

// Game logic helpers
void roll_dice(CBackgammon* env);
int get_direction(int player);
bool in_home_board(int player, int point);
bool can_bear_off(CBackgammon* env, int player);
bool is_dst_available(CBackgammon *env, int position, int player);
bool is_legal_move(CBackgammon* env, int from, int die_index);
bool has_legal_moves(CBackgammon* env);
void make_move(CBackgammon* env, int from, int die_index);
bool check_win(CBackgammon* env, int player);
void compute_observations(CBackgammon* env);
void opponent_move(CBackgammon* env);

// Logging
void add_log(CBackgammon* env);

// Rendering (optional)
Client* make_client(CBackgammon* env);
void close_client(Client* client);


// Implementations

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

float randi(int min, int max) {
    return min + rand() % (max - min + 1);
}

void init(CBackgammon *env) {
    env->log = (Log){0};
    env->tick = 0;
    env->client = NULL;
}

void c_reset(CBackgammon* env) {
    env->log = (Log){0};
    for (int i = 0; i <= NUM_POINTS; i++) env->board[i] = 0;
    env->bar[WHITE] = 0; env->bar[BLACK] = 0;
    env->off[WHITE] = 0; env->off[BLACK] = 0;
    env->tick = 0;
    env->episode_return = 0.0;
    env->moves_this_episode = 0;
    env->hits_this_episode = 0;
    env->turns_this_episode = 0;
    env->current_player = rand() % 2;
    env->must_enter_from_bar = false;

    // white
    env->board[24] = 2;
    env->board[13] = 5;
    env->board[8] = 3;
    env->board[6] = 5;
    // black
    env->board[1] = -2;
    env->board[12] = -5;
    env->board[17] = -3;
    env->board[19] = -5;

    env->rewards[0] = 0.0;
    env->terminals[0] = 0;

    roll_dice(env);
    compute_observations(env);

}

void roll_dice(CBackgammon *env) {
    env->dice[0] = 1 + (rand() % 6);
    env->dice[1] = 1 + (rand() % 6);
    if (env->dice[0] == env->dice[1]) {
        env->dice[2] = env->dice[0];
        env->dice[3] = env->dice[0];
        env->num_dice = 4;
    } else {
        env->dice[2] = 0;
        env->dice[3] = 0;
        env->num_dice = 2;
    }
    env->dice_used = 0;
    for (int i = 0; i < 4; i++) {
        env->dice_available[i] = i < env->num_dice;
    }
}


int get_direction(int player) {
    return player == WHITE ? WHITE_DIRECTION: BLACK_DIRECTION;
}


bool in_home_board(int player, int point) {
    if (player == WHITE) {
        return point >= 1 && point <= 6;
    } else {
        return point >= 19 && point <= 24;
    }
}


bool can_bear_off(CBackgammon *env, int player) {
    for (int i = 1; i <= NUM_POINTS; i++) {
        if (player == WHITE && env->board[i] > 0 && !in_home_board(WHITE, i)) {
            return false;
        }
        if (player == BLACK && env->board[i] < 0 && !in_home_board(BLACK, i)) {
            return false;
        }
    }
    return env->bar[player] == 0;
}

bool is_dst_available(CBackgammon *env, int position, int player) {
    int8_t val = env->board[position];
    
    if (val == 0) return true;
    
    if (player == WHITE && val > 0) return true;
    if (player == BLACK && val < 0) return true;
    
    if (player == WHITE && val == -1) return true;
    if (player == BLACK && val == 1) return true;
    
    return false;
}

bool is_legal_move(CBackgammon *env, int from, int die_index) {
    int8_t cp = env->current_player;
    int8_t die_value = env->dice[die_index];
    int direction = get_direction(cp);

    if (!env->dice_available[die_index]) return false;
    
    if (env->bar[cp] > 0) {
        if (from != 0) return false;
        
        int entry = cp == WHITE ? NUM_POINTS + 1 - die_value : die_value;
        return is_dst_available(env, entry, cp);
    }
    
    if (from < 1 || from > NUM_POINTS) return false;
    
    int8_t fvalue = env->board[from];
    if (cp == WHITE && fvalue <= 0) return false;
    if (cp == BLACK && fvalue >= 0) return false;
    
    int dst = from + (die_value * direction);
    
    if (cp == WHITE && dst < 1) {
        return can_bear_off(env, cp);
    }
    if (cp == BLACK && dst > NUM_POINTS) {
        return can_bear_off(env, cp);
    }
    
    if (dst < 1 || dst > NUM_POINTS) return false;
    return is_dst_available(env, dst, cp);
}


bool has_legal_moves(CBackgammon *env) {
    for (int i = 0; i <= NUM_POINTS; i++) {
        for (int d = 0; d < env->num_dice; d++) {
            if (is_legal_move(env, i, d)) return true;
        }
    }
    return false;
}


void make_move(CBackgammon* env, int from, int die_index) {
    int8_t cp = env->current_player;
    int8_t die_value = env->dice[die_index];
    int direction = get_direction(cp);
    
    env->dice_available[die_index] = false;
    env->dice_used++;
    env->moves_this_episode++;
    
    if (from == 0) {
        env->bar[cp]--;
        int dst = (cp == WHITE) ? (NUM_POINTS + 1 - die_value) : die_value;
        
        if ((cp == WHITE && env->board[dst] == -1) ||
            (cp == BLACK && env->board[dst] == 1)) {
            env->bar[cp ^ 1]++;
            env->board[dst] = 0;
            env->hits_this_episode++;
        }
        
        env->board[dst] += (cp == WHITE) ? 1 : -1;
    } else {
        env->board[from] += (cp == WHITE) ? -1 : 1; // remove from board
        int dst = from + (die_value * direction);
        // check bear off
        if ((cp == WHITE && dst < 1) || (cp == BLACK && dst > NUM_POINTS)) {
            env->off[cp]++;
            return;
        }
        
        // check hit
        if ((cp == WHITE && env->board[dst] == -1) ||
            (cp == BLACK && env->board[dst] == 1)) {
            env->bar[cp ^ 1]++;
            env->board[dst] = 0;
            env->hits_this_episode++;
        }
        
        env->board[dst] += (cp == WHITE) ? 1 : -1;
    }
}


bool check_win(CBackgammon* env, int player) {
    return env->off[player] == NUM_CHECKERS;
}


void compute_observations(CBackgammon* env) {
    float *obs = env->observations;
    int cur = 0;
    
    for (int i = 1; i <= NUM_POINTS; i++) {
        obs[cur++] = env->board[i] / (float)NUM_CHECKERS;
    }
    
    obs[cur++] = (float)env->bar[WHITE] / NUM_CHECKERS;
    obs[cur++] = (float)env->bar[BLACK] / NUM_CHECKERS;
    
    obs[cur++] = (float)env->off[WHITE] / NUM_CHECKERS;
    obs[cur++] = (float)env->off[BLACK] / NUM_CHECKERS;
    
    for (int i = 0; i < 4; i++) {
        obs[cur++] = env->dice_available[i] ? (float)env->dice[i] / 6.0f : 0.0f;
    }
    
    obs[cur++] = (float)env->current_player;
    
    obs[cur++] = can_bear_off(env, WHITE) ? 1.0f : 0.0f;
    obs[cur++] = can_bear_off(env, BLACK) ? 1.0f : 0.0f;
}


int score_move(CBackgammon* env, int from, int die_index) {
    int die_value = env->dice[die_index];
    int score = 0;
    
    if (from == 0) {
        score += 100;
        int dst = die_value;  // Black enters at die_value
        // Bonus for hitting white
        if (env->board[dst] == 1) score += 50;
        return score;
    }
    
    int dst = from + die_value;
    
    // Bearing off is highest priority
    if (dst > NUM_POINTS && can_bear_off(env, BLACK)) {
        score += 200;
        return score;
    }
    
    // Can't bear off and destination is off board
    if (dst > NUM_POINTS) return -1000;
    
    // Hitting a white checker is very good
    if (env->board[dst] == 1) {
        score += 80;
    }
    
    // Making a point (having 2+ checkers) is good for safety
    if (env->board[dst] == -1) {
        score += 30;
    }
    
    // Advancing toward home board is good
    score += dst;
    
    // Moving a lone checker (blot) to safety is good
    if (env->board[from] == -1) {
        score += 20;
    }
    
    // Prefer not to leave blots in opponent's home board
    if (env->board[from] == -2 && from >= 1 && from <= 6) {
        score -= 10;
    }
    
    return score;
}

void opponent_move(CBackgammon* env) {
    env->current_player = BLACK;
    roll_dice(env);
    env->turns_this_episode++;
    
    while (has_legal_moves(env)) {
        int chosen_from = -1;
        int chosen_die = -1;
        
        float r = (float)rand() / (float)RAND_MAX;
        if (r < OPPONENT_RANDOM_PROB) {
            int legal_moves[NUM_ACTIONS][2];  // [from, die_index]
            int num_legal = 0;
            
            for (int from = 0; from <= NUM_POINTS; from++) {
                for (int d = 0; d < env->num_dice; d++) {
                    if (is_legal_move(env, from, d)) {
                        legal_moves[num_legal][0] = from;
                        legal_moves[num_legal][1] = d;
                        num_legal++;
                    }
                }
            }
            
            if (num_legal > 0) {
                int pick = rand() % num_legal;
                chosen_from = legal_moves[pick][0];
                chosen_die = legal_moves[pick][1];
            }
        } else {
            // find best scoring legal move
            int best_score = -10000;
            
            for (int from = 0; from <= NUM_POINTS; from++) {
                for (int d = 0; d < env->num_dice; d++) {
                    if (is_legal_move(env, from, d)) {
                        int score = score_move(env, from, d);
                        if (score > best_score) {
                            best_score = score;
                            chosen_from = from;
                            chosen_die = d;
                        }
                    }
                }
            }
        }
        
        if (chosen_from < 0) break;  // no move found
        
        make_move(env, chosen_from, chosen_die);
        
        if (check_win(env, BLACK)) {
            return;
        }
    }
    
    env->current_player = WHITE;
    roll_dice(env);
}


void add_log(CBackgammon* env) {
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    
    env->log.win_rate += check_win(env, WHITE) ? 1.0f : 0.0f;
    env->log.black_win_rate += check_win(env, BLACK) ? 1.0f : 0.0f;
    
    if (env->turns_this_episode > 0) {
        env->log.avg_moves_per_turn += (float)env->moves_this_episode / env->turns_this_episode;
    }
    
    if (env->moves_this_episode > 0) {
        env->log.hit_rate += (float)env->hits_this_episode / env->moves_this_episode;
    }
    
    // Count checkers in home board and borne off for white
    int home_count = 0;
    for (int i = 1; i <= 6; i++) {
        if (env->board[i] > 0) home_count += env->board[i];
    }
    env->log.checkers_home += home_count;
    env->log.checkers_off += env->off[WHITE];
    
    env->log.n += 1;
}


void c_step(CBackgammon* env) {
    if (env->terminals[0]) {
        c_reset(env);
        return;
    }

    int action = env->actions[0];
    int from = action / 4;
    int die_index = action % 4;

    float reward = 0.0f;
    
    int old_off = env->off[WHITE];
    int old_bar_opponent = env->bar[BLACK];
    
    if (is_legal_move(env, from, die_index)) {
        make_move(env, from, die_index);
        
        // Reward shaping
        if (env->off[WHITE] > old_off) {
            reward += 0.05f;
        }
        if (env->bar[BLACK] > old_bar_opponent) {
            reward += 0.02f;
        }
    } else {
        reward -= 0.1f;
    }

    if (check_win(env, WHITE)) {
        reward = 1.0f;
        env->terminals[0] = 1;
        env->episode_return += reward;
        add_log(env);
        goto end;
    }

    if (env->dice_used >= env->num_dice || !has_legal_moves(env)) {
        opponent_move(env);
    }

    if (check_win(env, BLACK)) {
        reward = -1.0f;
        env->terminals[0] = 1;
        env->episode_return += reward;
        add_log(env);
        goto end;
    }

    if (env->tick >= MAX_STEPS) {
        env->terminals[0] = 1;
        env->episode_return += reward;
        add_log(env);
        goto end;
    }

    env->episode_return += reward;

end:
    env->rewards[0] = reward;
    compute_observations(env);
    env->tick++;
    
}


// Rendering logic

Client* make_client(CBackgammon* env) {
    return NULL;
}

void close_client(Client* client) {
}

void c_render(CBackgammon* env) {
}

void c_close(CBackgammon* env) {
    if (env->client) {
        close_client(env->client);
    }
}