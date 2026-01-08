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

#define MAX_STEPS 1000
#define DEFAULT_LOG_INTERVAL 128


typedef struct Log {
    float episode_return;
    float episode_length;       // Steps in episode
    float win_rate;             // Fraction of games won by white
    float avg_moves_per_turn;
    float hit_rate;             // Rate of hitting opponent blots
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
    
    // Dice values (1-6 each)
    // For doubles, all 4 entries have the same value
    int8_t dice[4];
    
    // Number of dice available (2 normally, 4 for doubles)
    int8_t num_dice;
    
    // Number of dice already used this turn
    int8_t dice_used;
    
    // Which specific dice are still available (for tracking after partial moves)
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
bool is_dst_available(CBackgammon* env, int player);
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
    if (!env->board[position] || env->board[position] == 1 || env->board[position] == -1) return true;
    if (player == WHITE) {
        return env->board[position] > 0;
    } else {
        return env->board[position] < 0;
    }

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