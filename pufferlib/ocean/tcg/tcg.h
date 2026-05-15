#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"

#define HAND_SIZE 10
#define BOARD_SIZE 10
#define DECK_SIZE 60
#define STACK_SIZE 100
#define MAX_TURNS 32
#define DRAW_PENALTY -0.5f
#define WIN_REWARD 1.0f
#define LOSE_REWARD -1.0f

#define ACTION_ENTER 10
#define ACTION_NOOP 11

#define NUM_PLAYERS 2
#define OBS_SIZE (sizeof(Obs))

#define TO_USER true;
#define TO_STACK false;

typedef struct TCG TCG;
typedef bool (*call)(TCG*, unsigned char);
bool phase_untap(TCG* env, unsigned char atn);
bool phase_draw(TCG* env, unsigned char atn);
bool phase_play(TCG* env, unsigned char atn);
bool phase_attack(TCG* env, unsigned char atn);
bool phase_block(TCG* env, unsigned char atn);
void reset(TCG* env);
void draw_episode(TCG* env);

typedef struct Stack Stack;
struct Stack {
    call data[STACK_SIZE];
    int idx;
};

void push(Stack* stack, call fn) {
    assert(stack->idx < STACK_SIZE);
    stack->data[stack->idx] = fn;
    stack->idx += 1;
}

call pop(Stack* stack) {
    assert(stack->idx > 0);
    stack->idx -= 1;
    return stack->data[stack->idx];
}

call peek(Stack* stack) {
    assert(stack->idx > 0);
    return stack->data[stack->idx - 1];
}

typedef struct Card Card;
struct Card {
    int cost;
    int attack;
    int health;
    bool is_land;
    bool remove;
    bool tapped;
    bool attacking;
    int defending;
};

typedef struct CardArray CardArray;
struct CardArray {
    Card* cards;
    int length;
    int max;
};

CardArray* allocate_card_array(int max) {
    CardArray* hand = (CardArray*)calloc(1, sizeof(CardArray));
    hand->cards = (Card*)calloc(max, sizeof(Card));
    hand->max = max;
    return hand;
}

void free_card_array(CardArray* ary) {
    free(ary->cards);
    free(ary);
}

void condense_card_array(CardArray* ary) {
    int idx = 0;
    for (int i = 0; i < ary->length; i++) {
        if (!ary->cards[i].remove) {
            ary->cards[idx] = ary->cards[i];
            idx += 1;
        }
    }
    ary->length = idx;
}

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

struct TCG {
    CardArray* my_hand;
    CardArray* my_board;
    CardArray* my_deck;
    int my_health;
    int my_mana;
    bool my_land_played;

    CardArray* op_hand;
    CardArray* op_board;
    CardArray* op_deck;
    int op_health;
    int op_mana;
    bool op_land_played;
    bool debug_logs;

    Stack* stack;
    //bool attackers[BOARD_SIZE];
    //bool defenders[BOARD_SIZE][BOARD_SIZE];
    int block_idx;
    int turn;

    int tick;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
};

static inline void dbg(TCG* env, const char* fmt, ...) {
    if (!env || !env->debug_logs) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

static inline CardArray* player_hand(TCG* env, int idx) {
    return idx == 0 ? env->my_hand : env->op_hand;
}

static inline CardArray* player_board(TCG* env, int idx) {
    return idx == 0 ? env->my_board : env->op_board;
}

static inline CardArray* player_deck(TCG* env, int idx) {
    return idx == 0 ? env->my_deck : env->op_deck;
}

static inline int* player_health(TCG* env, int idx) {
    return idx == 0 ? &env->my_health : &env->op_health;
}

static inline int* player_mana(TCG* env, int idx) {
    return idx == 0 ? &env->my_mana : &env->op_mana;
}

static inline bool* player_land_played(TCG* env, int idx) {
    return idx == 0 ? &env->my_land_played : &env->op_land_played;
}

static inline int current_player(TCG* env) {
    return env->turn % NUM_PLAYERS;
}

void add_log(TCG* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void end_episode(TCG* env, int winner) {
    int loser = 1 - winner;
    env->rewards[winner] = WIN_REWARD;
    env->rewards[loser] = LOSE_REWARD;
    env->terminals[winner] = 1;
    env->terminals[loser] = 1;
    add_log(env);
    reset(env);
}

void draw_episode(TCG* env) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        env->rewards[player] = DRAW_PENALTY;
        env->terminals[player] = 1;
    }
    add_log(env);
    reset(env);
}

typedef struct {
    int turn;
    int my_health;
    int my_mana;
    int op_health;
    int op_mana;
    int op_hand_length;

    Card my_hand[HAND_SIZE];
    Card my_board[BOARD_SIZE];
    Card op_board[BOARD_SIZE];
} Obs;

void init_tcg(TCG* env) {
    env->stack = calloc(1, sizeof(Stack));
    env->my_hand = allocate_card_array(HAND_SIZE);
    env->op_hand = allocate_card_array(HAND_SIZE);
    env->my_board = allocate_card_array(BOARD_SIZE);
    env->op_board = allocate_card_array(BOARD_SIZE);
    env->my_deck = allocate_card_array(DECK_SIZE);
    env->op_deck = allocate_card_array(DECK_SIZE);
    env->debug_logs = getenv("TCG_DEBUG") != NULL;
}

void allocate_tcg(TCG* env) {
    init_tcg(env);
    env->observations = (unsigned char*)calloc(NUM_PLAYERS * OBS_SIZE, sizeof(unsigned char));
    env->actions = (int*)calloc(NUM_PLAYERS, sizeof(int));
    env->rewards = (float*)calloc(NUM_PLAYERS, sizeof(float));
    env->terminals = (unsigned char*)calloc(NUM_PLAYERS, sizeof(unsigned char));
}

void free_tcg(TCG* env) {
    free_card_array(env->my_hand);
    free_card_array(env->op_hand);
    free_card_array(env->my_board);
    free_card_array(env->op_board);
    free_card_array(env->my_deck);
    free_card_array(env->op_deck);
    free(env->stack);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void randomize_deck(CardArray* deck) {
    for (int i = 0; i < deck->length; i++) {
        deck->cards[i].defending = -1;
        if (rand() % 3 == 0) {
            deck->cards[i].is_land = true;
        } else {
            int cost = rand() % 6;
            deck->cards[i].cost = cost;
            deck->cards[i].attack = cost + 1;
            deck->cards[i].health = cost + 1;
        }
    }
}

bool draw_card(TCG* env, int player_idx, CardArray* deck, CardArray* hand) {
    if (deck->length == 0) {
        end_episode(env, 1 - player_idx);
        return false;
    }
    if (hand->length == hand->max) {
        return true;
    }
    Card card = deck->cards[deck->length - 1];
    hand->cards[hand->length] = card;
    deck->length -= 1;
    hand->length += 1;
    return true;
}

bool can_attack(CardArray* board) {
    for (int i = 0; i < board->length; i++) {
        if (!board->cards[i].is_land) {
            return true;
        }
    }
    return false;
}

int tappable_mana(TCG* env) {
    CardArray* board = player_board(env, current_player(env));
    int tappable = 0;
    for (int i = 0; i < board->length; i++) {
        Card card = board->cards[i];
        if (card.is_land && !card.tapped) {
            tappable += 1;
        }
    }
    return tappable;
}

bool can_play(TCG* env) {
    int player = current_player(env);
    CardArray* hand = player_hand(env, player);
    int* mana = player_mana(env, player);
    bool* land_played = player_land_played(env, player);

    int min_cost = 99;
    for (int i = 0; i < hand->length; i++) {
        if (hand->cards[i].is_land && !*land_played) {
            return true;
        } else if (hand->cards[i].cost < min_cost) {
            min_cost = hand->cards[i].cost;
        }
    }

    int tappable = tappable_mana(env);
    return *mana + tappable >= min_cost;
}

bool phase_untap(TCG* env, unsigned char atn) {
    dbg(env, "PHASE_UNTAP\n");
    env->turn += 1;
    if (env->turn >= MAX_TURNS) {
        draw_episode(env);
        return TO_STACK;
    }

    int player = current_player(env);
    bool* land_played = player_land_played(env, player);
    *land_played = false;

    CardArray* board = player_board(env, player);

    int* mana = player_mana(env, player);
    *mana = 0;

    for (int i = 0; i < board->length; i++) {
        Card card = board->cards[i];
        if (card.is_land && card.tapped) {
            board->cards[i].tapped = false;
        }
    }
    
    push(env->stack, phase_draw);
    return TO_STACK;
}

bool phase_draw(TCG* env, unsigned char atn) {
    dbg(env, "PHASE_DRAW\n");
    int player = current_player(env);
    CardArray* deck = player_deck(env, player);
    CardArray* hand = player_hand(env, player);
    if (!draw_card(env, player, deck, hand)) {
        return TO_STACK;
    }
    push(env->stack, phase_play);
    return TO_STACK;
}

bool phase_play(TCG* env, unsigned char atn) {
    dbg(env, "PHASE_PLAY\n");
    int player = current_player(env);
    CardArray* hand = player_hand(env, player);
    CardArray* board = player_board(env, player);
    int* mana = player_mana(env, player);
    bool* land_played = player_land_played(env, player);

    if (board->length == BOARD_SIZE) {
        dbg(env, "\t Board full\n");
        push(env->stack, phase_attack);
        return TO_STACK;
    }

    if (!can_play(env)) {
        dbg(env, "\t No valid moves\n");
        push(env->stack, phase_attack);
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_play);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        push(env->stack, phase_attack);
        return TO_STACK;
    } else if (atn >= hand->length) {
        dbg(env, "\t Invalid action: %i\n. Hand length: %i\n", atn, hand->length);
        push(env->stack, phase_play);
        return TO_USER;
    }

    Card card = hand->cards[atn];
    if (card.is_land) {
        if (*land_played) {
            dbg(env, "\t Already played land this turn\n");
            push(env->stack, phase_play);
            return TO_USER;
        }
        board->cards[board->length] = card;
        board->length += 1;
        *land_played = true;
        hand->cards[atn].remove = true;
        condense_card_array(hand);
        dbg(env, "\t Land played\n");
        push(env->stack, phase_play);
        return TO_USER;
    }

    if (card.cost > *mana + tappable_mana(env)) {
        dbg(env, "\t Not enough mana\n");
        push(env->stack, phase_play);
        return TO_USER;
    }

    // Auto tap lands?
    for (int i = 0; i < board->length; i++) {
        if (card.cost <= *mana) {
            break;
        }
        Card card = board->cards[i];
        if (card.is_land && !card.tapped) {
            *mana += 1;
            board->cards[i].tapped = true;
        }
    }

    assert(*mana >= card.cost);
    *mana -= card.cost;
    board->cards[board->length] = card;
    board->length += 1;
    hand->cards[atn].remove = true;
    condense_card_array(hand);
    dbg(env, "\t Card played\n");
    push(env->stack, phase_play);
    return TO_USER;
}

bool phase_attack(TCG* env, unsigned char atn) {
    dbg(env, "PHASE_ATTACK\n");
    int attacker = current_player(env);
    CardArray* board = player_board(env, attacker);

    if (!can_attack(board)) {
        dbg(env, "\t No valid attacks. Phase end\n");
        push(env->stack, phase_untap);
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_attack);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        dbg(env, "\t Attacks confirmed. Phase end\n");
        push(env->stack, phase_block);
        return TO_STACK;
    } else if (atn >= board->length) {
        dbg(env, "\t Invalid action %i\n", atn);
        push(env->stack, phase_attack);
        return TO_USER;
    } else if (board->cards[atn].is_land) {
        dbg(env, "\t Cannot attack with land\n");
        push(env->stack, phase_attack);
        return TO_USER;
    } else {
        dbg(env, "\t Setting attacker %i\n", atn);
        board->cards[atn].attacking = !board->cards[atn].attacking;
        push(env->stack, phase_attack);
        return TO_USER;
    }
}

bool phase_block(TCG* env, unsigned char atn) {
    dbg(env, "PHASE_BLOCK\n");
    int attacker_player = current_player(env);
    int defender_player = 1 - attacker_player;
    CardArray* defender_board = player_board(env, defender_player);
    CardArray* board = player_board(env, attacker_player);
    int* health = player_health(env, defender_player);

    while (env->block_idx < board->length && !board->cards[env->block_idx].attacking) {
        dbg(env, "\t Skipping block for %i (not attacking)\n", env->block_idx);
        env->block_idx++;
    }
    
    bool can_block = false;
    for (int i = 0; i < defender_board->length; i++) {
        Card* card = &defender_board->cards[i];
        if (card->is_land) {
            continue;
        }
        if (card->defending == -1 || card->defending == env->block_idx) {
            can_block = true;
            dbg(env, "\t Can block with %i\n", i);
            break;
        }
    }
    if (!can_block) {
        env->block_idx = board->length;
    }
 
    if (env->block_idx == board->length) {
        dbg(env, "\t Attacker board length: %i\n", board->length);
        for (int atk = 0; atk < board->length; atk++) {
            dbg(env, "\t Resolving %i\n", atk);
            Card* attacker = &board->cards[atk];
            if (!attacker->attacking) {
                dbg(env, "\t Not attacking\n");
                continue;
            }
            int attacker_attack = attacker->attack;
            int attacker_health = attacker->health;
            for (int def = 0; def < defender_board->length; def++) {
                Card* defender = &defender_board->cards[def];
                if (defender->defending != atk) {
                    continue;
                }
                if (attacker_attack >= defender->health) {
                    attacker_attack -= defender->health;
                    attacker_health -= defender->attack;
                    defender->health = 0;
                    defender->remove = true;
                } else {
                    attacker_health -= defender->attack;
                    attacker_attack = 0;
                }
                if (attacker_health <= 0) {
                    attacker->remove = true;
                    break;
                }
            }
            int damage_to_player = attacker_attack;
            dbg(env, "\t Reducing health by %i\n", damage_to_player);
            *health -= damage_to_player;
        }

        if (*health <= 0) {
            int winner = attacker_player;
            end_episode(env, winner);
            return TO_STACK;
        }

        condense_card_array(env->my_board);
        condense_card_array(env->op_board);

        CardArray* defender_deck = player_deck(env, defender_player);
        CardArray* defender_hand = player_hand(env, defender_player);
        if (!draw_card(env, defender_player, defender_deck, defender_hand)) {
            return TO_STACK;
        }

        for (int i = 0; i < board->length; i++) {
            board->cards[i].attacking = false;
        }
        for (int i = 0; i < defender_board->length; i++) {
            defender_board->cards[i].defending = -1;
        }
        dbg(env, "\t Set block idx to 0\n");
        env->block_idx = 0;
        push(env->stack, phase_untap);
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_block);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        dbg(env, "\t Manual block confirm %i\n", env->block_idx);
        env->block_idx++;
        push(env->stack, phase_block);
        return TO_STACK;
    } else if (atn >= defender_board->length) {
        dbg(env, "\t Invalid block action %i\n", atn);
        push(env->stack, phase_block);
        return TO_USER;
    } else if (defender_board->cards[atn].is_land) {
        dbg(env, "\t Cannot block with land\n");
        push(env->stack, phase_block);
        return TO_USER;
    }

    for (int i = 0; i < env->block_idx; i++) {
        if (defender_board->cards[atn].defending == i) {
            dbg(env, "\t Already blocked\n");
            push(env->stack, phase_block);
            return TO_USER;
        }
    }
    dbg(env, "\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = &defender_board->cards[atn];
    if (card->defending == env->block_idx) {
        card->defending = -1;
    } else {
        card->defending = env->block_idx;
    }
    push(env->stack, phase_block);
    return TO_USER;
}

void write_card(unsigned char *obs, Card *c) {
    obs[0] = c->is_land;
    obs[1] = c->cost;
    obs[2] = c->attack;
    obs[3] = c->health;
    obs[4] = c->tapped;
    obs[5] = c->attacking;
    obs[6] = c->defending;
}

void update_observations(TCG* env) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        unsigned char* obs = env->observations + player * OBS_SIZE;
        int idx = 0;
        int opponent = 1 - player;
        int turn_player = current_player(env);

        CardArray* self_hand = player_hand(env, player);
        CardArray* self_board = player_board(env, player);
        CardArray* opp_board = player_board(env, opponent);
        CardArray* opp_hand = player_hand(env, opponent);

        obs[idx++] = *player_health(env, player);
        obs[idx++] = *player_health(env, opponent);
        obs[idx++] = (turn_player == player);
        obs[idx++] = *player_mana(env, player);
        obs[idx++] = *player_mana(env, opponent);
        obs[idx++] = self_hand->length;
        obs[idx++] = self_board->length;
        obs[idx++] = opp_board->length;

        for (int i = 0; i < HAND_SIZE; i++) {
            if (i < self_hand->length) {
                write_card(&obs[idx], &self_hand->cards[i]);
            } else {
                memset(&obs[idx], 0, 7);
            }
            idx += 7;
        }

        for (int i = 0; i < BOARD_SIZE; i++) {
            if (i < self_board->length) {
                write_card(&obs[idx], &self_board->cards[i]);
            } else {
                memset(&obs[idx], 0, 7);
            }
            idx += 7;
        }

        for (int i = 0; i < BOARD_SIZE; i++) {
            if (i < opp_board->length) {
                write_card(&obs[idx], &opp_board->cards[i]);
            } else {
                memset(&obs[idx], 0, 7);
            }
            idx += 7;
        }

        obs[idx++] = opp_hand->length;

    }
}

void step(TCG* env, unsigned char atn) {
    dbg(env, "Turn: %i (player %i), Action: %i\n", env->turn, current_player(env), atn);
    env->rewards[0] = 0;
    env->rewards[1] = 0;
    env->tick += 1;
    while (true) {
        call fn = pop(env->stack);
        bool return_to_user = fn(env, atn);
        if (return_to_user) {
            update_observations(env);
            return;
        }
        atn = ACTION_NOOP;
    }
}

void reset(TCG* env) {
    env->my_deck->length = DECK_SIZE;
    env->op_deck->length = DECK_SIZE;
    env->my_hand->length = 0;
    env->op_hand->length = 0;
    env->my_board->length = 0;
    env->op_board->length = 0;
    env->my_health = 20;
    env->op_health = 20;
    env->my_mana = 0;
    env->op_mana = 0;
    env->tick = 0;
    env->op_mana = 0;
    env->tick = 0;
    memset(env->observations, 0, NUM_PLAYERS * OBS_SIZE * sizeof(unsigned char));
    randomize_deck(env->my_deck);
    randomize_deck(env->op_deck);
    env->turn = rand() % 2;
    for (int i = 0; i < 5; i++) {
        draw_card(env, 0, env->my_deck, env->my_hand);
        draw_card(env, 1, env->op_deck, env->op_hand);
    }
    push(env->stack, phase_draw);
    step(env, ACTION_NOOP);
}

void init_client(TCG* env) {
    InitWindow(1080, 720, "Puffer the Schooling TCG");
    SetTargetFPS(60);
}

void close_client(TCG* env) {
    CloseWindow();
}

int card_x(int col, int n) {
    int cards_width = 72*n;
    int offset = 72*col;
    return GetScreenWidth()/2 - cards_width/2 + offset;
}

int card_y(int row) {
    return 64 + (128 + 20)*row;
}

void render_card(Card* card, int x, int y, Color color) {
    DrawRectangle(x, y, 64, 128, color);
    if (card->is_land) {
        DrawText("Land", x + 16, y + 40, 16, WHITE);
    } else {
        DrawText(TextFormat("%i", card->cost), x + 32, y+16, 20, WHITE);
        DrawText(TextFormat("%i", card->attack), x + 32, y + 40, 20, WHITE);
        DrawText(TextFormat("%i", card->health), x + 32, y + 64, 20, WHITE);
    }
}

void render_label(int x, int y, int idx) {
    DrawText(TextFormat("%i", (idx+1)%10), x+32, y+96, 20, YELLOW);
}

void render(TCG* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
   
    for (int i = 0; i < env->my_hand->length; i++) {
        Card card = env->my_hand->cards[i];
        int x = card_x(i, env->my_hand->length);
        int y = card_y(3);
        render_card(&card, x, y, RED);
        if (current_player(env) == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];
        int x = card_x(i, env->my_board->length);
        int y = card_y(2);
        if (card.attacking) {
            y -= 16;
        }
        Color color = (card.tapped) ? (Color){128, 0, 0, 255}: RED;
        render_card(&card, x, y, color);
        if (current_player(env) == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->op_board->length; i++) {
        Card card = env->op_board->cards[i];
        int x = card_x(i, env->op_board->length);
        int y = card_y(1);
        if (card.attacking) {
            y += 16;
        }
        Color color = (card.tapped) ? (Color){0, 0, 128, 255}: BLUE;
        render_card(&card, x, y, color);
    }

    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];
        if (card.defending == -1) {
            continue;
        }
        DrawLineEx(
            (Vector2){32+card_x(i, env->my_board->length), 64+card_y(2)},
            (Vector2){32+card_x(card.defending, env->op_board->length), 64+card_y(1)},
            3.0f, WHITE
        );
    }

    for (int i = 0; i < env->op_hand->length; i++) {
        Card card = env->op_hand->cards[i];
        int x = card_x(i, env->op_hand->length);
        int y = card_y(0);
        render_card(&card, x, y, BLUE);
    }

    int x = GetScreenWidth() - 128;
    int y = 32;

    call fn = peek(env->stack);
        if (fn == phase_draw) {
            DrawText("Draw", x, y, 20, WHITE);
        } else if (fn == phase_play) {
            DrawText("Play", x, y, 20, WHITE);
        } else if (fn == phase_attack) {
            DrawText("Attack", x, y, 20, WHITE);
        } else if (fn == phase_block) {
            DrawText("Block", x, y, 20, WHITE);
        } 

    DrawText(TextFormat("Health: %i", env->my_health), 32, 32, 20, WHITE);
    DrawText(TextFormat("Health: %i", env->op_health), 32, GetScreenHeight() - 64, 20, WHITE);

    EndDrawing();
}

void c_render(TCG* env) {
    if (!IsWindowReady()) {
        init_client(env);
    }
    render(env);
}

void c_step(TCG* env) {
    call phase = peek(env->stack);
    int actor = current_player(env);
    if (phase == phase_block) {
        actor = 1 - actor;
    }
    int action = env->actions[actor];
    if (action < 0 || action >= 12) {
        action = ACTION_NOOP;
    }
    step(env, (unsigned char)action);
}

void c_reset(TCG* env) {
    reset(env);
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(TCG* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
