#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define HAND_SIZE 10
#define BOARD_SIZE 10
#define DECK_SIZE 60
#define STACK_SIZE 100

#define ACTION_ENTER 10
#define ACTION_NOOP 11

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

    Stack* stack;
    //bool attackers[BOARD_SIZE];
    //bool defenders[BOARD_SIZE][BOARD_SIZE];
    int block_idx;
    int turn;
};

void allocate_tcg(TCG* env) {
    env->stack = calloc(1, sizeof(Stack));
    env->my_hand = allocate_card_array(HAND_SIZE);
    env->op_hand = allocate_card_array(HAND_SIZE);
    env->my_board = allocate_card_array(BOARD_SIZE);
    env->op_board = allocate_card_array(BOARD_SIZE);
    env->my_deck = allocate_card_array(DECK_SIZE);
    env->op_deck = allocate_card_array(DECK_SIZE);
}

void free_tcg(TCG* env) {
    free_card_array(env->my_hand);
    free_card_array(env->op_hand);
    free_card_array(env->my_board);
    free_card_array(env->op_board);
    free_card_array(env->my_deck);
    free_card_array(env->op_deck);
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

void draw_card(TCG* env, CardArray* deck, CardArray* hand) {
    if (deck->length == 0) {
        reset(env);
        return;
    }
    if (hand->length == hand->max) {
        return;
    }
    Card card = deck->cards[deck->length - 1];
    hand->cards[hand->length] = card;
    deck->length -= 1;
    hand->length += 1;
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
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;
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
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;

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
    printf("PHASE_UNTAP\n");
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;
    *land_played = false;

    env->turn = 1 - env->turn;
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;

    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
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
    printf("PHASE_DRAW\n");
    CardArray* deck = (env->turn == 0) ? env->my_deck : env->op_deck;
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    draw_card(env, deck, hand);
    push(env->stack, phase_play);
    return TO_STACK;
}

bool phase_play(TCG* env, unsigned char atn) {
    printf("PHASE_PLAY\n");
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;

    if (board->length == BOARD_SIZE) {
        printf("\t Board full\n");
        push(env->stack, phase_attack);
        return TO_STACK;
    }

    if (!can_play(env)) {
        printf("\t No valid moves\n");
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
        printf("\t Invalid action: %i\n. Hand length: %i\n", atn, hand->length);
        push(env->stack, phase_play);
        return TO_USER;
    }

    Card card = hand->cards[atn];
    if (card.is_land) {
        if (*land_played) {
            printf("\t Already played land this turn\n");
            push(env->stack, phase_play);
            return TO_USER;
        }
        board->cards[board->length] = card;
        board->length += 1;
        *land_played = true;
        hand->cards[atn].remove = true;
        condense_card_array(hand);
        printf("\t Land played\n");
        push(env->stack, phase_play);
        return TO_USER;
    }

    if (card.cost > *mana + tappable_mana(env)) {
        printf("\t Not enough mana\n");
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
    printf("\t Card played\n");
    push(env->stack, phase_play);
    return TO_USER;
}

bool phase_attack(TCG* env, unsigned char atn) {
    printf("PHASE_ATTACK\n");
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;

    if (!can_attack(board)) {
        printf("\t No valid attacks. Phase end\n");
        push(env->stack, phase_untap);
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_attack);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Attacks confirmed. Phase end\n");
        env->turn = 1 - env->turn;
        push(env->stack, phase_block);
        return TO_STACK;
    } else if (atn >= board->length) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, phase_attack);
        return TO_USER;
    } else if (board->cards[atn].is_land) {
        printf("\t Cannot attack with land\n");
        push(env->stack, phase_attack);
        return TO_USER;
    } else {
        printf("\t Setting attacker %i\n", atn);
        board->cards[atn].attacking = !board->cards[atn].attacking;
        push(env->stack, phase_attack);
        return TO_USER;
    }
}

bool phase_block(TCG* env, unsigned char atn) {
    printf("PHASE_BLOCK\n");
    CardArray* defender_board = (env->turn == 0) ? env->my_board : env->op_board;
    CardArray* board = (env->turn == 0) ? env->op_board : env->my_board;
    int* health = (env->turn == 0) ? &env->op_health : &env->my_health;

    while (env->block_idx < board->length && !board->cards[env->block_idx].attacking) {
        printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
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
            printf("\t Can block with %i\n", i);
            break;
        }
    }
    if (!can_block) {
        env->block_idx = board->length;
    }
 
    if (env->block_idx == board->length) {
        printf("\t Attacker board length: %i\n", board->length);
        for (int atk = 0; atk < board->length; atk++) {
            printf("\t Resolving %i\n", atk);
            Card* attacker = &board->cards[atk];
            if (!attacker->attacking) {
                printf("\t Not attacking\n");
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
            printf("\t Reducing health by %i\n", attacker_attack);
            *health -= attacker_attack;
        }

        if (*health <= 0) {
            printf("\t Game over\n");
            reset(env);
        }

        condense_card_array(env->my_board);
        condense_card_array(env->op_board);

        CardArray* defender_deck = (env->turn == 0) ? env->my_deck : env->op_deck;
        CardArray* defender_hand = (env->turn == 0) ? env->my_hand : env->op_hand;
        draw_card(env, defender_deck, defender_hand);

        for (int i = 0; i < board->length; i++) {
            board->cards[i].attacking = false;
        }
        for (int i = 0; i < defender_board->length; i++) {
            defender_board->cards[i].defending = -1;
        }
        printf("\t Set block idx to 0\n");
        env->block_idx = 0;
        env->turn = 1 - env->turn;
        push(env->stack, phase_untap);
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_block);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual block confirm %i\n", env->block_idx);
        env->block_idx++;
        push(env->stack, phase_block);
        return TO_STACK;
    } else if (atn >= defender_board->length) {
        printf("\t Invalid block action %i\n", atn);
        push(env->stack, phase_block);
        return TO_USER;
    } else if (defender_board->cards[atn].is_land) {
        printf("\t Cannot block with land\n");
        push(env->stack, phase_block);
        return TO_USER;
    }

    for (int i = 0; i < env->block_idx; i++) {
        if (defender_board->cards[atn].defending == i) {
            printf("\t Already blocked\n");
            push(env->stack, phase_block);
            return TO_USER;
        }
    }
    printf("\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = &defender_board->cards[atn];
    if (card->defending == env->block_idx) {
        card->defending = -1;
    } else {
        card->defending = env->block_idx;
    }
    push(env->stack, phase_block);
    return TO_USER;
}

void step(TCG* env, unsigned char atn) {
    printf("Turn: %i, Action: %i\n", env->turn, atn);
    while (true) {
        call fn = pop(env->stack);
        bool return_to_user = fn(env, atn);
        if (return_to_user) {
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
    randomize_deck(env->my_deck);
    randomize_deck(env->op_deck);
    env->turn = rand() % 2;
    for (int i = 0; i < 5; i++) {
        draw_card(env, env->my_deck, env->my_hand);
        draw_card(env, env->op_deck, env->op_hand);
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
        if (env->turn == 0) {
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
        if (env->turn == 0) {
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
