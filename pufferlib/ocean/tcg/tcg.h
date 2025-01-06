#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define HAND_SIZE 10
// TODO: Fix land and board sizes
#define BOARD_SIZE 6
#define LAND_SIZE 4
#define DECK_SIZE 60
#define GRAVEYARD_SIZE 60
#define STACK_SIZE 100

#define LAND_ZONE_OFFSET 128
#define GRAVEYARD_ZONE_OFFSET 128

#define ACTION_ENTER 10
#define ACTION_NOOP 11

#define TO_USER true;
#define TO_STACK false;

#define TAPPED_COLOR (Color){128, 0, 0, 255}
#define CREATURE_COLOR RED
#define CREATURE_SUMMONING_SICK_COLOR ORANGE
#define SPELL_COLOR BLUE
#define INSTANT_COLOR GREEN
#define LAND_COLOR YELLOW

#define GRAVEYARD_COLOR GRAY

typedef struct TCG TCG;
typedef bool (*call)(TCG*, unsigned char);
bool phase_untap(TCG* env, unsigned char atn);
bool phase_upkeep(TCG* env, unsigned char atn);
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

typedef enum {
    TYPE_LAND,
    TYPE_CREATURE,
    TYPE_INSTANT,
    TYPE_SORCERY,
    TYPE_ARTIFACT,
    TYPE_ENCHANTMENT,
    TYPE_PLANESWALKER
} CardType;

typedef struct Card Card;
typedef struct Effect Effect;

struct Effect {
    char description[256];
    void (*trigger)(TCG* env, Card* source);
    bool (*condition)(TCG* env, Card* source);
    void (*activate)(TCG* env, Card* source);
};

struct Card {
    CardType type;
    int cost;
    bool tapped;
    bool remove;
    Color color;
    bool is_permanent;

    // Type-specific data
    union {
        struct { Effect effect; int attack, health; bool summoning_sickness, attacking; int defending; } creature;
        struct { Effect effect; } spell;
    } data;
};

typedef struct CardArray CardArray;
struct CardArray {
    Card* cards;
    int length;
    int max;
};

Card* allocate_creature(int cost, int attack, int health, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_CREATURE;
    card->cost = cost;
    card->tapped = false;
    card->remove = false;
    card->color = CREATURE_COLOR;
    card->is_permanent = true;
    card->data.creature.effect = effect;
    card->data.creature.attack = attack;
    card->data.creature.health = health;
    card->data.creature.summoning_sickness = true;
    card->data.creature.attacking = false;
    card->data.creature.defending = -1;
    return card;
}

Card* allocate_instant(int cost, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_INSTANT;
    card->cost = cost;
    card->tapped = false;
    card->remove = false;
    card->color = INSTANT_COLOR;
    card->is_permanent = false;
    card->data.spell.effect = effect;
    return card;
}

Card* allocate_land() {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_LAND;
    card->cost = 0;
    card->tapped = false;
    card->remove = false;
    card->color = LAND_COLOR;
    card->is_permanent = true;
    return card;
}

CardArray* allocate_card_array(int max) {
    CardArray* hand = (CardArray*)calloc(1, sizeof(CardArray));
    hand->cards = (Card*)calloc(max, sizeof(Card));
    hand->max = max;
    return hand;
}

void free_card_array(CardArray* array) {
    free(array->cards);
    free(array);
}

void condense_card_array(CardArray* array) {
    int idx = 0;
    for (int i = 0; i < array->length; i++) {
        if (!array->cards[i].remove) {
            array->cards[idx] = array->cards[i];
            idx += 1;
        }
    }
    array->length = idx;
}

struct TCG {
    CardArray* my_hand;
    CardArray* my_board;
    CardArray* my_lands;
    CardArray* my_deck;
    CardArray* my_graveyard;
    int my_health;
    int my_mana;
    bool my_land_played;

    CardArray* op_hand;
    CardArray* op_board;
    CardArray* op_lands;
    CardArray* op_deck;
    CardArray* op_graveyard;
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
    env->my_lands = allocate_card_array(LAND_SIZE);
    env->op_lands = allocate_card_array(LAND_SIZE);
    env->my_deck = allocate_card_array(DECK_SIZE);
    env->op_deck = allocate_card_array(DECK_SIZE);
    env->my_graveyard = allocate_card_array(GRAVEYARD_SIZE);
    env->op_graveyard = allocate_card_array(GRAVEYARD_SIZE);
}

void free_tcg(TCG* env) {
    free_card_array(env->my_hand);
    free_card_array(env->op_hand);
    free_card_array(env->my_board);
    free_card_array(env->op_board);
    free_card_array(env->my_lands);
    free_card_array(env->op_lands);
    free_card_array(env->my_deck);
    free_card_array(env->op_deck);
    free_card_array(env->my_graveyard);
    free_card_array(env->op_graveyard);
}

// TODO: Actually randomize the deck based on how many cards of each type are in the deck
void randomize_deck(CardArray* deck) {
    for (int i = 0; i < deck->length; i++) {
        int type = rand() % 2;
        if (type == 0) {
            deck->cards[i] = *allocate_land();
        } else if (type == 1) {
            int cost = rand() % 6;
            Effect effect = {0};
            deck->cards[i] = *allocate_creature(cost, cost + 1, cost + 1, effect);
        } else if (type == 2) {
            int cost = rand() % 6;
            Effect effect = {0};
            deck->cards[i] = *allocate_instant(cost, effect);
        } 
    }
}

void add_card_to(TCG* env, Card card, CardArray* to) {
    to->cards[to->length] = card;
    to->length += 1;
}

void remove_card_from(TCG* env, CardArray* from, int idx) {
    for (int i = idx; i < from->length - 1; i++) {
        from->cards[i] = from->cards[i + 1];
    }
    from->length -= 1;
}

void move_card(TCG* env, CardArray* from, CardArray* to, int idx) {
    Card card = from->cards[idx];
    remove_card_from(env, from, idx);
    add_card_to(env, card, to);
}

void draw_card(TCG* env, CardArray* deck, CardArray* hand) {
    if (deck->length == 0) {
        reset(env);
        printf("Deck empty. You lose.\n");
        return;
    }
    if (hand->length == hand->max) {
        printf("Hand full. Skipping draw.\n");
        return;
    }
    move_card(env, deck, hand, deck->length - 1);
}


int tappable_mana(TCG* env) {
    CardArray* lands = (env->turn == 0) ? env->my_lands : env->op_lands;
    int tappable = 0;
    for (int i = 0; i < lands->length; i++) {
        Card card = lands->cards[i];
        if (!card.tapped) {
            tappable += 1;
        }
    }
    return tappable;
}

bool has_valid_moves(TCG* env) {
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;

    int min_cost = 99;
    for (int i = 0; i < hand->length; i++) {
        Card card = hand->cards[i];
        if (card.type == TYPE_LAND && !*land_played) {
            return true;
        } else if (card.cost < min_cost) {
            min_cost = card.cost;
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
    CardArray* lands = (env->turn == 0) ? env->my_lands : env->op_lands;

    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    *mana = 0;

    printf("Untapping board\n");
    for (int i = 0; i < board->length; i++) {
        Card card = board->cards[i];
        if (card.tapped) {
            board->cards[i].tapped = false;
        }
    }

    printf("Untapping lands\n");
    for (int i = 0; i < lands->length; i++) {
        Card card = lands->cards[i];
        if (card.tapped) {
            lands->cards[i].tapped = false;
        }
    }
    
    push(env->stack, phase_upkeep);
    return TO_STACK;
}

bool phase_upkeep(TCG* env, unsigned char atn) {
    printf("PHASE_UPKEEP\n");
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;
    for (int i = 0; i < board->length; i++) {
         Card card = board->cards[i];
         if (card.type == TYPE_CREATURE && card.data.creature.summoning_sickness) {
             board->cards[i].data.creature.summoning_sickness = false;
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
    CardArray* lands = (env->turn == 0) ? env->my_lands : env->op_lands;
    CardArray* graveyard = (env->turn == 0) ? env->my_graveyard : env->op_graveyard;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;

    if (board->length == BOARD_SIZE && lands->length == LAND_SIZE) {
        printf("\t Board and lands full\n");
        push(env->stack, phase_attack);
        return TO_STACK;
    }

    if (!has_valid_moves(env)) {
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
    if (card.type == TYPE_LAND) {
        if (*land_played) {
            printf("\t Already played land this turn\n");
            push(env->stack, phase_play);
            return TO_USER;
        }
        move_card(env, hand, lands, atn);
        *land_played = true;
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
    for (int i = 0; i < lands->length; i++) {
        if (card.cost <= *mana) {
            break;
        }
        if (!lands->cards[i].tapped) {
            *mana += 1;
            lands->cards[i].tapped = true;
        }
    }

    assert(*mana >= card.cost);
    *mana -= card.cost;
    move_card(env, hand, board, atn);
    /*card = board->cards[atn];*/
    if (card.type == TYPE_INSTANT) {
        board->cards[board->length].data.spell.effect.activate(env, &board->cards[board->length]);
    }
    if (!(card.is_permanent)) {
        printf("\t Removing non-permanent card\n");

        move_card(env, board, graveyard, atn);
    } 
    printf("\t Card played\n");
    push(env->stack, phase_play);
    return TO_USER;
}

bool phase_attack(TCG* env, unsigned char atn) {
    printf("PHASE_ATTACK\n");
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;

    if (atn == ACTION_NOOP) {
        push(env->stack, phase_attack);
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Attacks confirmed. Phase end\n");
        env->turn = 1 - env->turn;
        push(env->stack, phase_block);
        return TO_STACK;
    } else if (atn >= board->length || board->cards[atn].type != TYPE_CREATURE) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, phase_attack);
        return TO_USER;
    } else {
        if (board->cards[atn].data.creature.summoning_sickness) {
            printf("\t Cannot attack with summoning sickness\n");
            push(env->stack, phase_attack);
            return TO_USER;
        }
        printf("\t Setting attacker %i\n", atn);
        board->cards[atn].data.creature.attacking = !board->cards[atn].data.creature.attacking;
        board->cards[atn].tapped = !board->cards[atn].tapped;
        push(env->stack, phase_attack);
        return TO_USER;
    }
}



// Function to handle the blocking phase in the game
bool phase_block(TCG* env, unsigned char atn) {
    printf("PHASE_BLOCK\n");

    // Determine the board, graveyard, and health of the current player and opponent
    CardArray* defender_board = (env->turn == 0) ? env->my_board : env->op_board;
    CardArray* board = (env->turn == 0) ? env->op_board : env->my_board;
    CardArray* defender_graveyard = (env->turn == 0) ? env->my_graveyard : env->op_graveyard;
    CardArray* graveyard = (env->turn == 0) ? env->op_graveyard : env->my_graveyard;
    int* health = (env->turn == 0) ? &env->op_health : &env->my_health;

    // Skip non-attacking cards
    while (env->block_idx < board->length && !board->cards[env->block_idx].data.creature.attacking) {
        printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
        env->block_idx++;
    }

    bool can_block = false;
    for (int i = 0; i < defender_board->length; i++) {
        Card card = defender_board->cards[i];
        // If card is not defending yet or already defending the current attacker
        if (card.type == TYPE_CREATURE && !card.tapped && (card.data.creature.defending == -1 || card.data.creature.defending == env->block_idx)) {
            can_block = true;
            printf("\t Can block with %i\n", i);
            break;
        }
    }
    if (!can_block) {
        env->block_idx = board->length;
    }

    // If no more defenders are available, resolve attacks
    if (env->block_idx == board->length) {
        printf("\t Attacker board length: %i\n", board->length);

        for (int atk = 0; atk < board->length; atk++) {
            printf("\t Resolving %i\n", atk);
            Card* attacker = &board->cards[atk];

            // Skip non-creature or non-attacking cards
            if (attacker->type != TYPE_CREATURE) {
                printf("\t Not attacking because not a creature\n");
                continue;
            }
            if (!attacker->data.creature.attacking) {
                printf("\t Not attacking\n");
                continue;
            }

            int attacker_attack = attacker->data.creature.attack;
            int attacker_health = attacker->data.creature.health;

            // Resolve combat with each defender
            for (int def = 0; def < defender_board->length; def++) {
                Card* defender = &defender_board->cards[def];
                if (defender->type != TYPE_CREATURE) {
                    printf("\t Not a creature\n");
                    continue;
                }
                if (defender->data.creature.defending != atk) {
                    printf("\t Not defending this attacker\n");
                    continue;
                }

                int defender_attack = defender->data.creature.attack;
                int defender_health = defender->data.creature.health;

                // Resolve damage between attacker and defender
                if (attacker_attack >= defender_health) {
                    attacker_attack -= defender_health;
                    attacker_health -= defender_attack;

                    defender->data.creature.health = 0;
                    move_card(env, defender_board, defender_graveyard, def); // Move defender to graveyard
                } else {
                    attacker_health -= defender_attack;
                    attacker_attack = 0;
                }
                // Check if attacker is defeated
                if (attacker_health <= 0) {
                    move_card(env, board, graveyard, atk); // Move attacker to graveyard
                    break;
                }
            }

            // Reduce player health if attack goes through
            printf("\t Reducing health by %i\n", attacker_attack);
            *health -= attacker_attack;
        }

        // Handle end of turn logic
        if (*health <= 0) {
            printf("\t Game over\n");
            reset(env); // Reset the game
        }

        // Draw a card for the defender and reset states
        CardArray* defender_deck = (env->turn == 0) ? env->my_deck : env->op_deck;
        CardArray* defender_hand = (env->turn == 0) ? env->my_hand : env->op_hand;
        draw_card(env, defender_deck, defender_hand);

        // Reset attacking and defending statuses
        for (int i = 0; i < board->length; i++) {
            if (board->cards[i].type == TYPE_CREATURE) {
                board->cards[i].data.creature.attacking = false;
            }
        }
        for (int i = 0; i < defender_board->length; i++) {
            if (defender_board->cards[i].type == TYPE_CREATURE) {
                defender_board->cards[i].data.creature.defending = -1;
            }
        }

        printf("\t Set block idx to 0\n");
        env->block_idx = 0;
        env->turn = 1 - env->turn; // Switch turn
        push(env->stack, phase_untap); // Move to the next phase
        return TO_STACK;
    }

    // Handle player actions during blocking phase
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
    }

    // Validate the selected defender card for blocking
    for (int i = 0; i < env->block_idx; i++) {
        if (defender_board->cards[atn].data.creature.defending == i) {
            printf("\t Already blocked\n");
            push(env->stack, phase_block);
            return TO_USER;
        }
        if (defender_board->cards[atn].tapped) {
            printf("\t Cannot block with tapped card\n");
            push(env->stack, phase_block);
            return TO_USER;
        }
        if (defender_board->cards[atn].type != TYPE_CREATURE) {
            printf("\t Cannot block with non-creature card\n");
            push(env->stack, phase_block);
            return TO_USER;
        }
    }

    // Set the defending state for the selected card
    printf("\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = &defender_board->cards[atn];
    if (card->data.creature.defending == env->block_idx) {
        card->data.creature.defending = -1;
    } else {
        card->data.creature.defending = env->block_idx;
    }
    push(env->stack, phase_block); // Repeat the blocking phase
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
    env->my_lands->length = 0;
    env->op_lands->length = 0;
    env->my_graveyard->length = 0;
    env->op_graveyard->length = 0;
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
    if (card->type == TYPE_LAND) {
        DrawText("Land", x + 16, y + 40, 16, WHITE);
    } else if (card->type == TYPE_CREATURE) {
        DrawText(TextFormat("%i", card->cost), x + 32, y+16, 20, WHITE);
        DrawText(TextFormat("%i", card->data.creature.attack), x + 32, y + 40, 20, WHITE);
        DrawText(TextFormat("%i", card->data.creature.health), x + 32, y + 64, 20, WHITE);
    } else {
        printf("Invalid card type: %i\n", card->type);
    }
}

void render_label(int x, int y, int idx) {
    DrawText(TextFormat("%i", (idx+1)%10), x+32, y+96, 20, YELLOW);
}

void render_my_hand(TCG* env) {
    for (int i = 0; i < env->my_hand->length; i++) {
        Card card = env->my_hand->cards[i];
        int x = card_x(i, env->my_hand->length);
        int y = card_y(3);
        render_card(&card, x, y, RED);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }
}

void render_op_hand(TCG* env) {
    for (int i = 0; i < env->op_hand->length; i++) {
        Card card = env->op_hand->cards[i];
        int x = card_x(i, env->op_hand->length);
        int y = card_y(0);
        render_card(&card, x, y, BLUE);
    }
}

void render_my_board(TCG* env) {
    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];

        if ((card.type != TYPE_CREATURE)) {
            continue;
        }

        int x = card_x(i, env->my_board->length);
        int y = card_y(2);
        if (card.data.creature.attacking) {
            y -= 16;
        }
        Color color = (card.tapped) ? (Color){128, 0, 0, 255}: card.data.creature.summoning_sickness ? ORANGE : RED;
        render_card(&card, x, y, color);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];
        if (card.data.creature.defending == -1) {
            continue;
        }
        DrawLineEx(
            (Vector2){32+card_x(i, env->my_board->length), 64+card_y(2)},
            (Vector2){32+card_x(card.data.creature.defending, env->op_board->length), 64+card_y(1)},
            3.0f, WHITE
        );
    }
}

void render_op_board(TCG* env) {
    for (int i = 0; i < env->op_board->length; i++) {
        if ((env->op_board->cards[i].type != TYPE_CREATURE)) {
            continue;
        }

        Card card = env->op_board->cards[i];
        if (card.type != TYPE_CREATURE) {
            continue;
        }
        int x = card_x(i, env->op_board->length);
        int y = card_y(1);
        if (card.data.creature.attacking) {
            y += 16;
        }
        Color color = (card.tapped) ? (Color){128, 0, 0, 255}: card.data.creature.summoning_sickness ? SKYBLUE : BLUE;
        render_card(&card, x, y, color);
    }
}

void render_my_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->my_lands->length; i++) {
        Card card = env->my_lands->cards[i];
        int x = LAND_ZONE_OFFSET;
        int y = card_y(2);
        assert(card.type == TYPE_LAND);
        render_card(&card, x, y, RED);
        if (!card.tapped) {
            num_untapped += 1;
        }
    }
    if (env->my_lands->length > 0) {
        int x = LAND_ZONE_OFFSET + 4;
        int y = card_y(2) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, WHITE);
    }
}

void render_op_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->op_lands->length; i++) {
        Card card = env->op_lands->cards[i];
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET);
        int y = card_y(1);
        render_card(&card, x, y, BLUE);
        if (!card.tapped) {
            num_untapped += 1;
        }
    }
    if (env->op_lands->length > 0) {
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET) + 4;
        int y = card_y(1) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, YELLOW);
    }
}

void render_my_graveyard(TCG* env) {
    for (int i = 0; i < env->my_graveyard->length; i++) {
        Card card = env->my_graveyard->cards[i];
        int x = (GetScreenWidth() - GRAVEYARD_ZONE_OFFSET);
        int y = card_y(2);
        render_card(&card, x, y, GRAVEYARD_COLOR);
    }
}

void render_op_graveyard(TCG* env) {
    for (int i = 0; i < env->op_graveyard->length; i++) {
        Card card = env->op_graveyard->cards[i];
        int x = GRAVEYARD_ZONE_OFFSET;
        int y = card_y(1);
        render_card(&card, x, y, GRAVEYARD_COLOR);
    }
}

void render(TCG* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
   

    render_my_hand(env);

    render_my_board(env);
    render_op_board(env);

    render_my_lands(env);
    render_op_lands(env);

    render_my_graveyard(env);
    render_op_graveyard(env);

    render_op_hand(env);


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
